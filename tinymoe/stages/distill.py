"""Stage 6 - distillation: the lineage model teaches a smaller student (``student`` config).

Every student mode starts from one shared base (``student_pretrain_steps`` of next-token
training on the pre-training mixture) and gets the same optimizer-step budget, so the
modes form an ablation ladder over *what the student imitates*:

* ``baseline``  - SFT on the ground-truth SFT data. No teacher; the reference point.
* ``offpolicy`` - sequence-level KD (Kim & Rush, 2016): SFT on teacher samples, keeping
  math answers only when the verifier accepts them (rejection sampling, as used for the
  DeepSeek-R1 distilled models).
* ``logit``     - word-level KD (Hinton et al., 2015) on the same teacher samples:
  ``tau^2 * KL(p_T^tau || p_S^tau)`` over response tokens (+ ``kd_ce_weight`` * CE). The
  full distribution carries "dark knowledge" a single sampled token does not.
* ``onpolicy``  - on-policy distillation (GKD, Agarwal et al., 2024; MiniLLM, Gu et al.,
  2024): the student samples, the teacher scores every position, and the loss is the
  full-vocabulary reverse KL ``KL(p_S || p_T)`` on the student's own prefixes. Training on
  the states the student actually visits removes the train/test mismatch (exposure bias)
  of the off-policy modes, and reverse KL is mode-seeking: a small student commits to the
  teacher's main mode instead of smearing mass across modes it cannot represent. As GKD
  recommends, it starts from the ``offpolicy`` student when that mode ran. The gradient
  treats the sampled prefixes as fixed (no gradient through the sampling distribution).

Teacher and student share the tokenizer, so the vocabularies line up for the KL terms.
Teacher samples are generated once on the main rank into ``distill/teacher_samples.json``
(reused on rerun). Results: ``artifacts/distill_comparison.json`` (rows + ``markdown``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tinymoe.config import BatchPlan, RunConfig, derive_batch
from tinymoe.data import mathworld, stories
from tinymoe.data.batch import IGNORE_INDEX, Batch
from tinymoe.data.sft_data import (
    RowDataset,
    build_sft_examples,
    encode,
    rows_from_encoded,
    story_instruction,
)
from tinymoe.data.tokenizer import Tokenizer, stop_ids
from tinymoe.eval import probes
from tinymoe.eval.evaluator import Evaluator
from tinymoe.model.transformer import Transformer
from tinymoe.rl.rollout import pad_completions
from tinymoe.stages.common import StageContext, stage_main
from tinymoe.stages.pretrain import mixture
from tinymoe.train import distributed
from tinymoe.train.loop import TrainResult
from tinymoe.train.losses import LossOutput, forward_batch, masked_mean, token_ce, token_kl

log = logging.getLogger(__name__)

Encoded = tuple[list[int], list[int]]  # (token ids, loss mask)
STORY_EVERY = 4  # one story instruction per this many prompts (teacher data, rollouts)
MODES = ("baseline", "offpolicy", "logit", "onpolicy")
COMPARE = (  # (metric, column) for the comparison table
    ("eval/math/pass1", "math pass@1"),
    ("eval/math/format", "format"),
    ("eval/story/mention", "story mention"),
    ("val/loss/sft", "held-out SFT loss"),
)


@dataclass(frozen=True)
class Prompt:
    ids: list[int]
    kind: str  # "math" | "story"
    answer: int | None = None


def distill_prompts(cfg: RunConfig, tok: Tokenizer, n: int, salt: str) -> list[Prompt]:
    """``n`` chat prompts: MathWorld problems cycling through every bucket, with every
    ``STORY_EVERY``-th prompt a story instruction (the SFT mix, roughly)."""
    mw = cfg.data.mathworld
    buckets = mathworld.buckets_from(mw)
    out = []
    for i in range(n):
        if i % STORY_EVERY == STORY_EVERY - 1:
            subject = stories.SUBJECTS[(i // STORY_EVERY) % len(stories.SUBJECTS)]
            ids = probes.chat_prompt_ids(tok, story_instruction(subject))
            out.append(Prompt(ids, "story"))
        else:
            rng = mathworld.rng_for(cfg.run.seed, salt, i)
            p = mathworld.make_problem(rng, buckets[i % len(buckets)], mw)
            out.append(Prompt(probes.math_prompt_ids(tok, p.question, "chat"), "math", p.answer))
    return out


# ----------------------------------------------------------------- teacher data
@dataclass(frozen=True)
class TeacherSample:
    prompt: list[int]
    completion: list[int]
    kind: str
    ok: bool  # finished (stop token) and, for math, verified correct


def select_teacher_samples(samples: list[TeacherSample]) -> tuple[list[dict[str, Any]], bool]:
    """Rejection sampling: deduplicated samples that finished and (math) are correct.

    Returns ``(kept, filtered)``. A teacher too weak to finish any reply (an early or tiny
    model, as in the smoke pipeline) would otherwise leave the off-policy modes with no
    data at all. In that case every non-empty sample is kept instead (``filtered`` is
    False, and the caller logs the fallback): plain sequence-level KD (Kim & Rush, 2016)
    without the rejection step.
    """
    seen: set[tuple[int, ...]] = set()
    unique = []
    for s in samples:
        key = (*s.prompt, *s.completion)
        if s.completion and key not in seen:
            seen.add(key)
            unique.append(s)
    kept = [s for s in unique if s.ok]
    filtered = bool(kept)
    chosen = kept if filtered else unique
    return [
        {"prompt": s.prompt, "completion": s.completion, "kind": s.kind} for s in chosen
    ], filtered


def generate_teacher_samples(
    ctx: StageContext, teacher: Transformer, evaluator: Evaluator
) -> dict[str, Any]:
    """Sample the teacher and keep finished replies (and verified-correct math), deduped."""
    d, e = ctx.cfg.distill, ctx.cfg.eval
    prompts = distill_prompts(ctx.cfg, ctx.tok, d.teacher_prompts, "distill-teacher")
    stops = set(stop_ids(ctx.tok))
    samples: list[TeacherSample] = []
    accept: dict[str, float] = {}
    for kind, max_new in (("math", e.max_new_tokens), ("story", e.story_max_new_tokens)):
        group = [p for p in prompts if p.kind == kind for _ in range(d.teacher_samples_per_prompt)]
        if not group:
            continue
        g = torch.Generator().manual_seed(ctx.cfg.run.seed + (29 if kind == "math" else 31))
        gen = evaluator.generate(teacher, [p.ids for p in group], max_new, d.teacher_temperature, g)
        n_ok = 0
        for p, seq in zip(group, gen.sequences, strict=True):
            ok = bool(seq) and seq[-1] in stops
            if ok and p.answer is not None:
                ok = evaluator.judge(evaluator.response_text(seq), p.answer)[0]
            samples.append(TeacherSample(p.ids, list(seq), kind, ok))
            n_ok += ok
        accept[kind] = n_ok / len(group)
    kept, filtered = select_teacher_samples(samples)
    if not filtered:
        log.warning(
            "[distill] fallback: no teacher sample finished and verified (accept %s); "
            "training the off-policy modes on all %d unfiltered samples",
            accept,
            len(kept),
        )
    return {"samples": kept, "accept": accept, "filtered": filtered}


def teacher_data(
    ctx: StageContext, teacher: Transformer, evaluator: Evaluator
) -> tuple[list[Encoded], dict[str, float]]:
    """Teacher samples as ``(ids, mask)`` pairs: generated once on the main rank, shared
    through the run directory (and reused if the stage is rerun)."""
    path = ctx.run_dir / "distill" / "teacher_samples.json"
    if ctx.dist.is_main and not path.is_file():
        data = generate_teacher_samples(ctx, teacher, evaluator)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    distributed.barrier()
    data = json.loads(Path(path).read_text())
    encoded = [
        ([*s["prompt"], *s["completion"]], [0] * len(s["prompt"]) + [1] * len(s["completion"]))
        for s in data["samples"]
    ]
    return encoded, data["accept"]


# ----------------------------------------------------------------------- losses
def logit_kd_loss(
    teacher: Transformer, temperature: float, ce_weight: float
) -> Callable[[nn.Module, Batch], LossOutput]:
    """Hinton KD: forward KL from the teacher at temperature ``tau``, scaled by ``tau^2`` so
    gradient magnitudes stay comparable across temperatures."""

    def loss_fn(m: nn.Module, batch: Batch) -> LossOutput:
        out = forward_batch(m, batch)
        with torch.no_grad():
            t_logits = forward_batch(teacher, batch).logits
        mask = (batch.labels != IGNORE_INDEX).float()
        kl = masked_mean(token_kl(t_logits, out.logits, temperature), mask) * temperature**2
        ce = masked_mean(token_ce(out.logits, batch.labels), mask)
        loss = kl + ce_weight * ce + out.aux_loss
        return LossOutput(loss, {"loss": loss.detach(), "kd/kl": kl.detach(), "kd/ce": ce.detach()})

    return loss_fn


def reverse_kl_loss(teacher: Transformer) -> Callable[[nn.Module, Batch], LossOutput]:
    """On-policy distillation: per-position ``KL(p_S || p_T)`` over the full vocabulary on
    student-sampled completions. ``kd/teacher_nll`` (the teacher's NLL of the student's
    tokens) shows the samples becoming more teacher-like."""

    def loss_fn(m: nn.Module, batch: Batch) -> LossOutput:
        out = forward_batch(m, batch)
        with torch.no_grad():
            t_logits = forward_batch(teacher, batch).logits
        mask = (batch.labels != IGNORE_INDEX).float()
        rkl = masked_mean(token_kl(out.logits, t_logits), mask)
        with torch.no_grad():
            nll = masked_mean(token_ce(t_logits, batch.labels), mask)
        metrics = {"loss": rkl.detach(), "kd/teacher_nll": nll, "kd/tokens": mask.sum()}
        return LossOutput(rkl + out.aux_loss, metrics)

    return loss_fn


class OnPolicyRollouts:
    """Provider: each optimizer step the student samples completions for this rank's share
    of fresh prompts (when the step's first micro batch is requested). A step's rollout
    depends only on (seed, step, rank) and the weights, which resume restores, so the
    provider keeps no state of its own."""

    def __init__(
        self,
        student: Transformer,
        evaluator: Evaluator,
        prompts: Sequence[Prompt],
        plan: BatchPlan,
        rank: int,
        seed: int,
        max_new: int,
        temperature: float,
        pad_id: int,
        device: torch.device,
    ) -> None:
        self.student = student
        self.evaluator = evaluator
        self.prompts = prompts
        self.plan = plan
        self.rank = rank
        self.seed = seed
        self.max_new = max_new
        self.temperature = temperature
        self.pad_id = pad_id
        self.device = device
        self._step = -1
        self._rows: tuple[torch.Tensor, torch.Tensor] | None = None

    def _rollout(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_rank = self.plan.micro_batch_size * self.plan.grad_accum
        start = (step * self.plan.world_size + self.rank) * per_rank
        chosen = [self.prompts[(start + i) % len(self.prompts)].ids for i in range(per_rank)]
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + start)
        was_training = self.student.training
        self.student.eval()
        gen = self.evaluator.generate(self.student, chosen, self.max_new, self.temperature, g)
        self.student.train(was_training)
        return pad_completions(chosen, gen.sequences, self.pad_id)

    def batch(self, step: int, micro: int) -> Batch:
        if self._step != step or self._rows is None:
            self._rows = self._rollout(step)
            self._step = step
        ids, labels = self._rows
        rows = slice(micro * self.plan.micro_batch_size, (micro + 1) * self.plan.micro_batch_size)
        return Batch(ids[rows], labels[rows]).to(self.device)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


# ------------------------------------------------------------------------ stage
@dataclass
class ModeResult:
    mode: str
    init: str
    data: str
    objective: str
    steps: tuple[int, int]
    metrics: dict[str, float]
    checkpoint: Path | None


class Distiller:
    """Holds the teacher, the shared student base and the data; runs one mode at a time."""

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx
        cfg = ctx.cfg
        self.d = cfg.distill
        unknown = set(self.d.modes) - set(MODES)
        if unknown:
            raise ValueError(f"distill.modes: unknown {sorted(unknown)}; choose from {MODES}")
        self.teacher = ctx.build_model()
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.evaluator = ctx.evaluator()
        self.student = ctx.build_model("student", init=None)
        self.offset = 0
        self.train = cfg.train
        packing = cfg.sft.packing
        seq, seed, pad = cfg.train.seq_len, cfg.run.seed, ctx.tok.pad_id

        n_story = self.d.teacher_prompts // STORY_EVERY  # same prompt mix as the teacher data
        truth = build_sft_examples(
            cfg.data, self.d.teacher_prompts - n_story, n_story, cfg.sft.think_traces, seed
        )
        rows = rows_from_encoded(encode(ctx.tok, truth, seq), seq, packing, pad, seed)
        n_val = max(int(len(rows) * cfg.data.sft_val_fraction), 1)
        self.truth = RowDataset(rows[n_val:], seed, isolate_docs=packing)
        self.val = RowDataset(rows[:n_val], seed, isolate_docs=packing)
        self.n_val = n_val

        encoded, self.accept = teacher_data(ctx, self.teacher, self.evaluator)
        fitting = [x for x in encoded if len(x[0]) <= seq + 1]
        if not fitting:
            raise RuntimeError("[distill] the teacher produced no usable samples")
        self.teacher_rows = RowDataset(
            rows_from_encoded(fitting, seq, packing, pad, seed), seed, isolate_docs=packing
        )
        log.info(
            "[distill] teacher samples: %d kept (accept %s); ground truth: %d train / %d val rows",
            len(fitting),
            {k: round(v, 3) for k, v in self.accept.items()},
            len(rows) - n_val,
            n_val,
        )

    # ------------------------------------------------------------ helpers
    def evaluate_as(self, model: Transformer, tag: str) -> Callable[[int], dict[str, float]]:
        def evaluate(step: int) -> dict[str, float]:
            m = self.evaluator.run(model, step, tag=tag)
            m.update(self.evaluator.rows_loss(model, self.val, self.n_val, "val/loss/sft"))
            return {f"distill/{tag}/{k}": v for k, v in m.items()}

        return evaluate

    def _advance(self, result: TrainResult) -> tuple[int, int]:
        span = (self.offset, self.offset + result.step)
        self.offset += result.step
        return span

    # -------------------------------------------------------------- phases
    def teacher_metrics(self) -> dict[str, float]:
        metrics = self.evaluate_as(self.teacher, "teacher")(0)
        metrics.update({f"distill/teacher/accept/{k}": v for k, v in self.accept.items()})
        self.ctx.logger.log_metrics(self.ctx.stage, 0, metrics)
        return metrics

    def pretrain_student(self) -> dict[str, torch.Tensor]:
        steps = self.d.student_pretrain_steps
        if steps:
            sampler = mixture(self.ctx, self.ctx.cfg.data.pretrain_mix, self.train.seq_len)
            t = replace(self.train, max_steps=steps, lr=self.d.student_lr)
            log.info("[distill] pre-training the student base for %d steps", steps)
            result = self.ctx.fit_lm(
                self.student,
                sampler,
                sampler.names,
                None,
                train=t,
                ckpt_subdir="student_base",
                step_offset=self.offset,
            )
            self._advance(result)
        return {k: v.detach().clone() for k, v in self.student.state_dict().items()}

    def run_mode(self, mode: str, init: dict[str, torch.Tensor], init_name: str) -> ModeResult:
        ctx, d, t = self.ctx, self.d, self.train
        self.student.load_state_dict(init)
        evaluate = self.evaluate_as(self.student, mode)
        log.info("[distill] mode %s from %s", mode, init_name)
        if mode == "onpolicy":
            result = self._onpolicy(evaluate)
            data, objective = "student samples", "reverse KL(S||T), full vocab"
        else:
            source = self.truth if mode == "baseline" else self.teacher_rows
            loss = (
                logit_kd_loss(self.teacher, d.kd_temperature, d.kd_ce_weight)
                if mode == "logit"
                else None
            )
            result = ctx.fit_lm(
                self.student,
                source,
                None,
                evaluate,
                train=t,
                ckpt_subdir=mode,
                step_offset=self.offset,
                loss_fn=loss,
            )
            data = "ground truth" if mode == "baseline" else "teacher samples"
            objective = (
                f"tau^2 KL(T||S), tau={d.kd_temperature}, +{d.kd_ce_weight} CE"
                if mode == "logit"
                else "CE"
            )
        span = self._advance(result)
        prefix = f"distill/{mode}/"
        metrics = {k[len(prefix) :]: v for k, v in result.eval_metrics.items()}
        return ModeResult(mode, init_name, data, objective, span, metrics, result.checkpoint)

    def _onpolicy(self, evaluate: Callable[[int], dict[str, float]]) -> TrainResult:
        ctx, d = self.ctx, self.d
        n = d.onpolicy_steps * d.onpolicy_prompts_per_step
        prompts = distill_prompts(ctx.cfg, ctx.tok, n, "distill-onpolicy")
        seq = min(max(len(p.ids) for p in prompts) + d.onpolicy_max_new_tokens, self.train.seq_len)
        t = replace(
            self.train,
            max_steps=d.onpolicy_steps,
            lr=d.onpolicy_lr,
            seq_len=seq,
            warmup_steps=min(self.train.warmup_steps, d.onpolicy_steps // 10),
        )
        loss_fn = reverse_kl_loss(self.teacher)
        world = ctx.dist.world_size
        per_rank = d.onpolicy_prompts_per_step // world
        mbs = ctx.micro_batch(self.student, t, probe_loss=lambda m, b: loss_fn(m, b).loss)
        plan = derive_batch(d.onpolicy_prompts_per_step, 1, min(mbs, per_rank), world)
        provider = OnPolicyRollouts(
            self.student,
            self.evaluator,
            prompts,
            plan,
            ctx.dist.rank,
            ctx.cfg.run.seed,
            d.onpolicy_max_new_tokens,
            d.onpolicy_temperature,
            ctx.tok.pad_id,
            ctx.device,
        )
        hooks, bind = ctx.hooks(self.student, evaluate)
        trainer = ctx.trainer(
            self.student,
            plan,
            loss_fn,
            provider,
            hooks,
            train=t,
            ckpt_subdir="onpolicy",
            units_name="sequences",
            step_offset=self.offset,
        )
        bind(trainer)
        log.info("[distill] on-policy: %d steps, plan %s, seq %d", t.max_steps, plan, seq)
        return trainer.fit()


def comparison(teacher: dict[str, float], rows: Sequence[ModeResult]) -> dict[str, Any]:
    """Artifact for the viewer: one row per student plus the teacher, and a markdown table."""

    def cells(metrics: dict[str, float]) -> list[str]:
        return [f"{metrics[k]:.3f}" if k in metrics else "-" for k, _ in COMPARE]

    t_metrics = {k.removeprefix("distill/teacher/"): v for k, v in teacher.items()}
    head = ["model", "init", "data", "objective", "steps", *[c for _, c in COMPARE]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines.append("| teacher | - | - | - | - | " + " | ".join(cells(t_metrics)) + " |")
    for r in rows:
        meta = [r.mode, r.init, r.data, r.objective, f"{r.steps[0]}-{r.steps[1]}"]
        lines.append("| " + " | ".join(meta + cells(r.metrics)) + " |")
    return {
        "teacher": {k: t_metrics[k] for k, _ in COMPARE if k in t_metrics},
        "rows": [
            {
                "mode": r.mode,
                "init": r.init,
                "data": r.data,
                "objective": r.objective,
                "steps": list(r.steps),
                "metrics": {k: r.metrics[k] for k, _ in COMPARE if k in r.metrics},
                "checkpoint": str(r.checkpoint) if r.checkpoint else None,
            }
            for r in rows
        ],
        "columns": [{"key": k, "label": c} for k, c in COMPARE],
        "markdown": "\n".join(lines),
    }


def best(rows: Sequence[ModeResult]) -> ModeResult:
    """Highest math pass@1; ties go to the lower held-out SFT loss."""
    return max(
        rows,
        key=lambda r: (
            r.metrics.get("eval/math/pass1", 0.0),
            -r.metrics.get("val/loss/sft", float(np.inf)),
        ),
    )


def run(ctx: StageContext) -> dict[str, Any]:
    dist = Distiller(ctx)
    teacher = dist.teacher_metrics()
    base = dist.pretrain_student()
    rows: list[ModeResult] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for mode in dist.d.modes:
        init, init_name = base, "student base"
        if mode == "onpolicy" and "offpolicy" in states:
            init, init_name = states["offpolicy"], "offpolicy"
        rows.append(dist.run_mode(mode, init, init_name))
        if mode == "offpolicy":
            states[mode] = {k: v.detach().clone() for k, v in dist.student.state_dict().items()}
    if not rows:
        raise ValueError("distill.modes is empty")
    table = comparison(teacher, rows)
    ctx.logger.write_artifact("distill_comparison.json", table)
    log.info("[distill] comparison\n%s", table["markdown"])
    top = best(rows)
    # The stage's checkpoint is the best student, so its metrics are the stage's summary
    # under the same keys as every other stage (the report table lines stages up by key).
    summary = dict(top.metrics)
    summary["distill/best_index"] = float(MODES.index(top.mode))
    summary.update({k: v for k, v in teacher.items() if k.startswith("distill/teacher/eval/math")})
    return ctx.finish(top.checkpoint, summary)


if __name__ == "__main__":
    stage_main("distill")
