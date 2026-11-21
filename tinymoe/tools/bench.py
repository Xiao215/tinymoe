"""``tinymoe bench <pipeline.yaml>``: measured step time and memory per stage, plus a
wall-clock estimate for the whole pipeline, on random data and the configured hardware.

Every number comes from running the real code paths for a few steps: the
:class:`~tinymoe.train.loop.Trainer` (same autocast, loss scaling, optimizer and MoE
bias updates as a real run), KV-cache ``generate`` for rollouts / eval / teacher
sampling, and a captured forward for snapshot overhead. Stage estimates multiply those
measurements by the step and sample counts in the stage's config, so they are upper
bounds for generation (random weights never emit a stop token).

The memory check compares the largest measured peak (device memory where the backend
reports it, process RSS otherwise) with ``hardware.memory_budget_gb``. The command exits
non-zero if any stage is over budget, so it can gate long runs.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tinymoe import hardware
from tinymoe.config import (
    BatchPlan,
    ModelConfig,
    RunConfig,
    TrainConfig,
    derive_batch,
    to_dict,
)
from tinymoe.data.batch import Batch
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.config import param_count
from tinymoe.model.hooks import Capture
from tinymoe.model.transformer import Transformer, generate
from tinymoe.pipeline import stage_configs
from tinymoe.stages.distill import logit_kd_loss, reverse_kl_loss
from tinymoe.train.autobatch import model_micro_batch
from tinymoe.train.distributed import DistInfo
from tinymoe.train.loop import LossFn, Trainer
from tinymoe.train.losses import LossOutput, forward_batch, lm_loss

log = logging.getLogger(__name__)
GB = 1e9


# -------------------------------------------------------------------- measurement


class RandomRows:
    """Provider of uniformly random token rows (fixed seed per (step, micro))."""

    def __init__(self, vocab: int, rows: int, seq_len: int, device: torch.device) -> None:
        self.vocab, self.rows, self.seq_len, self.device = vocab, rows, seq_len, device

    def batch(self, step: int, micro: int) -> Batch:
        g = torch.Generator().manual_seed(step * 1009 + micro)
        x = torch.randint(0, self.vocab, (self.rows, self.seq_len + 1), generator=g)
        return Batch(x[:, :-1].contiguous(), x[:, 1:].contiguous()).to(self.device)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


@dataclass
class Measure:
    seconds: float
    peak_gb: float
    extra: dict[str, float] = field(default_factory=dict)


def _fresh(runtime: Runtime) -> None:
    gc.collect()
    hardware.empty_cache(runtime.device)
    hardware.reset_peak_memory(runtime.device)


def _peak_gb(runtime: Runtime) -> float:
    return hardware.memory_bytes(runtime.device)["peak"] / GB


def _probe(loss: LossFn | None) -> Callable[[Transformer, Batch], torch.Tensor] | None:
    if loss is None:
        return None
    fn = loss
    return lambda m, b: fn(m, b).loss


def train_step(
    cfg: RunConfig,
    runtime: Runtime,
    model: Transformer,
    train: TrainConfig,
    plan: BatchPlan,
    seq_len: int,
    frozen: Sequence[Transformer] = (),
    loss: LossFn | None = None,
) -> Measure:
    """Seconds per optimizer step through the real Trainer: next-token CE, plus a no-grad
    forward of each ``frozen`` model per micro batch (DPO/GRPO reference), or the stage's
    own ``loss`` (distillation losses run their teacher inside)."""
    bc = cfg.bench
    mtp_w = model.cfg.mtp.loss_weight

    def lm(m: nn.Module, batch: Batch) -> LossOutput:
        for ref in frozen:
            with torch.no_grad():
                forward_batch(ref, batch)
        return lm_loss(m, batch, mtp_w, train.z_loss_weight)

    loss_fn = loss or lm

    t = replace(
        train,
        max_steps=bc.warmup + bc.steps,
        log_interval=10**9,
        ckpt_interval=10**9,
        eval_interval=10**9,
        resume="never",
    )
    trainer = Trainer(
        stage="bench",
        model=model,
        runtime=runtime,
        cfg=t,
        plan=plan,
        loss_fn=loss_fn,
        provider=RandomRows(model.cfg.vocab_size, plan.micro_batch_size, seq_len, runtime.device),
        logger=RunLogger(Path(), enabled=False),
        ckpt_dir=Path(),
        dist=DistInfo(0, 0, 1),
        save_final=False,
    )
    _fresh(runtime)
    result = trainer.fit()
    hardware.synchronize(runtime.device)
    times = result.step_times[bc.warmup :]
    peak = _peak_gb(runtime)
    del trainer
    model.zero_grad(set_to_none=True)
    return Measure(sum(times) / max(len(times), 1), peak)


def generation(
    runtime: Runtime, model: Transformer, batch: int, prompt_len: int, new_tokens: int
) -> Measure:
    """Seconds for one batch of ``batch`` sequences x ``new_tokens`` sampled tokens."""
    prompts = [
        [1 + (i * 7 + j) % (model.cfg.vocab_size - 1) for j in range(prompt_len)]
        for i in range(batch)
    ]
    g = torch.Generator().manual_seed(0)
    _fresh(runtime)
    with runtime.autocast(), runtime.sdpa_context():
        generate(model, prompts[: min(batch, 2)], 4, 1.0, generator=g)  # warm-up
        hardware.synchronize(runtime.device)
        t0 = time.perf_counter()
        out = generate(model, prompts, new_tokens, 1.0, generator=g)
        hardware.synchronize(runtime.device)
    dt = time.perf_counter() - t0
    n = sum(len(s) for s in out.sequences)
    return Measure(dt, _peak_gb(runtime), {"tokens_per_s": n / max(dt, 1e-9)})


def forward_pass(runtime: Runtime, model: Transformer, rows: int, seq_len: int) -> Measure:
    """Seconds for one no-grad forward of ``rows`` x ``seq_len`` (reference / teacher passes)."""
    was = model.training
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (rows, seq_len), device=runtime.device)
    with torch.no_grad(), runtime.autocast(), runtime.sdpa_context():
        model(x)
        hardware.synchronize(runtime.device)
        t0 = time.perf_counter()
        model(x)
        hardware.synchronize(runtime.device)
    model.train(was)
    return Measure(time.perf_counter() - t0, _peak_gb(runtime))


def snapshot_forward(runtime: Runtime, model: Transformer, tokens: int) -> Measure:
    """One snapshot ~ two probe forwards + one 4x-long specialisation forward, captured."""
    was = model.training
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (1, tokens), device=runtime.device)
    long = torch.randint(
        0, model.cfg.vocab_size, (1, min(4 * tokens, model.cfg.max_seq_len)), device=runtime.device
    )
    t0 = time.perf_counter()
    with torch.no_grad(), runtime.autocast():
        for x in (ids, ids, long):
            model(x, capture=Capture())
    hardware.synchronize(runtime.device)
    model.train(was)
    return Measure(time.perf_counter() - t0, _peak_gb(runtime))


# ----------------------------------------------------------------- stage estimates


@dataclass
class StageBench:
    stage: str
    model: str
    plan: str
    step_s: float
    peak_gb: float
    est_min: float
    notes: list[str] = field(default_factory=list)
    detail: dict[str, float] = field(default_factory=dict)


class Bench:
    def __init__(self, cfg: RunConfig) -> None:
        self.runtime = hardware.resolve_runtime(cfg.hardware)
        self._models: dict[str, Transformer] = {}
        self._gen: dict[tuple[str, int, int, int], Measure] = {}

    # model building ---------------------------------------------------------
    def model(self, mcfg: ModelConfig, vocab: int, max_seq_len: int | None = None) -> Transformer:
        c = replace(mcfg, vocab_size=mcfg.vocab_size or vocab)
        if max_seq_len:
            c.max_seq_len = max(c.max_seq_len, max_seq_len)
        key = json.dumps(to_dict(c), sort_keys=True)
        if key not in self._models:
            hardware.seed_everything(0)
            self._models = {}  # keep at most one model resident (memory honesty)
            _fresh(self.runtime)
            self._models[key] = Transformer(c).to(self.runtime.device)
        return self._models[key]

    def release(self) -> None:
        """Drop the resident model and cached allocations (after an OOM)."""
        self._models = {}
        _fresh(self.runtime)

    def plan(
        self,
        cfg: RunConfig,
        t: TrainConfig,
        model: Transformer,
        units: int | None,
        seq: int,
        unit: int,
        rows_per_unit: int = 1,
        loss: LossFn | None = None,
    ) -> BatchPlan:
        mbs = model_micro_batch(
            model,
            self.runtime,
            seq,
            t.micro_batch_size,
            t.auto_batch_max,
            cfg.hardware,
            _probe(loss),
            t,
        )
        mbs = max(mbs // rows_per_unit, 1)
        return derive_batch(units or mbs * unit, unit, mbs, 1)

    def gen(self, model: Transformer, tag: str, batch: int, prompt: int, new: int) -> Measure:
        key = (tag, batch, prompt, new)
        if key not in self._gen:
            self._gen[key] = generation(self.runtime, model, batch, prompt, new)
        return self._gen[key]

    # shared costs -----------------------------------------------------------
    def eval_cost(self, cfg: RunConfig, model: Transformer, tag: str, chat: bool) -> float:
        """Seconds per full evaluation (generation-dominated)."""
        e = cfg.eval
        n_buckets = len(cfg.data.mathworld.digits) * len(cfg.data.mathworld.steps)
        n_math = n_buckets * e.math_per_bucket
        p = cfg.bench.prompt_tokens
        per = self.gen(model, tag, e.batch_size, p, e.max_new_tokens).seconds
        math_batches = math.ceil(n_math / e.batch_size) * (
            1 + (e.passk_samples if e.passk_samples >= e.passk_k else 0)
        )
        total = math_batches * per
        if chat and e.story_prompts > 0:
            per_story = self.gen(model, tag, e.batch_size, p, e.story_max_new_tokens).seconds
            total += math.ceil(e.story_prompts / e.batch_size) * per_story
        n_probes = len(e.probe_stories) + len(e.probe_instructions) + e.probe_math
        total += self.gen(model, tag, max(n_probes, 1), p, e.probe_max_new_tokens).seconds
        return total

    def lm_stage(
        self,
        cfg: RunConfig,
        name: str,
        mcfg: ModelConfig,
        vocab: int,
        steps: int,
        train: TrainConfig,
        chat: bool,
        loss: LossFn | None = None,
        label: str = "model",
    ) -> StageBench:
        model = self.model(mcfg, vocab, train.seq_len)
        seq = train.seq_len
        plan = self.plan(cfg, train, model, train.tokens_per_step, seq, seq, loss=loss)
        m = train_step(cfg, self.runtime, model, train, plan, seq, loss=loss)
        n_evals = math.ceil(steps / train.eval_interval) if steps else 0
        ev = self.eval_cost(cfg, model, label, chat) * n_evals
        snap = 0.0
        if cfg.snapshot.enabled and cfg.snapshot.interval and steps:
            one = snapshot_forward(self.runtime, model, cfg.snapshot.max_probe_tokens).seconds
            snap = one * math.ceil(steps / cfg.snapshot.interval)
        total = m.seconds * steps + ev + snap
        return StageBench(
            name,
            label,
            _plan_str(plan),
            m.seconds,
            m.peak_gb,
            total / 60,
            detail={
                "train_s": m.seconds * steps,
                "eval_s": ev,
                "snapshot_s": snap,
                "snapshot_overhead": snap / max(m.seconds * steps, 1e-9),
                "tokens_per_s": plan.units_per_step / max(m.seconds, 1e-9),
            },
        )

    # per-stage workloads ----------------------------------------------------
    def stage(self, cfg: RunConfig, vocab: int) -> list[StageBench]:
        name, t = cfg.stage.name, cfg.train
        if name in ("pretrain", "sft"):
            return [self.lm_stage(cfg, name, cfg.model, vocab, t.max_steps, t, name == "sft")]
        if name == "midtrain":
            out = [self.lm_stage(cfg, "midtrain/anneal", cfg.model, vocab, t.max_steps, t, False)]
            ce = cfg.midtrain.context_extension
            if ce.enabled and ce.steps:
                ext = replace(
                    t,
                    seq_len=ce.seq_len,
                    micro_batch_size=ce.micro_batch_size,
                    tokens_per_step=ce.tokens_per_step,
                    eval_interval=max(ce.steps, 1),
                )
                out.append(
                    self.lm_stage(cfg, "midtrain/context", cfg.model, vocab, ce.steps, ext, False)
                )
            return out
        if name == "dpo":
            return [self._dpo(cfg, vocab)]
        if name == "grpo":
            return [self._grpo(cfg, vocab)]
        return self._distill(cfg, vocab)

    def _dpo(self, cfg: RunConfig, vocab: int) -> StageBench:
        """Policy step on chosen+rejected rows; the reference is a one-off forward pre-pass
        over every pair (stages/dpo.py keeps no second model resident)."""
        t, d = cfg.train, cfg.dpo
        model = self.model(cfg.model, vocab)
        plan = self.plan(cfg, t, model, t.examples_per_step, t.seq_len, 1, rows_per_unit=2)
        rows = replace(plan, micro_batch_size=plan.micro_batch_size * 2)  # chosen + rejected
        m = train_step(cfg, self.runtime, model, t, rows, t.seq_len)
        prepass = 0.0
        if d.loss != "simpo":
            fwd = forward_pass(self.runtime, model, rows.micro_batch_size, t.seq_len)
            prepass = fwd.seconds * math.ceil(d.n_pairs / plan.micro_batch_size)
        ev = self.eval_cost(cfg, model, "model", True) * math.ceil(t.max_steps / t.eval_interval)
        total = m.seconds * t.max_steps + prepass + ev
        return StageBench(
            "dpo",
            "model",
            _plan_str(plan) + " pairs",
            m.seconds,
            m.peak_gb,
            total / 60,
            [f"reference pre-pass {prepass:.0f}s"],
            {"train_s": m.seconds * t.max_steps, "prepass_s": prepass, "eval_s": ev},
        )

    def _grpo(self, cfg: RunConfig, vocab: int) -> StageBench:
        g, t, p = cfg.grpo, cfg.train, cfg.bench.prompt_tokens
        seq = p + g.max_new_tokens
        model = self.model(cfg.model, vocab)
        ref = self.model_copy(model)
        n_seq = g.prompts_per_rollout * g.group_size
        gen = self.gen(model, "model", min(n_seq, cfg.eval.batch_size), p, g.max_new_tokens)
        gen_s = gen.seconds * math.ceil(n_seq / cfg.eval.batch_size)
        rows = max(n_seq // g.minibatches_per_epoch, 1)
        mbs = model_micro_batch(
            model,
            self.runtime,
            seq,
            t.micro_batch_size,
            t.auto_batch_max,
            cfg.hardware,
            train=replace(t, seq_len=seq),
        )
        plan = derive_batch(rows, 1, min(mbs, rows), 1)
        m = train_step(cfg, self.runtime, model, replace(t, seq_len=seq), plan, seq, [ref])
        updates = g.ppo_epochs * g.minibatches_per_epoch
        per_iter = gen_s + updates * m.seconds
        ev = self.eval_cost(cfg, model, "model", True) * math.ceil(t.max_steps / t.eval_interval)
        return StageBench(
            "grpo",
            "model",
            f"{n_seq} rollouts, {updates} updates of {_plan_str(plan)}",
            m.seconds,
            max(m.peak_gb, gen.peak_gb),
            (per_iter * t.max_steps + ev) / 60,
            ["+ reference model", f"generation {gen.extra['tokens_per_s']:.0f} tok/s"],
            {"rollout_s": gen_s, "update_s": m.seconds, "iteration_s": per_iter, "eval_s": ev},
        )

    def _distill(self, cfg: RunConfig, vocab: int) -> list[StageBench]:
        d, t = cfg.distill, cfg.train
        out = [self._teacher_samples(cfg, vocab)]
        student_train = replace(t, lr=d.student_lr)
        if d.student_pretrain_steps:
            out.append(
                self.lm_stage(
                    cfg,
                    "distill/student_pretrain",
                    cfg.student,
                    vocab,
                    d.student_pretrain_steps,
                    student_train,
                    False,
                    label="student",
                )
            )
        for mode in d.modes:
            if mode in ("baseline", "offpolicy"):
                out.append(
                    self.lm_stage(
                        cfg,
                        f"distill/{mode}",
                        cfg.student,
                        vocab,
                        t.max_steps,
                        student_train,
                        True,
                        label="student",
                    )
                )
            elif mode == "logit":
                teacher = self.model_copy(self.model(cfg.model, vocab))
                out.append(self._with_teacher(cfg, vocab, teacher, student_train))
                del teacher
            elif mode == "onpolicy":
                out.append(self._onpolicy(cfg, vocab))
        return out

    def _teacher_samples(self, cfg: RunConfig, vocab: int) -> StageBench:
        d, p = cfg.distill, cfg.bench.prompt_tokens
        teacher = self.model(cfg.model, vocab)
        n = d.teacher_prompts * d.teacher_samples_per_prompt
        g = self.gen(teacher, "model", cfg.eval.batch_size, p, cfg.eval.max_new_tokens)
        total = g.seconds * math.ceil(n / cfg.eval.batch_size)
        return StageBench(
            "distill/teacher_samples",
            "model",
            f"{n} samples",
            0.0,
            g.peak_gb,
            total / 60,
            [f"generation {g.extra['tokens_per_s']:.0f} tok/s"],
        )

    def _with_teacher(
        self, cfg: RunConfig, vocab: int, teacher: Transformer, train: TrainConfig
    ) -> StageBench:
        sb = self.lm_stage(
            cfg,
            "distill/logit",
            cfg.student,
            vocab,
            cfg.train.max_steps,
            train,
            True,
            loss=logit_kd_loss(teacher, cfg.distill.kd_temperature, cfg.distill.kd_ce_weight),
            label="student",
        )
        sb.notes.append("+ teacher forward per micro batch")
        return sb

    def _onpolicy(self, cfg: RunConfig, vocab: int) -> StageBench:
        d, p = cfg.distill, cfg.bench.prompt_tokens
        seq = p + d.onpolicy_max_new_tokens
        teacher = self.model_copy(self.model(cfg.model, vocab))
        student = self.model(cfg.student, vocab)
        n = d.onpolicy_prompts_per_step
        gen = generation(self.runtime, student, n, p, d.onpolicy_max_new_tokens)
        train = replace(cfg.train, seq_len=seq, lr=d.onpolicy_lr)
        loss = reverse_kl_loss(teacher)
        mbs = model_micro_batch(
            student,
            self.runtime,
            seq,
            cfg.train.micro_batch_size,
            cfg.train.auto_batch_max,
            cfg.hardware,
            _probe(loss),
            train,
        )
        plan = derive_batch(n, 1, min(mbs, n), 1)
        m = train_step(cfg, self.runtime, student, train, plan, seq, loss=loss)
        per = gen.seconds + m.seconds
        return StageBench(
            "distill/onpolicy",
            "student",
            f"{n} rollouts/step",
            per,
            max(m.peak_gb, gen.peak_gb),
            per * d.onpolicy_steps / 60,
            ["+ teacher forward"],
            {"rollout_s": gen.seconds, "update_s": m.seconds},
        )

    def model_copy(self, model: Transformer) -> Transformer:
        """A frozen second resident model (reference / teacher), as the stage holds it."""
        ref = Transformer(model.cfg).to(self.runtime.device)
        ref.load_state_dict(model.state_dict())
        ref.requires_grad_(False)
        ref.eval()
        return ref


def _plan_str(plan: BatchPlan) -> str:
    return f"{plan.micro_batch_size}x{plan.grad_accum}"


# ------------------------------------------------------------------------- report


def run(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    configs = stage_configs(path, overrides)
    first = configs[0]
    bench = Bench(first)
    vocab = first.model.vocab_size or first.data.tokenizer.vocab_size
    rows: list[StageBench] = []
    for cfg in configs:
        log.info("[bench] stage %s", cfg.stage.name)
        try:
            rows += bench.stage(cfg, vocab)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not hardware.is_oom(exc):
                raise
            log.error("[bench] stage %s ran out of memory", cfg.stage.name, exc_info=exc)
            bench.release()
            rows.append(StageBench(cfg.stage.name, "-", "-", math.nan, math.inf, math.nan, ["OOM"]))
    pc = param_count(replace(first.model, vocab_size=vocab))
    budget = first.hardware.memory_budget_gb
    peak = max(r.peak_gb for r in rows)
    rss = hardware.process_rss_bytes() / GB
    return {
        "pipeline": str(path),
        "runtime": bench.runtime.summary(),
        "model_params": {"total": pc.total, "active": pc.active},
        "stages": [r.__dict__ for r in rows],
        "total_minutes": sum(r.est_min for r in rows),
        "peak_gb": peak,
        "process_rss_gb": rss,
        "budget_gb": budget,
        "fits": budget is None or max(peak, rss) <= budget,
    }


def markdown(res: dict[str, Any]) -> str:
    lines = [
        f"runtime: {res['runtime']}",
        f"model: {res['model_params']['total'] / 1e6:.1f}M total / "
        f"{res['model_params']['active'] / 1e6:.1f}M active",
        "",
        "| stage | model | batch | s/step | peak GB | est. min | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res["stages"]:
        lines.append(
            f"| {r['stage']} | {r['model']} | {r['plan']} | {r['step_s']:.3f} | "
            f"{r['peak_gb']:.2f} | {r['est_min']:.1f} | {'; '.join(r['notes'])} |"
        )
    lines += [
        "",
        f"total estimate: {res['total_minutes']:.1f} min; peak device memory "
        f"{res['peak_gb']:.2f} GB; process RSS {res['process_rss_gb']:.2f} GB; budget "
        f"{res['budget_gb']} GB -> {'FITS' if res['fits'] else 'OVER BUDGET'}",
    ]
    return "\n".join(lines)


def main(path: Path, overrides: Sequence[str], json_out: str | None) -> int:
    res = run(path, overrides)
    print(markdown(res))
    if json_out:
        Path(json_out).write_text(json.dumps(res, indent=2))
    return 0 if res["fits"] else 1
