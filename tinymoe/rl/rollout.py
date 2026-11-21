"""RL rollouts and the trainer-facing provider.

One *iteration* = sample ``group_size`` responses for each of ``prompts_per_rollout``
problems, score them with the verifier, compute group-relative advantages, then run
``ppo_epochs x minibatches_per_epoch`` optimizer steps over the buffer. The provider maps
the trainer's step counter onto (iteration, epoch, minibatch) and rolls out lazily, so
the generic :class:`~tinymoe.train.loop.Trainer` drives RL like any other stage. The
buffer and prompt counter live in ``state_dict``: resume is exact mid-iteration.

Train/inference mismatch, measured and (optionally) corrected:

* ``old_logp`` is recomputed with the *training* forward at the start of the iteration;
  ``rollout_logp`` is what the KV-cache sampler saw. Their gap is logged
  (``rl/train_infer_gap``). IcePop (GLM-4.5/5) masks tokens whose probability ratio
  ``exp(old - rollout)`` leaves ``[low, high]``.
* MoE *routing replay* (R3 / "rollout routing replay"): the training forward re-uses the
  experts chosen during sampling (``routing_override``), so the gradient is taken through
  the same experts that produced the sample. ``rl/routing_mismatch`` is the fraction of
  (token, layer) expert sets the training forward would have chosen differently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tinymoe.config import BatchPlan, GRPOConfig
from tinymoe.data.batch import IGNORE_INDEX
from tinymoe.data.mathworld import Problem
from tinymoe.data.tokenizer import Tokenizer
from tinymoe.eval import probes
from tinymoe.eval.evaluator import Evaluator
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.transformer import Transformer
from tinymoe.rl import grpo
from tinymoe.rl.rewards import score
from tinymoe.train.losses import sequence_logprobs

log = logging.getLogger(__name__)


@dataclass
class Buffer:
    """One iteration's rollouts as padded (N, T) arrays; labels are completion-only."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    rollout_logp: torch.Tensor
    old_logp: torch.Tensor
    ref_logp: torch.Tensor | None
    advantages: torch.Tensor  # (N,)
    lengths: torch.Tensor  # (N,) completion tokens
    valid: torch.Tensor  # (N,) 0 for dropped (zero-variance) groups
    routing: dict[int, torch.Tensor] | None  # layer -> (N, T, k)

    def state(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    token_mask: torch.Tensor
    old_logp: torch.Tensor
    ref_logp: torch.Tensor | None
    advantages: torch.Tensor
    row_weight: torch.Tensor
    routing: dict[int, torch.Tensor] | None
    denom: float
    scale: float

    def to(self, device: torch.device) -> RolloutBatch:
        def mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return RolloutBatch(
            self.input_ids.to(device),
            self.labels.to(device),
            self.token_mask.to(device),
            self.old_logp.to(device),
            mv(self.ref_logp),
            self.advantages.to(device),
            self.row_weight.to(device),
            {k: v.to(device) for k, v in self.routing.items()} if self.routing else None,
            self.denom,
            self.scale,
        )


@dataclass
class Sampled:
    problem: Problem
    prompt: list[int]
    completion: list[int]
    logp: list[float]
    finished: bool
    routing: dict[int, np.ndarray] | None
    text: str = ""
    reward: float = 0.0
    correct: bool = False
    format_ok: bool = False


@dataclass
class RolloutStats:
    metrics: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)


def pad_completions(
    prompts: Sequence[Sequence[int]], completions: Sequence[Sequence[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded training rows for sampled text: inputs = prompt + completion[:-1],
    labels = the completion tokens at their positions (IGNORE elsewhere)."""
    width = max(max(len(p) + len(c) - 1 for p, c in zip(prompts, completions, strict=True)), 1)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    labels = torch.full((len(prompts), width), IGNORE_INDEX, dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, completions, strict=True)):
        inp = [*p, *c][:-1]
        ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        start = len(p) - 1
        labels[i, start : start + len(c)] = torch.tensor(list(c), dtype=torch.long)
    return ids, labels


def pad_rows(
    samples: Sequence[Sampled], pad_id: int, top_k: int, routed: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor] | None]:
    """:func:`pad_completions` plus the sampler's log-probs and recorded expert routing."""
    ids, labels = pad_completions(
        [s.prompt for s in samples], [s.completion for s in samples], pad_id
    )
    n, width = ids.shape
    rlp = torch.zeros(n, width)
    routing: dict[int, torch.Tensor] | None = None
    if routed and samples[0].routing:
        routing = {
            layer: torch.zeros(n, width, top_k, dtype=torch.long) for layer in samples[0].routing
        }
    for i, s in enumerate(samples):
        start = len(s.prompt) - 1
        rlp[i, start : start + len(s.completion)] = torch.tensor(s.logp)
        if routing is not None and s.routing is not None:
            for layer, r in s.routing.items():
                routing[layer][i, : r.shape[0]] = torch.from_numpy(r)
    return ids, labels, rlp, routing


class RolloutProvider:
    """``batch(step, micro)`` for the Trainer; rolls out when a new iteration starts."""

    def __init__(
        self,
        *,
        cfg: GRPOConfig,
        policy: Transformer,
        ref: Transformer | None,
        tok: Tokenizer,
        runtime: Runtime,
        evaluator: Evaluator,
        problem_at: Callable[[int], Problem],
        plan: BatchPlan,
        rank: int,
        seed: int,
        logger: RunLogger,
        stage: str,
        step_offset: int,
        snapshot_every: int,
        eval_batch: int,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.ref = ref
        self.tok = tok
        self.runtime = runtime
        self.evaluator = evaluator
        self.problem_at = problem_at
        self.plan = plan
        self.rank = rank
        self.seed = seed
        self.logger = logger
        self.stage = stage
        self.step_offset = step_offset
        self.snapshot_every = snapshot_every
        self.eval_batch = eval_batch
        self.updates = cfg.ppo_epochs * cfg.minibatches_per_epoch
        self.n_rows = cfg.prompts_per_rollout * cfg.group_size
        self.rows_per_mb = self.n_rows // cfg.minibatches_per_epoch
        self.iteration = -1
        self.next_prompt = 0
        self.buffer: Buffer | None = None
        self.moe = bool(policy.moe_layers())
        self.replay = cfg.routing_replay and self.moe

    # ---------------------------------------------------------------- sampling
    def sample(self, problems: Sequence[Problem], generator: torch.Generator) -> list[Sampled]:
        g = self.cfg
        prompts = [probes.math_prompt_ids(self.tok, p.question, "chat") for p in problems]
        flat = [p for p in prompts for _ in range(g.group_size)]
        gen = self.evaluator.generate(
            self.policy,
            flat,
            g.max_new_tokens,
            g.temperature,
            generator,
            record_routing=self.moe,
            batch_size=self.eval_batch,
            top_p=g.top_p,
        )
        out = []
        for i, seq in enumerate(gen.sequences):
            problem = problems[i // g.group_size]
            s = Sampled(
                problem,
                flat[i],
                list(seq),
                gen.logprobs[i],
                gen.finished[i],
                gen.routing[i] if gen.routing else None,
            )
            s.text = self.evaluator.response_text(seq)
            r = score(s.text, problem.answer, len(seq), s.finished, g.max_new_tokens, g)
            s.reward, s.correct, s.format_ok = r.total, r.correct, r.format_ok
            out.append(s)
        return out

    def collect_groups(self, it: int) -> tuple[list[Sampled], list[bool], int]:
        """``prompts_per_rollout`` groups; with dynamic sampling (DAPO) groups whose accuracy
        is all 0 or all 1 are replaced by fresh prompts for up to ``max_resample_rounds``
        extra rounds. The filter is on *correctness*, as in DAPO: format and overlong terms
        can vary inside a group whose answers are all wrong, but that variance teaches
        length and layout, not reasoning."""
        g = self.cfg
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + it)
        keep: list[list[Sampled]] = []
        spare: list[list[Sampled]] = []
        rounds = 0
        dropped = 0
        while True:
            need = g.prompts_per_rollout - len(keep)
            problems = [self.problem_at(self.next_prompt + j) for j in range(need)]
            self.next_prompt += need
            samples = self.sample(problems, gen)
            groups = [samples[i * g.group_size : (i + 1) * g.group_size] for i in range(need)]
            for grp in groups:
                informative = len({s.correct for s in grp}) > 1
                if informative or not g.dynamic_sampling:
                    keep.append(grp)
                else:
                    spare.append(grp)
                    dropped += 1
            if len(keep) >= g.prompts_per_rollout or rounds >= g.max_resample_rounds:
                break
            rounds += 1
        valid = [True] * len(keep)
        while len(keep) < g.prompts_per_rollout:  # fill: these rows carry no gradient
            keep.append(spare.pop())
            valid.append(False)
        flat = [s for grp in keep for s in grp]
        row_valid = [v for v in valid for _ in range(g.group_size)]
        return flat, row_valid, dropped

    # -------------------------------------------------------------- iteration
    @torch.no_grad()
    def rollout(self, it: int, step: int) -> None:
        g = self.cfg
        samples, row_valid, dropped = self.collect_groups(it)
        top_k = self.policy.cfg.moe.top_k
        ids, labels, rlp, routing = pad_rows(samples, self.tok.pad_id, top_k, self.moe)
        rewards = torch.tensor([s.reward for s in samples]).view(
            g.prompts_per_rollout, g.group_size
        )
        adv = grpo.group_advantages(rewards, g.std_normalize).reshape(-1)
        valid = torch.tensor(row_valid, dtype=torch.float32)
        adv = adv * valid
        lengths = torch.tensor([len(s.completion) for s in samples], dtype=torch.float32)
        old, mismatch = self._logps(self.policy, ids, labels, routing, measure=self.moe)
        ref = None
        if self.ref is not None:
            ref, _ = self._logps(self.ref, ids, labels, routing, measure=False)
        self.buffer = Buffer(ids, labels, rlp, old, ref, adv, lengths, valid, routing)
        self.iteration = it
        stats = self.stats(samples, rewards, row_valid, dropped, mismatch)
        self.logger.log_metrics(self.stage, self.step_offset + step, stats.metrics)
        if self.snapshot_every and it % self.snapshot_every == 0:
            self.logger.save_snapshot(
                self.stage,
                self.step_offset + step,
                {"iteration": it, "rollouts": stats.examples},
                name="rollouts",
            )

    def _logps(
        self,
        model: Transformer,
        ids: torch.Tensor,
        labels: torch.Tensor,
        routing: dict[int, torch.Tensor] | None,
        measure: bool,
    ) -> tuple[torch.Tensor, float | None]:
        """Per-token log-probs under ``model`` (training forward, eval mode, no grad); with
        ``measure`` also the routing mismatch against the rollout's recorded experts."""
        was = model.training
        model.eval()
        out_lp, diff, total = [], 0.0, 0.0
        bs = max(self.plan.micro_batch_size, 1)
        device = self.runtime.device
        for start in range(0, ids.shape[0], bs):
            sl = slice(start, start + bs)
            x, y = ids[sl].to(device), labels[sl].to(device)
            override = (
                {k: v[sl].to(device) for k, v in routing.items()}
                if (routing and self.replay)
                else None
            )
            with self.runtime.autocast(), self.runtime.sdpa_context():
                out = model(x, routing_override=override)
                if measure and routing:
                    natural = out.routing if override is None else model(x).routing
                    fed = _fed_mask(y)
                    for layer, rec in routing.items():
                        a = natural[layer].sort(-1).values
                        b = rec[sl].to(device).sort(-1).values
                        differs = (a != b).any(-1) & fed
                        diff += float(differs.sum())
                        total += float(fed.sum())
            out_lp.append(sequence_logprobs(out.logits, y, self.cfg.temperature).float().cpu())
        model.train(was)
        return torch.cat(out_lp), (diff / total if total else None)

    def stats(
        self,
        samples: Sequence[Sampled],
        rewards: torch.Tensor,
        row_valid: Sequence[bool],
        dropped: int,
        mismatch: float | None,
    ) -> RolloutStats:
        assert self.buffer is not None
        b = self.buffer
        comp = b.labels != IGNORE_INDEX
        gap = ((b.old_logp - b.rollout_logp).abs() * comp).sum() / comp.sum().clamp_min(1)
        m: dict[str, float] = {
            "rl/reward": float(rewards.mean()),
            "rl/reward_std": float(rewards.std(-1, unbiased=False).mean()),
            "rl/pass": float(np.mean([s.correct for s in samples])),
            "rl/format": float(np.mean([s.format_ok for s in samples])),
            "rl/resp_len": float(b.lengths.mean()),
            "rl/think_len": float(
                np.mean([self.evaluator.think_len(s.completion) for s in samples])
            ),
            "rl/truncated": float(np.mean([not s.finished for s in samples])),
            "rl/zero_var_groups": float(dropped),
            "rl/valid_frac": float(np.mean(row_valid)),
            "rl/train_infer_gap": float(gap),
            "rl/prompts_seen": float(self.next_prompt),
        }
        if mismatch is not None:
            m["rl/routing_mismatch"] = mismatch
        buckets = sorted({s.problem.bucket for s in samples})
        for bucket in buckets:
            sel = [s for s in samples if s.problem.bucket == bucket]
            m[f"rl/pass/{bucket}"] = float(np.mean([s.correct for s in sel]))
        examples = [
            {
                "bucket": bucket,
                "question": samples[i].problem.question,
                "answer": samples[i].problem.answer,
                "completion": samples[i].text,
                "reward": samples[i].reward,
                "advantage": float(b.advantages[i]),
                "correct": samples[i].correct,
                "format_ok": samples[i].format_ok,
                "length": len(samples[i].completion),
            }
            for bucket in buckets
            for i in [j for j, s in enumerate(samples) if s.problem.bucket == bucket][:2]
        ]
        return RolloutStats(m, examples)

    # --------------------------------------------------------------- provider
    def batch(self, step: int, micro: int) -> RolloutBatch:
        it, u = divmod(step, self.updates)
        if it != self.iteration:
            self.rollout(it, step)
        assert self.buffer is not None
        b = self.buffer
        epoch, mb = divmod(u, self.cfg.minibatches_per_epoch)
        perm = torch.from_numpy(
            np.random.default_rng([self.seed, it, epoch]).permutation(self.n_rows)
        )
        mb_rows = perm[mb * self.rows_per_mb : (mb + 1) * self.rows_per_mb]
        p = self.plan
        start = (micro * p.world_size + self.rank) * p.micro_batch_size
        rows = mb_rows[start : start + p.micro_batch_size]
        g = self.cfg
        denom = grpo.denominator(b.lengths[mb_rows], b.valid[mb_rows], g.loss_agg, g.max_new_tokens)
        comp = (b.labels[rows] != IGNORE_INDEX).float() * b.valid[rows, None]
        if g.icepop.enabled:
            ratio = torch.exp(b.old_logp[rows] - b.rollout_logp[rows])
            comp = comp * ((ratio >= g.icepop.low) & (ratio <= g.icepop.high)).float()
        cut = slice(0, int(_last_fed(b.labels[rows]).max()))  # drop all-padding columns
        batch = RolloutBatch(
            input_ids=b.input_ids[rows][:, cut],
            labels=b.labels[rows][:, cut],
            token_mask=comp[:, cut],
            old_logp=b.old_logp[rows][:, cut],
            ref_logp=b.ref_logp[rows][:, cut] if b.ref_logp is not None else None,
            advantages=b.advantages[rows],
            row_weight=grpo.row_weights(b.lengths[rows], g.loss_agg),
            routing={k: v[rows][:, cut] for k, v in b.routing.items()}
            if (b.routing and self.replay)
            else None,
            denom=denom,
            scale=float(p.grad_accum * p.world_size),
        )
        return batch.to(self.runtime.device)  # the buffer lives in host memory

    def state_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "next_prompt": self.next_prompt,
            "buffer": self.buffer.state() if self.buffer is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.iteration = int(state["iteration"])
        self.next_prompt = int(state["next_prompt"])
        self.buffer = Buffer(**state["buffer"]) if state["buffer"] is not None else None


def _last_fed(labels: torch.Tensor) -> torch.Tensor:
    """(B,) one past the last fed position: inputs are prompt + completion[:-1], whose last
    position carries the last completion label."""
    pos = torch.arange(1, labels.shape[1] + 1, device=labels.device)
    return ((labels != IGNORE_INDEX).long() * pos).amax(-1)


def _fed_mask(labels: torch.Tensor) -> torch.Tensor:
    """(B, T) True where a token was actually fed to the sampler (not right padding)."""
    pos = torch.arange(labels.shape[1], device=labels.device)
    return pos[None] < _last_fed(labels)[:, None]
