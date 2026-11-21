"""Shared evaluation, run periodically and at the end of every stage.

* per-source validation loss / perplexity (and long-context loss in mid-training);
* math pass@1 (greedy) and unbiased pass@k (Chen et al., 2021) per difficulty bucket,
  format compliance, ``<think>`` length, response length;
* story instruction following (mentions the requested subject, sensible length);
* probe generations stored for the viewer's "generations over time".
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tinymoe.config import RunConfig
from tinymoe.data import mathworld, stories
from tinymoe.data.batch import IGNORE_INDEX
from tinymoe.data.mathworld import Problem
from tinymoe.data.pretrain_mix import MixtureSampler
from tinymoe.data.sft_data import story_instruction
from tinymoe.data.tokenizer import Tokenizer, stop_ids
from tinymoe.eval import probes
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.transformer import Generation, Transformer, generate
from tinymoe.train.losses import forward_batch, masked_mean, token_ce
from tinymoe.train.providers import RowSource

log = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass
class Scored:
    """One generated response with its verdict (used by DPO mining, calibration, eval)."""

    problem: Problem
    text: str
    tokens: list[int]
    correct: bool
    format_ok: bool
    think_len: int


class Evaluator:
    def __init__(
        self,
        cfg: RunConfig,
        tok: Tokenizer,
        runtime: Runtime,
        logger: RunLogger,
        stage: str,
        val_shards: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.cfg = cfg
        self.tok = tok
        self.runtime = runtime
        self.logger = logger
        self.stage = stage
        self.kind = probes.eval_kind(stage)
        self.val_shards = dict(val_shards or {})
        buckets = mathworld.buckets_from(cfg.data.mathworld)
        self.problems = mathworld.problem_set(
            cfg.data.mathworld, cfg.eval.seed, "eval", cfg.eval.math_per_bucket, buckets
        )
        self.probe_set = probes.probes(cfg)

    # ------------------------------------------------------------- generation
    def generate(
        self,
        model: Transformer,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
        record_routing: bool = False,
        batch_size: int | None = None,
        top_p: float = 1.0,
    ) -> Generation:
        bs = batch_size or self.cfg.eval.batch_size
        seqs: list[list[int]] = []
        lps: list[list[float]] = []
        done: list[bool] = []
        routes: list[dict[int, np.ndarray]] = []
        for i in range(0, len(prompts), bs):
            with self.runtime.autocast(), self.runtime.sdpa_context():
                g = generate(
                    model,
                    prompts[i : i + bs],
                    max_new_tokens,
                    temperature,
                    top_p,
                    stop_ids(self.tok),
                    self.tok.pad_id,
                    generator,
                    record_routing,
                )
            seqs += g.sequences
            lps += g.logprobs
            done += g.finished
            routes += g.routing or []
        return Generation(seqs, lps, done, routes if record_routing else None)

    def response_text(self, tokens: Sequence[int]) -> str:
        """Decoded response without the stop token; base-format output is cut at the first
        blank line (where the pre-training format starts a new problem)."""
        body = list(tokens)
        if body and body[-1] in stop_ids(self.tok):
            body = body[:-1]
        text = self.tok.decode(body)
        return text.split("\n\n", 1)[0] if self.kind == "base" else text

    def think_len(self, tokens: Sequence[int]) -> int:
        if self.tok.think_id in tokens and self.tok.think_end_id in tokens:
            return max(tokens.index(self.tok.think_end_id) - tokens.index(self.tok.think_id) - 1, 0)
        return 0

    def judge(self, text: str, answer: int) -> tuple[bool, bool]:
        """(correct, format_ok). Base format only needs a parsable answer line."""
        if self.kind == "base":
            parsed = mathworld.extract_answer(text)
            return parsed == answer, parsed is not None
        v = mathworld.verify(text, answer)
        return v.correct, v.format_ok

    def score(self, problems: Sequence[Problem], gen: Generation) -> list[Scored]:
        out = []
        for problem, seq in zip(problems, gen.sequences, strict=True):
            text = self.response_text(seq)
            correct, fmt = self.judge(text, problem.answer)
            out.append(Scored(problem, text, list(seq), correct, fmt, self.think_len(seq)))
        return out

    def sample_problems(
        self,
        model: Transformer,
        problems: Sequence[Problem],
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[Scored]]:
        """``n`` sampled responses per problem, scored by the verifier."""
        prompts = [probes.math_prompt_ids(self.tok, p.question, self.kind) for p in problems]
        flat = [p for p in prompts for _ in range(n)]
        flat_problems = [p for p in problems for _ in range(n)]
        g = torch.Generator().manual_seed(seed)
        scored = self.score(
            flat_problems, self.generate(model, flat, max_new_tokens, temperature, g)
        )
        return [scored[i * n : (i + 1) * n] for i in range(len(problems))]

    # ------------------------------------------------------------------ evals
    @torch.no_grad()
    def lm_losses(
        self, model: Transformer, seq_len: int | None = None, prefix: str = "val"
    ) -> dict[str, float]:
        seq_len = seq_len or self.cfg.train.seq_len
        out: dict[str, float] = {}
        model.eval()
        rows = 8
        for name, shard in self.val_shards.items():
            if len(shard) <= seq_len + 1:
                continue
            sampler = MixtureSampler(
                {name: shard}, {name: 1.0}, seq_len, self.cfg.eval.seed, self.tok.bos_id
            )
            losses = []
            for b in range(self.cfg.eval.lm_batches):
                batch = sampler.batch(range(b * rows, (b + 1) * rows)).to(self.runtime.device)
                with self.runtime.autocast(), self.runtime.sdpa_context():
                    logits = model(batch.input_ids).logits
                mask = torch.ones_like(batch.labels, dtype=torch.float32)
                losses.append(float(masked_mean(token_ce(logits, batch.labels), mask)))
            loss = float(np.mean(losses))
            out[f"{prefix}/loss/{name}"] = loss
            out[f"{prefix}/ppl/{name}"] = math.exp(min(loss, 30.0))
        return out

    @torch.no_grad()
    def rows_loss(
        self, model: Transformer, source: RowSource, n_rows: int, key: str, rows_per_batch: int = 8
    ) -> dict[str, float]:
        """Mean masked CE (only supervised tokens) over the first ``n_rows`` of a row source,
        e.g. held-out SFT conversations: ``{key: loss}``."""
        model.eval()
        total, count = 0.0, 0.0
        for start in range(0, n_rows, rows_per_batch):
            idx = range(start, min(start + rows_per_batch, n_rows))
            batch = source.batch(idx).to(self.runtime.device)
            with self.runtime.autocast(), self.runtime.sdpa_context():
                logits = forward_batch(model, batch).logits
            mask = (batch.labels != IGNORE_INDEX).float()
            total += float((token_ce(logits, batch.labels) * mask).sum())
            count += float(mask.sum())
        return {key: total / max(count, 1.0)}

    @torch.no_grad()
    def math(self, model: Transformer) -> tuple[dict[str, float], dict[str, float]]:
        """Returns (metrics, pass@1 per bucket)."""
        problems = [p for ps in self.problems.values() for p in ps]
        prompts = [probes.math_prompt_ids(self.tok, p.question, self.kind) for p in problems]
        scored = self.score(problems, self.generate(model, prompts, self.cfg.eval.max_new_tokens))
        metrics: dict[str, float] = {}
        per_bucket: dict[str, float] = {}
        for bucket in self.problems:
            sel = [s for s in scored if s.problem.bucket == bucket]
            per_bucket[bucket] = float(np.mean([s.correct for s in sel]))
            metrics[f"eval/math/pass1/{bucket}"] = per_bucket[bucket]
        metrics["eval/math/pass1"] = float(np.mean([s.correct for s in scored]))
        metrics["eval/math/format"] = float(np.mean([s.format_ok for s in scored]))
        metrics["eval/math/think_len"] = float(np.mean([s.think_len for s in scored]))
        metrics["eval/math/resp_len"] = float(np.mean([len(s.tokens) for s in scored]))
        n, k = self.cfg.eval.passk_samples, self.cfg.eval.passk_k
        if n >= k > 0:
            sampled = self.sample_problems(
                model,
                problems,
                n,
                self.cfg.eval.temperature,
                self.cfg.eval.max_new_tokens,
                self.cfg.eval.seed,
            )
            metrics[f"eval/math/pass{k}"] = float(
                np.mean([pass_at_k(n, sum(s.correct for s in group), k) for group in sampled])
            )
        return metrics, per_bucket

    @torch.no_grad()
    def stories(self, model: Transformer) -> dict[str, float]:
        if self.kind == "base" or self.cfg.eval.story_prompts <= 0:
            return {}
        subjects = [
            stories.SUBJECTS[i % len(stories.SUBJECTS)] for i in range(self.cfg.eval.story_prompts)
        ]
        prompts = [probes.chat_prompt_ids(self.tok, story_instruction(s)) for s in subjects]
        gen = self.generate(model, prompts, self.cfg.eval.story_max_new_tokens)
        texts = [self.response_text(seq) for seq in gen.sequences]
        words = [len(t.split()) for t in texts]
        lo, hi = self.cfg.eval.story_min_words, self.cfg.eval.story_max_words
        return {
            "eval/story/mention": float(
                np.mean([stories.mentions(t, s) for t, s in zip(texts, subjects, strict=True)])
            ),
            "eval/story/length_ok": float(np.mean([lo <= w <= hi for w in words])),
            "eval/story/words": float(np.mean(words)),
        }

    @torch.no_grad()
    def probe_generations(self, model: Transformer) -> list[dict[str, Any]]:
        prompts = [probes.probe_ids(self.tok, p, self.kind) for p in self.probe_set]
        gen = self.generate(model, prompts, self.cfg.eval.probe_max_new_tokens)
        out = []
        for probe, seq in zip(self.probe_set, gen.sequences, strict=True):
            text = self.tok.decode(seq[:-1] if seq and seq[-1] in stop_ids(self.tok) else seq)
            rec: dict[str, Any] = {
                "id": probe.id,
                "kind": probe.kind,
                "prompt": probe.prompt,
                "completion": text,
            }
            if probe.answer is not None:
                correct, fmt = self.judge(self.response_text(seq), probe.answer)
                rec.update(answer=probe.answer, correct=correct, format_ok=fmt)
            out.append(rec)
        return out

    def run(
        self, model: Transformer, step: int, with_lm: bool = True, tag: str | None = None
    ) -> dict[str, float]:
        """Full eval; generations are written as a snapshot named ``generations``. ``tag``
        names the model in that snapshot when a stage evaluates several (distillation)."""
        was_training = model.training
        model.eval()
        metrics: dict[str, float] = {}
        if with_lm:
            metrics.update(self.lm_losses(model))
        math_metrics, _ = self.math(model)
        metrics.update(math_metrics)
        metrics.update(self.stories(model))
        gens = self.probe_generations(model)
        payload: dict[str, Any] = {"kind": self.kind, "generations": gens}
        if tag is not None:
            payload["model"] = tag
        self.logger.save_snapshot(self.stage, step, payload, name="generations")
        model.train(was_training)
        log.info("[%s] eval step %d: %s", self.stage, step, _short(metrics))
        return metrics


def _short(metrics: Mapping[str, float]) -> str:
    keys = [k for k in metrics if k.count("/") <= 2 and not k.startswith("val/ppl")]
    return ", ".join(f"{k}={metrics[k]:.3f}" for k in keys)
