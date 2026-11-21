"""Stage 4 - preference tuning: DPO / IPO / SimPO (``dpo.loss``) on verifier-defined pairs.

Pairs: synthetic (chosen = correct, well-formatted solution; rejected = propagated
arithmetic error, broken format or rambling - data/pref_data.py) plus, optionally, pairs
mined from the SFT model's own samples ranked by the verifier (``model_sampled_prompts``).

Reference log-probs come from the *initial* policy and are computed once in a pre-pass,
so no second model is resident during training (the memory a 16 GB machine needs for
the policy's activations). SimPO is reference-free and skips the pre-pass.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from tinymoe.data import rl_prompts
from tinymoe.data.batch import IGNORE_INDEX, shuffled_index
from tinymoe.data.mathworld import buckets_from
from tinymoe.data.pref_data import PrefPair, build_pairs, pairs_from_samples
from tinymoe.data.tokenizer import Message, Tokenizer, render_chat
from tinymoe.model.transformer import Transformer
from tinymoe.stages.common import StageContext, stage_main
from tinymoe.train.losses import COUNT_PREFIX, LossOutput, sequence_logprobs
from tinymoe.train.preference import preference_loss
from tinymoe.train.providers import RowProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedPair:
    chosen: tuple[np.ndarray, np.ndarray]  # (inputs, labels)
    rejected: tuple[np.ndarray, np.ndarray]
    kind: int


def encode_side(
    tok: Tokenizer, prompt: Sequence[Message], reply: str, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    ids, mask = render_chat(tok, [*prompt, Message("assistant", reply)])
    ids, mask = ids[: seq_len + 1], mask[: seq_len + 1]
    tokens = np.asarray(ids, dtype=np.int64)
    labels = np.where(np.asarray(mask[1:]) == 1, tokens[1:], IGNORE_INDEX)
    return tokens[:-1], labels


@dataclass
class PairBatch:
    """``2B`` rows - chosen 0..B-1 then rejected 0..B-1 - padded on the right (causal
    attention never looks right, so no key mask is needed), with reference log-probs."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    ref_logp: torch.Tensor  # (2B,)
    kinds: torch.Tensor  # (B,)

    def to(self, device: torch.device) -> PairBatch:
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class PairDataset:
    def __init__(self, pairs: Sequence[EncodedPair], pad_id: int, seed: int) -> None:
        if not pairs:
            raise ValueError("no preference pairs")
        self.pairs = list(pairs)
        self.pad_id = pad_id
        self.seed = seed
        self.ref_logp = np.zeros((len(pairs), 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def items(self, indices: Sequence[int]) -> list[int]:
        return [shuffled_index(i, len(self.pairs), self.seed) for i in indices]

    def collate(self, items: Sequence[int]) -> PairBatch:
        sides = [self.pairs[i].chosen for i in items] + [self.pairs[i].rejected for i in items]
        width = max(len(x) for x, _ in sides)
        ids = np.full((len(sides), width), self.pad_id, dtype=np.int64)
        labels = np.full((len(sides), width), IGNORE_INDEX, dtype=np.int64)
        for row, (x, y) in enumerate(sides):
            ids[row, : len(x)] = x
            labels[row, : len(y)] = y
        ref = np.concatenate([self.ref_logp[list(items), 0], self.ref_logp[list(items), 1]])
        return PairBatch(
            torch.from_numpy(ids),
            torch.from_numpy(labels),
            torch.from_numpy(ref),
            torch.tensor([self.pairs[i].kind for i in items]),
        )

    def batch(self, indices: Sequence[int]) -> PairBatch:
        return self.collate(self.items(indices))


def summed_logps(model: nn.Module, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor, Any]:
    out = model(batch.input_ids)
    lp = sequence_logprobs(out.logits, batch.labels).sum(-1)
    return lp, (batch.labels != IGNORE_INDEX).sum(-1).float(), out


@torch.no_grad()
def fill_reference(ctx: StageContext, model: Transformer, data: PairDataset, bs: int) -> None:
    model.eval()
    for start in range(0, len(data), bs):
        items = list(range(start, min(start + bs, len(data))))
        batch = data.collate(items).to(ctx.device)
        with ctx.runtime.autocast(), ctx.runtime.sdpa_context():
            lp, _, _ = summed_logps(model, batch)
        n = len(items)
        data.ref_logp[items, 0] = lp[:n].float().cpu().numpy()
        data.ref_logp[items, 1] = lp[n:].float().cpu().numpy()
    model.load_counts()  # discard routing counts of the pre-pass
    model.train()


def mined_pairs(ctx: StageContext, model: Transformer) -> list[PrefPair]:
    d = ctx.cfg.dpo
    if d.model_sampled_prompts <= 0:
        return []
    buckets = buckets_from(ctx.cfg.data.mathworld)
    per = max(d.model_sampled_prompts // len(buckets), 1)
    problems = [
        p
        for ps in rl_prompts.rl_problems(
            ctx.cfg.data.mathworld, buckets, per, ctx.cfg.run.seed, "dpo-mine"
        ).values()
        for p in ps
    ]
    evaluator = ctx.evaluator()
    scored = evaluator.sample_problems(
        model,
        problems,
        d.samples_per_prompt,
        ctx.cfg.eval.temperature,
        ctx.cfg.eval.max_new_tokens,
        ctx.cfg.run.seed,
    )
    pairs = pairs_from_samples(problems, [[s.text for s in group] for group in scored])
    log.info("[dpo] mined %d verifier-ranked pairs from %d prompts", len(pairs), len(problems))
    return pairs


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, d = ctx.cfg, ctx.cfg.dpo
    model = ctx.build_model()
    pairs = build_pairs(cfg.data, d, cfg.data.seed) + mined_pairs(ctx, model)
    kinds = sorted({p.kind for p in pairs})
    encoded = [
        EncodedPair(
            encode_side(ctx.tok, p.prompt, p.chosen, cfg.train.seq_len),
            encode_side(ctx.tok, p.prompt, p.rejected, cfg.train.seq_len),
            kinds.index(p.kind),
        )
        for p in pairs
    ]
    n_val = max(int(len(encoded) * cfg.data.sft_val_fraction), 1)
    train_set = PairDataset(encoded[n_val:], ctx.tok.pad_id, cfg.run.seed)
    val_set = PairDataset(encoded[:n_val], ctx.tok.pad_id, cfg.run.seed)
    log.info(
        "[dpo] %s loss, %d train / %d val pairs, kinds %s", d.loss, len(train_set), n_val, kinds
    )
    plan = ctx.example_plan(model, rows_per_example=2)
    if d.loss != "simpo":
        for data in (train_set, val_set):
            fill_reference(ctx, model, data, max(plan.micro_batch_size, 1))

    def loss_fn(m: nn.Module, batch: PairBatch) -> LossOutput:
        lp, lens, out = summed_logps(m, batch)
        b = batch.kinds.shape[0]
        pl = preference_loss(
            d, lp[:b], lp[b:], batch.ref_logp[:b], batch.ref_logp[b:], lens[:b], lens[b:]
        )
        win = (pl.chosen_reward > pl.rejected_reward).float()
        metrics: dict[str, torch.Tensor] = {
            "loss": pl.loss.detach(),
            "aux_loss": out.aux_loss.detach(),
            "dpo/accuracy": win.mean(),
            "dpo/margin": (pl.chosen_reward - pl.rejected_reward).mean(),
            "dpo/reward_chosen": pl.chosen_reward.mean(),
            "dpo/reward_rejected": pl.rejected_reward.mean(),
            "dpo/logp_chosen_per_token": (lp[:b] / lens[:b].clamp_min(1)).detach().mean(),
            "dpo/logp_rejected_per_token": (lp[b:] / lens[b:].clamp_min(1)).detach().mean(),
        }
        for i, kind in enumerate(kinds):
            sel = (batch.kinds == i).float()
            metrics[f"dpo/accuracy/{kind}"] = (win * sel).sum()
            metrics[f"{COUNT_PREFIX}dpo/accuracy/{kind}"] = sel.sum()
        return LossOutput(pl.loss + out.aux_loss, metrics)

    evaluator = ctx.evaluator()

    @torch.no_grad()
    def evaluate(step: int) -> dict[str, float]:
        metrics = evaluator.run(model, step)
        model.eval()
        wins, total = 0.0, 0
        for start in range(0, len(val_set), plan.micro_batch_size):
            items = list(range(start, min(start + plan.micro_batch_size, len(val_set))))
            batch = val_set.collate(items).to(ctx.device)
            with ctx.runtime.autocast(), ctx.runtime.sdpa_context():
                out = loss_fn(model, batch)
            wins += float(out.metrics["dpo/accuracy"]) * len(items)
            total += len(items)
        model.load_counts()
        metrics["dpo/val_accuracy"] = wins / max(total, 1)
        return metrics

    provider = RowProvider(train_set, plan, ctx.dist.rank, ctx.device)
    hooks, bind = ctx.hooks(model, evaluate)
    trainer = ctx.trainer(model, plan, loss_fn, provider, hooks, units_name="pairs")
    bind(trainer)
    log.info("[dpo] batch plan %s (%d pairs/step)", plan, plan.units_per_step)
    result = trainer.fit()
    return ctx.finish(result.checkpoint, result.eval_metrics)


if __name__ == "__main__":
    stage_main("dpo")
