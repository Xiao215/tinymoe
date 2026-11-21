"""Stage 1 - pre-training: next-token prediction on the TinyStories + MathWorld mixture.

Loss = CE + MoE balance terms + MTP (DeepSeek-V3) [+ z-loss]. The WSD schedule's stable
phase runs here; the decay is left to mid-training (``decay_frac: 0`` in the recipe), so
annealing happens on the high-quality mix.
"""

from __future__ import annotations

from typing import Any

from tinymoe.data.pretrain_mix import MixtureSampler, open_sources
from tinymoe.stages.common import StageContext, stage_main


def mixture(ctx: StageContext, weights: dict[str, float], seq_len: int) -> MixtureSampler:
    sources = sorted(k for k, w in weights.items() if w > 0)
    paths = ctx.shards(sources)
    return MixtureSampler(
        open_sources(paths, "train"),
        weights,
        seq_len,
        ctx.cfg.run.seed,
        ctx.tok.bos_id,
        ctx.cfg.data.doc_mask,
    )


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg
    sampler = mixture(ctx, cfg.data.pretrain_mix, cfg.train.seq_len)
    model = ctx.build_model()
    evaluator = ctx.evaluator()
    result = ctx.fit_lm(model, sampler, sampler.names, lambda step: evaluator.run(model, step))
    return ctx.finish(result.checkpoint, result.eval_metrics)


if __name__ == "__main__":
    stage_main("pretrain")
