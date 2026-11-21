"""Stage 2 - mid-training: anneal on high-quality data, then extend the context window.

1. *Annealing* (Llama-3 §3.1.3, OLMo-2 "mid-training", MiniCPM): the WSD decay phase runs
   on a mixture up-weighting worked, multi-step solutions (``math_hq``).
2. *Context extension* (Code Llama, Qwen2.5, DeepSeek-V3 YaRN): rebuild RoPE with base /
   YaRN scaling and train briefly on longer packed sequences. Long-sequence validation
   loss is measured before and after, so the effect is visible.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from tinymoe.stages.common import StageContext, stage_main
from tinymoe.stages.pretrain import mixture

log = logging.getLogger(__name__)


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg
    model = ctx.build_model()
    evaluator = ctx.evaluator()
    sampler = mixture(ctx, cfg.data.midtrain_mix, cfg.train.seq_len)
    result = ctx.fit_lm(model, sampler, sampler.names, lambda step: evaluator.run(model, step))
    summary = dict(result.eval_metrics)
    ce = cfg.midtrain.context_extension
    final = result.checkpoint
    if ce.enabled and ce.steps > 0:
        model.cfg.max_seq_len = max(model.cfg.max_seq_len, ce.seq_len)
        model.rope.reset(model.cfg.max_seq_len, model.cfg.rope_base, model.cfg.rope_scaling)
        before = evaluator.lm_losses(model, ce.seq_len, prefix="val_long")
        ratio = ce.seq_len / cfg.train.seq_len
        factor = ce.rope_scaling.factor if ce.rope_scaling.factor > 1 else ratio
        scaling = replace(ce.rope_scaling, factor=factor, original_max_seq_len=cfg.train.seq_len)
        model.extend_context(max(model.cfg.max_seq_len, ce.seq_len), scaling)
        log.info("[midtrain] context %d -> %d with %s", cfg.train.seq_len, ce.seq_len, scaling)
        scaled = evaluator.lm_losses(model, ce.seq_len, prefix="val_long")
        ext_train = replace(
            cfg.train,
            max_steps=ce.steps,
            seq_len=ce.seq_len,
            lr=ce.lr,
            schedule="constant",
            warmup_steps=0,
            tokens_per_step=ce.tokens_per_step,
            micro_batch_size=ce.micro_batch_size,
            eval_interval=max(ce.steps, 1),
            eval_batches=ce.eval_batches,
        )
        long_sampler = mixture(ctx, cfg.data.midtrain_mix, ce.seq_len)
        ext = ctx.fit_lm(
            model,
            long_sampler,
            long_sampler.names,
            lambda step: evaluator.lm_losses(model, ce.seq_len, prefix="val_long"),
            train=ext_train,
            ckpt_subdir="context_extension",
            step_offset=result.step,
        )
        final = ext.checkpoint
        final_eval = evaluator.run(model, result.step + ext.step)
        ctx.logger.log_metrics(ctx.stage, result.step + ext.step, final_eval)
        summary.update(final_eval)
        after = ext.eval_metrics
        summary.update({k.replace("val_long", "val_long_before"): v for k, v in before.items()})
        summary.update({k.replace("val_long", "val_long_scaled"): v for k, v in scaled.items()})
        summary.update({k.replace("val_long", "val_long_after"): v for k, v in after.items()})
        ctx.logger.write_artifact(
            "context_extension.json", {"before": before, "rope_scaled": scaled, "after": after}
        )
    return ctx.finish(final, summary)


if __name__ == "__main__":
    stage_main("midtrain")
