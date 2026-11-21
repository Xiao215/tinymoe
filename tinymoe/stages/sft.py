"""Stage 3 - supervised fine-tuning on chat-formatted data (single chat template).

Data: MathWorld problems answered as ``<think> worked steps </think>\\nAnswer: N`` plus
story-writing instructions. The loss covers assistant tokens only (``render_chat``'s
mask), including ``<|end|>`` so the model learns to stop. Rows are packed to
``train.seq_len`` with per-conversation attention isolation and RoPE positions
(``doc_ids`` / ``position_ids``), so packing changes throughput, not the objective.

``sft.lora.enabled`` trains low-rank adapters on a frozen model instead and merges them
before saving, so later stages always load a plain checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from tinymoe.data.sft_data import RowDataset, build_sft_examples, make_rows
from tinymoe.model.lora import apply_lora, merge_lora
from tinymoe.stages.common import StageContext, stage_main
from tinymoe.train import checkpoint, distributed

log = logging.getLogger(__name__)


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, s = ctx.cfg, ctx.cfg.sft
    examples = build_sft_examples(cfg.data, s.n_math, s.n_story, s.think_traces, cfg.data.seed)
    rows = make_rows(ctx.tok, examples, cfg.train.seq_len, s.packing, cfg.run.seed)
    n_val = max(int(len(rows) * cfg.data.sft_val_fraction), 1)
    train_rows, val_rows = rows[n_val:], rows[:n_val]
    log.info("[sft] %d examples -> %d train / %d val rows", len(examples), len(train_rows), n_val)
    train_set = RowDataset(train_rows, cfg.run.seed, isolate_docs=s.packing)
    val_set = RowDataset(val_rows, cfg.run.seed, isolate_docs=s.packing)

    model = ctx.build_model()
    if s.lora.enabled:
        n = apply_lora(model, s.lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(
            "[sft] LoRA r=%d on %d projections: %.2fM trainable", s.lora.rank, n, trainable / 1e6
        )
    evaluator = ctx.evaluator()

    def evaluate(step: int) -> dict[str, float]:
        metrics = evaluator.run(model, step)
        metrics.update(evaluator.rows_loss(model, val_set, len(val_rows), "val/loss/sft"))
        return metrics

    result = ctx.fit_lm(model, train_set, None, evaluate)
    final = result.checkpoint
    if s.lora.enabled:
        log.info("[sft] merged %d LoRA adapters", merge_lora(model))
        if ctx.dist.is_main and final is not None:
            checkpoint.save_model(final, model, {"step": result.step, "lora_merged": True})
        distributed.barrier()
    return ctx.finish(final, result.eval_metrics)


if __name__ == "__main__":
    stage_main("sft")
