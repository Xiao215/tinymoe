# Architecture

How the pieces fit together. Technique details are in [techniques.md](techniques.md);
judgment calls in [decisions.md](decisions.md).

## Layers

```
configs/*.yaml ──► config.py (typed dataclasses, groups, dot-overrides, validation)
                       │
hardware.py ◄──────────┤  Capabilities (device, dtypes, compile, SDPA, fused AdamW, memory)
  └─ Runtime            │  → Runtime: device, precision, autocast, sdpa context, memory cap
                       ▼
pipeline.py ──► stages/<name>.py ──► stages/common.py (StageContext)
                                       ├─ data/*        tokenizer, shards, SFT/DPO/RL data
                                       ├─ model/*       Transformer, MoE, LoRA, hooks
                                       ├─ train/*       Trainer, optimizers, losses, providers
                                       ├─ rl/*          rewards, GRPO family, rollouts
                                       └─ eval/*        evaluator, probes, snapshots, report
                                       ▼
                               logging.RunLogger  ──► runs/<run_id>/  ◄── viewer/server (FastAPI)
                                                                              ▲
                                                                       viewer/web (React/TS)
```

* **No module but `hardware.py` names a device**, and nothing hard-codes a path or size
  (`tests/test_hardcode.py` enforces both). Stages consult the `Runtime`, never "if mac".
* **Every stage is a small file** that builds data, a loss function and a provider, then
  hands them to the shared `Trainer` (`train/loop.py`) through `StageContext`. Gradient
  accumulation, AMP and loss scaling, clipping, schedules, checkpoints and resume, eval and
  snapshot hooks, metrics, and MoE bias updates live in the trainer once.

## Run directory (the file contract)

`RunLogger` writes on rank 0 only. The viewer reads nothing else:

```
runs/<run_id>/
  manifest.json          run id, hardware/capabilities, runtime summary, tokenizer, and per
                         stage: status, config, init checkpoint, final checkpoint, summary
  metrics.jsonl          {"stage", "step", "time", <metric>: value, ...} per log/eval step
  log.txt                structured log
  tokenizer.json
  checkpoints/<stage>/[<sub>/]step_*.pt, final.pt
  snapshots/<stage>/<step:08d>/<name>.json (+ <name>.npz)
                         name = data (internals), generations, rollouts
  artifacts/             report.json/.md, context_extension.json, calibration.json,
                         distill_comparison.json
  distill/teacher_samples.json
```

Stage **lineage** comes from the manifest: `stage.init_from: auto` resolves to the
latest finished earlier stage, so pipelines never contain checkpoint paths. A pipeline
rerun skips stages marked done.

## Training loop contract

```python
loss_fn(model, batch) -> LossOutput(loss, metrics)   # metrics: detached scalars;
                                                      # "_count/<k>" makes <k> a ratio of sums
provider.batch(step, micro) -> batch.to(device)       # deterministic in (step, micro)
provider.state_dict() / load_state_dict()             # resume
```

Providers:

* `RowProvider` over any row-indexed source (packed memmap windows, SFT rows, DPO pairs).
  Rows are addressed by `((step·A + micro)·W + rank)·B`, so `W` ranks × `A` accumulation
  steps see the same rows as one process with `A·W` steps.
* `RolloutProvider` (GRPO): maps trainer steps to (iteration, epoch, minibatch); rolls out
  at each iteration start; its buffer is part of the checkpoint.
* `OnPolicyRollouts` (distillation): samples the student per step; stateless.

## Batch sizing

`train.tokens_per_step` (or `examples_per_step`) is the **global** batch. `derive_batch`
turns it into micro batch × accumulation × world size, lowering the micro batch to the
largest divisor so the global batch stays exact. `micro_batch_size: auto` runs the
autobatch probe (`train/autobatch.py`): a real optimizer step at its memory worst case,
under the hard cap `hardware.memory_budget_gb` × `autobatch_headroom`.

## Generation

`Transformer.generate` does batched, left-padded sampling with a KV cache, temperature and
top-p. It returns per-token log-probs and (optionally) the experts chosen per layer. Eval,
DPO pair mining, GRPO rollouts, teacher sampling, on-policy distillation and `tinylab chat`
all use it through `Evaluator.generate`.

## Viewer

* `viewer/server/runs.py`: `RunStore` reads the run directory. Metrics are read
  incrementally by byte offset (a partly written last line is deferred). Global steps
  (`gstep`) concatenate stage spans in manifest order, so every chart spans the lifecycle.
* `viewer/server/app.py`: typed FastAPI endpoints (runs, manifest, metrics, snapshots,
  arrays, artifacts) plus `/events`, an SSE stream that polls file signatures and pushes
  new metric records, snapshot/manifest/artifact changes.
* `viewer/web`: React 18 + TypeScript (strict) + ECharts. `useRun` keeps one run current
  through SSE. Panels are a registry in `App.tsx`; each one renders with partial or missing
  data.
