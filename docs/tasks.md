# TinyLab task list (milestones → concrete steps)

Spec: [`PLAN.md`](../PLAN.md). Status lives in [`progress.md`](progress.md).

## M0 — Scaffold
- pyproject (uv; FastAPI/uvicorn are core deps, dev tools in the default `dev` dependency group), `.python-version`, `.gitignore`, `Makefile` (`check`, `lint`, `type`, `test`, `smoke`, `web`), `.pre-commit-config.yaml`.
- `tinylab/config.py`: typed dataclass schema; YAML group files (`hardware=`, `model=`, `student=`, `data=`, `stage=`); dot-overrides; strict unknown-key / type / cross-field validation; batch math (`derive_batch`).
- `tinylab/hardware.py`: `Capabilities` (device, count, memory, bf16/fp16, autocast, compile, fused AdamW, SDPA backends); `Runtime` resolution with logged fallbacks; device helpers (sync, memory, seeding) so no other module names a device.
- `tinylab/logging.py`: `RunLogger` (manifest, metrics.jsonl, snapshots, artifacts, rank-0 only) + structured std logging.
- `tinylab/cli.py`: `tinylab info`. Tests: config validation, overrides, batch math, hardware resolution, logger.

## M1 — Tokenizer + data
- `data/tokenizer.py`: byte-level BPE, individual digits, special tokens, single chat template (`render_chat`) returning ids + assistant mask.
- `data/stories.py`: TinyStories via HF streaming, cached to JSONL; deterministic synthetic story generator (offline/smoke); subject extraction for instructions.
- `data/mathworld.py`: difficulty buckets (digits × steps × ops), problem generators (arithmetic chains, word problems, number facts, algebra), worked traces, verifier (answer parse + format check).
- `data/pretrain_mix.py`: tokenize → memmap uint16/uint32 shards per source/split; position-addressable seeded mixture sampler; optional doc-boundary masks; long-sequence windows.
- `data/sft_data.py`, `data/pref_data.py`, `data/rl_prompts.py`.
- Tests: digits split, round-trip, determinism, adversarial verifier, memmap read, sampler determinism.

## M2 — Model
- norms, rope (ntk/linear/yarn scaling), attention (GQA, QK-norm, SDPA, KV cache, sliding/global), SwiGLU, MoE (router sigmoid/softmax, shared expert, aux-free bias, seq-aux, switch aux, z-loss, capacity, sorted dispatch, routing record/replay), transformer (MTP, generate with KV cache, speculative MTP decode), param counting, hooks.
- Tests: shapes, causal mask, KV-cache == full forward, MoE == naive ref, balancing near-uniform, param formula, routing replay.

## M3 — Trainer
- `train/optim.py` (Muon w/ Newton–Schulz + AdamW groups, fused/foreach fallback), `schedule.py` (WSD/cosine), `checkpoint.py` (atomic, RNG, provider state), `distributed.py` (DDP/gloo), `loop.py` (accum, AMP + loss scaler, clipping, eval/snapshot/ckpt hooks, metrics), `tools/bench.py` (+ `micro_batch_size: auto`).
- Tests: overfit (dense/MoE × AdamW/Muon), resume identical, precision paths forced, compile on/off, 2-proc gloo DDP == single-proc accumulation.

## M4 — Pre/mid-training
- `stages/common.py` (context: tokenizer/model/runtime/logger/manifest lineage), `stages/pretrain.py`, `stages/midtrain.py` (anneal + context extension), `eval/evaluator.py`, `eval/probes.py`, pipeline runner + `configs/pipeline/*.yaml`.
- Checks: smoke loss decreases per source; bench + short mac run with samples; expert specialization stats.

## M5 — Instrumentation
- `model/hooks.py` capture; `eval/snapshots.py` (per-layer norms, attention entropy/maps, MoE load/assignments/co-activation/bias, logit lens, embedding PCA, generations, grad/weight/update ratios).
- Tests: bit-identical loss with/without snapshots; overhead measured in bench.

## M6 — Viewer v1
- FastAPI server (typed models, runs/manifest/metrics/snapshots/report, SSE), React+TS+ECharts frontend (timeline, scalars, generations), API tests, `npm run build` type-checks.

## M7 — SFT
- packing with per-example isolation (or padding), assistant-only masks, optional LoRA (merged at end), math + story eval. Masking test.

## M8 — DPO
- DPO / IPO / SimPO with frozen reference; synthetic + model-sampled pairs; metrics.

## M9 — GRPO family
- rollout (batched KV-cache sampling, routing record), rewards, GRPO/Dr.GRPO/DAPO losses, dynamic sampling, routing replay + IcePop, calibration of buckets. Toggle tests.

## M10 — Distillation
- student pretrain → baseline SFT, off-policy, logit KD, on-policy reverse KL; automatic comparison table.

## M11 — Viewer v2
- anatomy + scrubber, attention heatmap/arcs, MoE explorer, logit lens, embeddings, stage comparison.

## M12 — Docs + final validation
- README, `docs/architecture.md`, `docs/techniques.md`, hard-coding lint test, clean-clone validation (Section 10).
