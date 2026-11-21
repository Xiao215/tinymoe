# Decisions log

Judgment calls made while implementing [`PLAN.md`](../PLAN.md). Each entry: options, choice, why.

## D1 — Config library: dataclasses + PyYAML (not OmegaConf/pydantic)
Options: OmegaConf structured configs, pydantic models, plain dataclasses + YAML.
Choice: plain dataclasses with a small recursive validator (`tinylab/config.py`).
Why: zero extra deps, `mypy --strict` sees every field, unknown keys/type errors are
reported with "did you mean" hints. Group selection (`model=moe_mac`) and dot-overrides
(`train.lr=3e-3`) are ~80 lines. A custom YAML loader accepts `3e-4` as a float.

## D2 — Group-file scoping
`hardware/`, `model/`, `data/` files are section-scoped (their content *is* that section);
`student=<name>` loads a `configs/model/` file into the `student` section; `stage/` recipes
are root-level partials since a stage touches `train`, `eval`, and its own section.

## D3 — Python 3.12 via uv
The system Python is 3.15, which PyTorch wheels do not target. `.python-version` pins 3.12;
`requires-python = ">=3.11,<3.14"`.

## D4 — Loss scaling for fp16 is implemented in-house
`torch.amp.GradScaler` device coverage varies across backends/versions. A dynamic loss
scaler that reuses the gradient-norm computation (non-finite norm -> skip step, halve the
scale; N clean steps -> double) is backend-agnostic and trivially testable on CPU.

## D5 — `precision: auto` benchmarks instead of assuming
bf16 is chosen when the device reports native bf16 (CUDA). Otherwise each autocast dtype
the device supports is benchmarked against fp32 with a 1024² matmul; the fastest one with
≥ `low_precision_min_speedup` wins, preferring bf16 within 5% (no loss scaling needed).
On this M1 Pro that selects fp16 (1.26x).

## D6 — torch.compile only where recommended
`compile: auto` enables compile only when the device is CUDA *and* a probe compile
succeeds; MoE dispatch has data-dependent shapes, so compile gives little on MPS/CPU.
`compile: on` forces it anywhere, falling back to eager if the probe fails.

## D7 — Synthetic stories for smoke/CI, TinyStories (streamed) for real runs
`data.stories.source: synthetic|hf`. The smoke pipeline must run offline in minutes, so it
uses a deterministic template story generator. Real runs stream only `num_train_docs`
TinyStories rows and cache them as JSONL; download failure falls back to synthetic stories
with a warning (`fallback_to_synthetic`, on by default).

## D8 — Random-window packing over flat shards
Pre-training rows are random `seq_len+1` windows over the concatenated `<bos> doc <eos>`
stream, with the window chosen by an RNG seeded from `(seed, global_row_index)`. This is
nanoGPT-style packing; it makes resume and DDP sharding trivial (no loader state beyond the
step counter). Intra-document masking is optional (`data.doc_mask`).

## D9 — Verifier strictness
`Answer: N` must be the only non-empty line after the single `</think>`; answers inside
(closed or unclosed) think blocks never count; hedged/multiple answers parse as None;
non-ASCII digits and > 12 digits are rejected. Correctness (parsed == answer) and format are
reported separately so RL can weight them independently.
