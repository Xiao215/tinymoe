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
D16 replaces the single short timing with interleaved best-of-3 rounds. The M1 Pro still
lands on both sides of the threshold (1.07–1.21x), so the Mac preset pins fp16 (D16).

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

## D10 — Context extension defaults to NTK base scaling with factor = length ratio
Measured on the smoke pre-train checkpoint (trained at 128, evaluated at 256, before any
extension training), loss on the [first half, second half] of the window:

| scaling | stories | math |
|---|---|---|
| none (extrapolate) | 0.27 / 0.34 | 1.03 / 0.98 |
| ntk x2 | 0.27 / **0.30** | 1.04 / **0.95** |
| linear x2 (PI) | 0.52 / 0.64 | 1.54 / 1.60 |
| yarn x2 | 0.43 / 0.46 | 1.19 / 1.18 |
| yarn x4 | 0.80 / 0.95 | 1.46 / 1.55 |

NTK leaves short-range frequencies untouched and is the only option that helps without
training. YaRN and PI need far more extension steps than a tiny model gets to recover. The
recipe therefore uses `rope_scaling: {type: ntk}`, and `factor <= 1` means
`context_extension.seq_len / train.seq_len`. YaRN/linear stay available, and the
before / rope-scaled / after losses land in `artifacts/context_extension.json` so the
comparison is one override away.

## D11 — Bench measures real code paths, and generation estimates are upper bounds
`tinylab bench` runs the actual Trainer (autocast, loss scaling, optimizer, MoE bias
update), the KV-cache generator and a captured snapshot forward for a few steps on random
data. It then multiplies by the counts in each stage's config. Random weights never emit a
stop token, so every generation runs to `max_new_tokens`, which makes rollout/eval
estimates upper bounds. Peak memory is device memory where the backend reports it (CUDA
allocator peak, MPS driver allocation), otherwise process RSS. The budget check uses the
max of device peak and RSS.

## D12 — Mac preset: padded MoE dispatch, 16k-token steps, hard memory cap
Measured on the M1 Pro (fp16 autocast, `moe_mac`, micro batch 8 × 512, fwd+bwd):

| variant | tok/s |
|---|---|
| dense with the same active size | 5.4k |
| MoE `dispatch: padded` | **3.3k** |
| MoE `dispatch: loop` | 2.4k |
| MoE `dispatch: auto` | 2.0k |
| 16 experts top-2 (hidden 512), padded | 3.6k |

Generation (KV cache, 64 × 96 tokens): padded 900 tok/s, auto 710 tok/s.
On MPS, kernel-launch overhead dominates: one batched matmul over padded experts beats a
per-expert loop even when padding wastes compute. `auto` is the slowest of the three
because it flips per layer as the load changes, which churns the allocator. `moe_mac`
therefore pins `dispatch: padded`. `auto` remains the default for other presets, and
`dispatch: loop` is still selectable (tested in test_model).

Memory: an early bench run with only an *advisory* budget pushed the machine into swap
(MPS pages GPU buffers out instead of raising OOM). `hardware.memory_budget_gb` is now
applied as a hard per-process cap (`torch.mps.set_per_process_memory_fraction`, CUDA
equivalent). Autobatch extrapolates peak memory linearly and does not try sizes projected
over budget. It reserves 8 bytes/param of optimizer state and keeps
`autobatch_headroom` (0.85) of the remainder. `enforce_memory_budget: false` restores
the advisory behaviour (tested).

At about 3.3k tok/s, 16k-token steps take about 5 s, which sizes the Mac pipeline at
about 7–8 h overnight (`tinylab bench configs/pipeline/mac.yaml` prints the current
estimate).

## D13 — Memory probing at the worst case (supersedes D12's 8 bytes/param reservation)
Later Mac bench runs still hit the 10 GB cap *during* training, after autobatch had passed.
Three gaps between the probe and a real step, all measured:
1. Optimizer state, Newton–Schulz workspace, and gradients held across accumulation. At
   micro batch 8 × 512, a forward/backward alone peaked at 6.6 GB; the real step peaked at
   8.75 GB. The probe now runs two accumulated passes plus a real optimizer step. The step
   runs at lr 0 on zeroed gradients, so the weights provably do not change, even if fp16
   random-token gradients overflow. The 8 bytes/param guess is only used when no train
   config is given.
2. MoE padded dispatch sizes its buffer by the busiest expert. A fresh router is
   near-balanced, so the probe saw small buffers that training later outgrew. Under
   imbalance an unbounded buffer can grow up to `n_experts`×. The width is now capped at
   `moe.padded_limit` (1.5) × the balanced load. Assignments past the cap (contiguous in
   sorted order) run through the per-expert path for just the hot experts, so results stay
   exact (tested against `loop`, with and without a capacity factor). The probe pins every
   buffer at the cap (`worst_case_routing`), so the micro batch it picks covers any routing
   training can reach, up to a small overflow absorbed by the 0.85 headroom.
3. The bench now reports an OOM as "does not fit" (with a traceback) instead of aborting.
   It also caught a real config error before any long run: SFT's 384-token rows do not
   divide 16384 tokens/step.

## D14 — DAPO dynamic sampling filters on accuracy, not total reward
PLAN.md paraphrases dynamic sampling as "drop groups with all-same reward". DAPO
(Yu et al., 2025) filters prompts whose *accuracy* is 0 or 1. The two differ once format
and overlong terms are in the reward: a group whose answers are all wrong still has reward
variance from length, and its advantages would then train length and layout rather than
reasoning. We follow the paper (`rl/rollout.py`, test_rl).

The same test file had asserted that `"Answer: 5"` (no think block) earns no correctness
reward. That contradicts the verifier spec pinned by test_data (correct, format not ok)
and DeepSeek-R1's separate accuracy and format rewards. The assertion was corrected to the
spec, with an added case for an answer hidden inside `<think>` (no reward).

## D15 — Distillation design
* Every student mode starts from one shared base (a short next-token pre-training of the
  student on the pre-training mixture) and gets the same optimizer-step budget. Without the
  base, the off-policy modes would also be teaching basic language modeling, and the
  comparison would measure that instead.
* Logit KD runs on the *same* teacher samples as sequence-level KD. Their difference is
  then exactly "full distribution vs sampled token". The baseline uses ground-truth SFT
  data over the same prompt mix (3 math : 1 story).
* On-policy distillation uses the exact full-vocabulary reverse KL at student-sampled
  prefixes (GKD), not a sampled-token REINFORCE estimate: same tokenizer, small vocab, and
  zero variance. It starts from the off-policy student when that mode ran, as GKD advises.
* Teacher samples are generated once on the main rank and shared through the run
  directory, so DDP ranks do not each sample the teacher and a rerun reuses them.

## D16 — Mac sizing from the full bench: fp16 pinned, 14336-token steps, ~7.3 h
The first full `tinylab bench configs/pipeline/mac.yaml` (every stage, real optimizer
steps) found three things:
1. **Precision was not stable.** `precision: auto` timed one 1024² matmul per dtype. That
   selected fp16 at 1.26x in one process and fp32 (fp16 at 1.00x) in the next. The probe now
   interleaves the dtypes over three rounds at 2048² and keeps each dtype's best round
   (`hardware.best_tflops`, tested). That removed the cold-start noise. But across
   processes the M1 Pro still measured fp16 at 1.07x, 1.08x, 1.09x, 1.12x and 1.21x, on
   both sides of `low_precision_min_speedup` (1.15), so `auto` cannot decide reliably.
   End to end, the midtrain anneal row (same model, same 7 × 4 × 512 plan) of two
   benches with nothing else running:

   | precision | s/step | peak GB |
   |---|---|---|
   | fp32 | 7.17 | 7.78 |
   | fp16 + loss scaling | 6.70 | 6.77 |

   The Mac preset therefore pins `precision: fp16`, which is about 7% faster and uses
   13% less memory. `hardware.precision=fp32` stays a tested override. `auto` remains
   the default elsewhere, and on CUDA it picks bf16 without timing anything.
2. **The micro batch the probe picks has to divide the global batch.** Autobatch picks 7
   rows of 512 for pretrain and midtrain, but 16384 tokens is 32 rows, so `derive_batch`
   (which keeps the global batch exact) fell back to 4 × 8. At 14336 tokens = 28 rows the
   plan is 7 × 4: 7.1 s instead of 8.6 s per step.
3. **The optimizer is a fixed cost per step.** One Muon step (5 Newton–Schulz iterations
   over 102M parameters in fp32) takes 0.63 s, against 0.03 s for AdamW and about 2 s for a
   7 × 512 forward and backward. It does not depend on the micro batch, so it shrinks per
   token only as the global batch grows. At 14336 tokens it is about 10% of a step.

The first bench estimated 10.3 h, mostly pretraining (2400 steps × 8.6 s). (A second
bench run is discarded: a test run on the GPU at the same time doubled its pretraining
step time.) The preset now runs 2000 pretraining steps of 14336 tokens (28.7M tokens, about one token per active
parameter: a short run by Chinchilla standards, sized to one night), and the other stages
are unchanged. The final bench estimates 7.3 h with a 7.8 GB peak (table in the README).

The earlier 3.3k tok/s in D12 timed forward and backward only, on a fresh router, with an
uncapped padded buffer. A full step at the probed worst-case micro batch is the number
that sizes a run.
