# TinyLab — an end-to-end LLM pre-training & post-training lab

> **Purpose:** experience the *whole* modern LLM lifecycle — tokenizer → pre-training → mid-training → SFT → preference tuning → RL → distillation — at a scale that trains on a MacBook (M1 Pro, 16 GB), using the same techniques frontier open-weight labs use (DeepSeek, Qwen, GLM, Kimi), and **see** every stage through a live visualization viewer.
>
> **Hard rules:** nothing hardware-specific is hard-coded; every number lives in config; the same code must run unchanged on CPU, Apple MPS, a single CUDA GPU, or multi-GPU.

---

## 0. Instructions to the implementing agent (read first)

You are implementing this plan autonomously. Work **milestone by milestone** (Section 9). For every milestone:

1. **Implement** the milestone fully — no stubs, no `TODO: implement`, no placeholder returns.
2. **Validate** it: run the unit tests, the smoke config, and the milestone's acceptance checks (Section 9). Run `ruff check`, `ruff format --check`, and `mypy` (or `pyright`) on the package.
3. **If anything fails, fix it and re-run** — keep looping until every check passes cleanly. Do not move on with a failing, flaky, or skipped check. Do not weaken a test to make it pass; fix the code. If a test is genuinely wrong, explain why in the commit message before changing it.
4. **Fallback checks:** when a feature depends on hardware/backend support (MPS op coverage, bf16, `torch.compile`, fused AdamW, FlashAttention/SDPA kernels, DDP), implement a detected fallback path and *test both paths* where the machine allows (force the fallback via config to test it). Log clearly which path was chosen at startup.
5. **Commit** with a clear message once the milestone is green (git init the repo at the start). Keep a `CHANGELOG.md` / `docs/progress.md` noting what was done, what was validated, and any known limitations.
6. **Keep going** to the next milestone without waiting for me, unless you hit a decision that is irreversible or genuinely ambiguous — then write the options in `docs/decisions.md`, pick the most reasonable default, note it, and continue.
7. At the end, run the **full final validation** (Section 10) from a clean clone/venv, and fix anything that breaks. Only stop when everything works correctly and cleanly.

Quality bar: idiomatic, typed (type hints everywhere, `from __future__ import annotations`), documented (docstrings explaining *why*, citing the technique/paper), small focused modules, no dead code, no copy-paste between stages (shared utilities), deterministic seeding, reproducible runs, structured logging (no bare `print` in library code).

---

## 1. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | |
| Env / deps | `uv` + `pyproject.toml` | optional extras: `[cuda]`, `[viewer]`, `[dev]` |
| DL framework | PyTorch ≥ 2.4 | device-agnostic; SDPA for attention |
| Tokenizer | HF `tokenizers` (train our own) | |
| Data | HF `datasets` (TinyStories) + synthetic generators | streamed/cached to disk |
| Config | dataclasses + YAML (OmegaConf or `pydantic`) with CLI dot-overrides | `python -m tinylab.pretrain model=moe_mac train.lr=3e-3` |
| Logging | own JSONL run logger + optional W&B / TensorBoard adapters | viewer reads the JSONL/snapshots |
| Viewer backend | FastAPI (+ WebSocket or SSE for live updates) | serves run data |
| Viewer frontend | Vite + React + TypeScript, Plotly or ECharts / D3 | static build served by FastAPI |
| Tests | `pytest`, `hypothesis` optional | |
| Lint / types | `ruff`, `mypy` (strict on `tinylab/`) | `pre-commit` config |

No CUDA-only dependency may be required. CUDA-only accelerators (flash-attn, liger kernels, vLLM, bitsandbytes) may be *optional* extras behind capability checks.

---

## 2. Project structure

```
tinylab/
  pyproject.toml
  README.md
  configs/
    hardware/        auto.yaml, mac_m1_16gb.yaml, cuda_24gb.yaml, cuda_80gb.yaml, cpu_smoke.yaml
    model/           smoke.yaml, dense_tiny.yaml, moe_mac.yaml, moe_gpu.yaml, student_dense.yaml
    data/            smoke.yaml, default.yaml
    stage/           pretrain.yaml, midtrain.yaml, sft.yaml, dpo.yaml, grpo.yaml, distill.yaml
    pipeline/        smoke.yaml, mac.yaml, gpu.yaml       # chains all stages
  tinylab/
    config.py        # dataclass schemas, YAML loading, overrides, validation
    hardware.py      # device/dtype/capability detection + fallbacks
    logging.py       # RunLogger: metrics JSONL, snapshots, artifacts, run manifest
    data/
      tokenizer.py   # train/load BPE, special tokens, chat template
      stories.py     # TinyStories download/cache/clean
      mathworld.py   # synthetic math "world": generators, difficulty knobs, verifier
      pretrain_mix.py# mixing, packing, binary shards (memmap)
      sft_data.py    # chat-formatted, masked examples
      pref_data.py   # chosen/rejected pairs
      rl_prompts.py  # prompt sets with difficulty buckets
    model/
      config.py
      norms.py       # RMSNorm
      rope.py        # RoPE (+ base scaling for context extension)
      attention.py   # GQA + QK-Norm + KV cache (+ optional MLA, sliding window)
      mlp.py         # SwiGLU
      moe.py         # router, fine-grained experts, shared expert, balancing
      transformer.py # model, MTP head (optional), generate()
      hooks.py       # instrumentation for the viewer
    train/
      optim.py       # AdamW, Muon (+ AdamW for embeddings/norms), param groups
      schedule.py    # WSD + cosine schedules
      loop.py        # shared trainer: accum, clipping, AMP, checkpointing, eval, logging
      checkpoint.py  # save/resume (model, optim, sched, RNG, data position)
      distributed.py # DDP init when WORLD_SIZE>1, no-op otherwise
    stages/
      pretrain.py  midtrain.py  sft.py  dpo.py  grpo.py  distill.py
    rl/
      rollout.py     # batched sampling with KV cache
      rewards.py     # verifiable rewards, format reward, length penalties
      grpo.py        # loss variants (GRPO / Dr.GRPO / DAPO options)
    eval/
      evaluator.py   # perplexity, pass@1, pass@k, per-difficulty accuracy
      probes.py      # fixed probe prompts used for snapshots across all stages
    tools/
      bench.py       # measures throughput/memory, estimates wall-clock per config
      inspect_ckpt.py
      chat.py        # chat REPL with any checkpoint (shows <think>)
  viewer/
    server/          # FastAPI app
    web/             # React/TS frontend
  tests/
  docs/
    architecture.md  techniques.md  decisions.md  progress.md
  runs/              # gitignored; one dir per run
```

---

## 3. Configuration & hardware abstraction (no hard-coding)

- **Every** hyper-parameter, path, size, batch, sequence length, dtype, step count, log interval and seed comes from config. Code contains only sane library defaults in dataclasses.
- `hardware.py` detects: device (`cuda` > `mps` > `cpu`, overridable), device count, memory, bf16/fp16 support, whether `torch.compile` works, whether fused AdamW is available, SDPA backend availability. It returns a `Capabilities` object; stages consult it, never `if mac:`.
- **Precision policy** from config: `auto | fp32 | bf16 | fp16`. `auto` picks bf16 on CUDA with bf16 support, otherwise benchmarks/chooses fp32 vs fp16 on MPS (M1 has no fast bf16). fp16 path uses a GradScaler (or loss scaling) where required. Master weights and optimizer states stay fp32.
- **Batch math is derived**, not hard-coded: user sets `tokens_per_step` (global batch in tokens) and `micro_batch_size`; grad-accumulation steps are computed from world size × micro batch × seq len. Optional `micro_batch_size: auto` finds the largest size that fits (binary search with OOM catch, MPS + CUDA).
- `tools/bench.py`: given any config, measures tokens/s, peak memory, and prints estimated wall-clock for each stage. README tells the user to run it first.
- Multi-GPU: `torchrun` works out of the box via `distributed.py` (DDP; FSDP optional later). Single-device runs need no launcher.
- Resume from any checkpoint mid-run with identical results (seeded RNG + data loader position saved).
- `pipeline/*.yaml` chains stages; each stage reads the previous stage's checkpoint path from the run manifest, not from a literal path.

---

## 4. Data — a small "world" where every stage has a visible effect

A 20–50M-active-param model cannot learn open-domain reasoning, so we design a **domain where it can succeed** and every stage shows a measurable change.

1. **Tokenizer** — byte-level BPE trained on our own corpus, vocab from config (default 8192). **Digits split individually** (standard for arithmetic: Llama-3/Qwen-style), special tokens: `<|bos|> <|eos|> <|pad|> <|user|> <|assistant|> <|system|> <|end|> <think> </think>`. A chat template function (single source of truth) used by SFT/DPO/RL/eval/chat.
2. **Pre-training mix** (ratios in config):
   - TinyStories (natural language, fluency).
   - `mathworld` synthetic text: equations, multi-step worked arithmetic, word problems in prose, number facts, simple algebra/variables. Generated deterministically from seeds; sizes in config.
   - Documents packed into fixed-length sequences with EOS separators; store as memmapped `uint16/uint32` shards; document-boundary-aware attention masking optional (config).
   - Held-out validation splits for each source.
3. **Mid-training / annealing mix:** higher-quality, more "reasoning-like" data (worked solutions with explicit steps, harder problems), up-weighted during the LR decay phase — mirrors modern "annealing on high-quality data". Also a **context-extension** step (e.g. 256 → 1024 tokens) with RoPE base adjustment.
4. **SFT data:** chat-formatted instruction data — math word problems with `<think>` step-by-step traces then `Answer: N`, plus story-writing instructions ("Write a short story about a cat who…") so the model learns to follow instructions in both domains. Loss only on assistant tokens.
5. **Preference data (DPO):** chosen = correct, well-formatted solution; rejected = wrong answer / missing format / rambling. Built synthetically (and optionally from SFT-model samples ranked by the verifier).
6. **RL prompts:** difficulty-bucketed (digits × steps × operation types). Includes a calibration utility that picks the buckets where the SFT model's pass@1 is ~10–60% (so RL has signal), stored in the run manifest.
7. **Verifier:** exact parsing of the final answer; separate format check (think tags present, one answer line). Used by RL, DPO data building, rejection sampling and evaluation.

---

## 5. Model — modern open-weight architecture, scaled down

Decoder-only transformer, all features toggleable in config:

- **Pre-norm RMSNorm**, no biases, optional tied embeddings.
- **RoPE** with configurable base; context extension by base scaling (YaRN-style optional).
- **Attention:** Grouped-Query Attention (configurable `n_kv_heads`), **QK-Norm**, PyTorch SDPA (uses Flash/efficient kernels when available, math fallback otherwise), KV cache for generation. Optional: sliding-window/global interleaving (Gemma/Kimi-style hybrid), and **Multi-Head Latent Attention (MLA)** as a stretch option.
- **FFN:** SwiGLU.
- **MoE (DeepSeek-V3/GLM-style):**
  - fine-grained routed experts (e.g. 16–64 small experts), **top-k** routing, **1 shared expert** always on;
  - first `n_dense_layers` layers dense (config), the rest MoE;
  - router with sigmoid or softmax scoring (config);
  - **auxiliary-loss-free load balancing** (per-expert bias updated by sign of load error, DeepSeek-V3) + small sequence-level balance loss; classic Switch-style aux loss and router z-loss available as alternatives for comparison experiments;
  - capacity factor / token dropping optional (default: no dropping);
  - efficient dispatch (sort/group tokens by expert, `index_add_`/scatter) — not a Python loop per token;
  - exposes routing stats every forward pass (for the viewer).
- **Multi-Token Prediction (MTP)** auxiliary head (DeepSeek-V3/GLM-style), optional, can be used for speculative decoding in `generate()`.
- **Initialization:** scaled init for residual projections; documented.
- Parameter counting utility reports **total vs active** params per token.

**Presets** (all in `configs/model/`, values are starting points — verify with `bench.py` and adjust):
- `smoke`: ~1M params, 2 layers, 4 experts — CI / CPU in < 5 min for the full pipeline.
- `moe_mac`: target ~100–150M total / ~25–40M active, seq 256→1024, must fit comfortably in 16 GB unified memory during training (leave headroom for the OS and viewer).
- `student_dense`: ~5–10M dense model for distillation.
- `moe_gpu`: larger (e.g. 0.5–1B total) to show the same code scales on a GPU box.

---

## 6. Training stages (modern recipe)

All stages share `train/loop.py` (gradient accumulation, clipping, AMP, eval, checkpoints, logging, snapshot hooks). Each stage = small file that defines data + loss.

### 6.1 Pre-training
- Objective: next-token cross-entropy (+ MoE balance loss + optional MTP loss, weights in config).
- Optimizer: **Muon** for 2-D hidden weight matrices (as used by Kimi K2 / GLM-4.5) with **AdamW** for embeddings, norms, router, head — selectable `optimizer: muon | adamw`. Implement Muon (Newton-Schulz orthogonalization) device-agnostically; test it against AdamW on the smoke config.
- LR schedule: **WSD (warmup–stable–decay)** default; cosine available. Decay phase hands off to mid-training.
- Gradient clipping, weight decay excluding norms/biases/embeddings (config), z-loss optional.
- Tracked: train/val loss per data source, perplexity, tokens seen, throughput, MFU estimate, grad norm, update/weight ratio, LR, expert load & entropy, dead experts, router bias values.

### 6.2 Mid-training (annealing + long context)
- Continue from pre-training checkpoint during the WSD decay with the high-quality mix; then context extension with RoPE scaling on longer packed sequences. Show val loss on long sequences before/after.

### 6.3 SFT
- Chat template, **assistant-only loss masking**, sequence packing with per-example attention isolation (or padding fallback), "cold-start" reasoning traces in `<think>`.
- Optional LoRA path (config) for comparison with full fine-tuning.
- Tracked: SFT loss, format-compliance rate, pass@1 per difficulty, story-instruction following (simple automatic checks: mentions the requested subject, length).

### 6.4 Preference optimization (DPO)
- Standard DPO (β in config) with frozen reference model; optionally SimPO/IPO variants.
- Tracked: reward margins, chosen/rejected log-probs, accuracy of implicit reward, downstream pass@1.

### 6.5 RL with verifiable rewards (GRPO family)
- Rollouts: batched sampling with KV cache, temperature/top-p from config, G samples per prompt.
- Rewards: correctness (verifier) + format reward + optional overlong penalty.
- Loss options (config flags, document each with its paper):
  - GRPO (group-normalized advantages, PPO-style clipped ratio, optional KL to reference with k3 estimator);
  - **Dr. GRPO** (no std normalization, no length normalization bias);
  - **DAPO** tricks: clip-higher (asymmetric ε), token-level loss aggregation, **dynamic sampling** (drop groups with all-same reward), overlong reward shaping.
- Multiple PPO epochs per rollout batch (config), with old-logprob caching.
- **MoE-specific:** record routing during rollout and optionally **replay routing** during the training forward pass; log the rollout-vs-train routing mismatch rate and logprob mismatch (train–inference mismatch is a known MoE RL instability; GLM-5's IcePop addresses this). Keep a toggle to compare.
- Keep MoE balancing active but low weight; log expert load drift vs SFT.
- Tracked: mean reward, pass@1 / pass@k per difficulty, response length, `<think>` length, entropy, KL to reference, clip fraction, fraction of groups filtered, routing mismatch.

### 6.6 Distillation
- Teacher = RL model; student = `student_dense` (and optionally a smaller MoE).
- Three modes, all implemented and comparable:
  1. **Off-policy / sequence-level** — rejection-sampled correct teacher traces → SFT the student.
  2. **Logit (token-level) KD** — forward-KL to teacher distribution on the same sequences (same tokenizer).
  3. **On-policy distillation** — student samples, teacher scores each token; minimize per-token reverse KL.
- Baseline: same student trained with SFT data only. Report the comparison table automatically.

### 6.7 Evaluation (shared, run at the end of every stage and periodically during)
- Story val perplexity; math pass@1/pass@k per difficulty bucket; format compliance; average think length; a fixed set of **probe prompts** whose generations are stored at every eval so the viewer can show "the same question answered over time".
- `eval/report.py` produces a cross-stage summary table (markdown + JSON).

---

## 7. Instrumentation (what gets recorded for the viewer)

`RunLogger` writes to `runs/<run_id>/`:

- `manifest.json` — config, git hash, hardware/capabilities, stage lineage, checkpoint paths.
- `metrics.jsonl` — scalar metrics per step (stage, step, tokens, wall time).
- `snapshots/<stage>/<step>/` — every N steps (config), on fixed probe inputs, small enough to stay cheap:
  - **per-layer**: residual-stream norm, attention entropy per head, attention maps for probe prompts (downsampled), MLP/expert output norms, gradient norm & weight norm & update/weight ratio per parameter group;
  - **MoE**: expert-load histogram per layer, routing assignment per token per layer for the probe prompts, router bias/score distributions, expert co-activation matrix;
  - **logit lens**: top-k tokens decoded from each layer's residual stream for probe positions;
  - **embeddings**: 2-D projection (PCA; UMAP optional) of token embeddings for digits/operators/common words;
  - **generations**: probe prompt completions (with `<think>`), reward/verifier result;
  - **RL**: sample rollouts per difficulty with rewards and advantages.
- Snapshots use a compact format (JSON for small, `.npz`/safetensors for arrays) with a schema version. Snapshot capture must not change training numerics (no-grad, eval mode toggled back, RNG state restored) — **test this**.
- Overhead budget: instrumentation ≤ ~5% of step time at default intervals (measure in `bench.py`).

---

## 8. Viewer (visualizing the model as it trains)

`tinylab viewer --runs runs/` starts FastAPI + built frontend; works live during training (SSE/WebSocket or polling) and offline on finished runs.

Pages / panels:
1. **Pipeline timeline** — one horizontal timeline across all stages (pretrain → mid → SFT → DPO → RL → distill) with stage boundaries; every metric chart can span the whole lifecycle.
2. **Scalars dashboard** — loss curves (per data source), LR, grad norm, throughput; RL reward, pass@1 per difficulty, response/think length, KL, entropy; comparison overlay of multiple runs (e.g. Muon vs AdamW, aux-loss vs aux-loss-free).
3. **Model anatomy** — a layer-stack diagram of the actual architecture (from config); click a layer to see its attention heads, expert grid, norms; a **step scrubber** to move through snapshots and watch it change.
4. **Attention viewer** — heatmaps per layer/head for probe prompts; token-to-token arc view.
5. **MoE explorer** — expert load per layer over time (heatmap: layer × expert × step), tokens colored by assigned expert, expert specialization view (which token types each expert handles), dead-expert alerts, routing-replay mismatch during RL.
6. **Logit lens** — per-layer predictions for a probe token, over training steps.
7. **Embedding space** — animated 2-D projection across steps (watch digits arrange themselves).
8. **Generations over time** — the same probe prompts answered at each checkpoint, side by side, with `<think>` collapsible and verifier result; diff view between stages.
9. **Stage comparison report** — the eval summary table and per-difficulty bar charts per stage.

Frontend quality: TypeScript strict, componentized, handles missing/partial data gracefully, dark/light mode, no hard-coded run names or sizes (everything from the API/schema). API has typed response models and tests.

---

## 9. Milestones & acceptance criteria

Each milestone must pass: unit tests, `ruff`, `mypy`, the listed checks, and the **smoke pipeline** still runs end-to-end.

| # | Milestone | Acceptance checks |
|---|---|---|
| M0 | Scaffold: repo, `uv`, pyproject, pre-commit, CI-style `make check`, config system, hardware detection, RunLogger | `tinylab info` prints detected capabilities; config overrides work; tests for config validation & batch math |
| M1 | Tokenizer + data (stories, mathworld, packing, shards, verifier) | digits tokenized individually; round-trip encode/decode; mathworld deterministic by seed; verifier unit tests incl. adversarial formats; shards readable via memmap |
| M2 | Model (dense + MoE, GQA, QK-Norm, RoPE, SwiGLU, KV cache, MTP) | shape tests; causal mask test (future tokens don't affect past logits); **KV-cache generation == full forward** numerically; MoE dispatch == naive reference implementation; balancing keeps load near-uniform on random data; param count total/active matches formula |
| M3 | Trainer + optimizers + schedules + checkpointing + `bench.py` | overfit a tiny batch to near-zero loss (dense and MoE, AdamW and Muon); resume gives identical loss trajectory; fp32/fp16/bf16 paths per capability with forced-fallback tests; `bench.py` prints estimates |
| M4 | Pre-training + mid-training stages | smoke: loss decreases on every source; mac preset: run `bench.py`, then a short real run producing coherent TinyStories-like samples; expert specialization visible in stats |
| M5 | Instrumentation + snapshots | snapshots don't change training numerics (bit-identical loss with/without, same seed); overhead measured and within budget |
| M6 | Viewer v1 (scalars, timeline, generations over time) | live-updates during a running smoke training; API tests; frontend builds & type-checks |
| M7 | SFT | assistant-only masking test; format compliance rises to high level; pass@1 reported per difficulty |
| M8 | DPO | implicit-reward accuracy > 50% and rising; no collapse of fluency (story ppl tracked) |
| M9 | GRPO family + rollout + routing replay | reward and pass@1 increase on calibrated buckets in a short run; dynamic-sampling and clip-higher toggles tested; routing mismatch metric logged; KL/entropy stable |
| M10 | Distillation (3 modes) + baseline | auto comparison table; distilled student ≥ baseline student on pass@1 |
| M11 | Viewer v2 (anatomy, attention, MoE explorer, logit lens, embeddings, stage comparison) | all panels render for smoke and mac runs; scrubber works across stages |
| M12 | Docs + final validation | Section 10 passes from clean environment |

---

## 10. Final validation (must pass before stopping)

1. Fresh clone, `uv sync`, `make check` (ruff, mypy, pytest) — all green, no warnings treated lightly.
2. `tinylab pipeline configs/pipeline/smoke.yaml` on CPU completes all stages in a few minutes and the viewer shows every panel populated.
3. The same smoke pipeline with `hardware=mac_m1_16gb` on MPS completes; with forced fallbacks (fp32, no compile, math SDPA) also completes.
4. `bench.py` on the `mac` pipeline prints per-stage time/memory estimates and they fit 16 GB.
5. Code paths for CUDA and DDP are exercised at least by tests that run on CPU with gloo backend (2 processes) to prove multi-process correctness without a GPU.
6. No hard-coded device names, paths, sizes, or step counts outside `configs/` (add a test/lint script that greps for `"mps"`, `"cuda"`, absolute paths, etc. outside `hardware.py`/configs).
7. README: quickstart (Mac), quickstart (GPU box), how to run each stage alone, how to open the viewer, expected timings from `bench.py`, and "experiments to try" (aux-loss vs aux-free balancing, Muon vs AdamW, GRPO vs Dr.GRPO vs DAPO, routing replay on/off, distillation modes, top-1 vs top-2 routing, shared expert on/off).
8. `docs/techniques.md`: for each technique, what it is, why industry uses it, where it lives in the code, and the paper/model it comes from.

---

## 11. Experiments the project should make easy (each is just a config override)

- MoE: aux-loss-free vs Switch aux loss vs none (watch collapse); top-1 vs top-2 vs top-4; shared expert on/off; number of experts.
- Optimizer: Muon vs AdamW at equal tokens.
- Schedule: WSD vs cosine; effect of annealing data in decay phase.
- SFT with vs without `<think>` traces; full FT vs LoRA.
- RL: GRPO vs Dr.GRPO vs DAPO tricks; KL on/off; G (group size); temperature; routing replay on/off.
- Distillation: off-policy vs logit KD vs on-policy; student size.
