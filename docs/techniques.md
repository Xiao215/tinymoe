# Techniques

Each entry says what the technique is, why production labs use it, where it is in the code,
and where it comes from. Config keys are given as `section.key` (see `tinymoe/config.py`
for every default). Judgment calls and measurements live in [decisions.md](decisions.md).

## Data and tokenizer

| Technique | What / why | Code | Source |
|---|---|---|---|
| Byte-level BPE, split digits | Every number is tokenized one digit at a time, so arithmetic sees consistent place values instead of arbitrary multi-digit merges. | `data/tokenizer.py` | GPT-2 byte-level BPE; per-digit splitting as in Llama 3 / Qwen |
| Single chat template | One `render_chat` builds ids **and** the assistant-only loss mask, used by SFT, DPO, RL, eval and chat. A train/inference template mismatch is a classic silent bug. | `data/tokenizer.py` | ChatML-style templates (Qwen, GLM) |
| Synthetic "MathWorld" | Deterministic, seeded arithmetic / word-problem generator with difficulty buckets (digits × steps × ops) plus an exact verifier, so each stage produces a measurable change in a 30M-active model. | `data/mathworld.py` | RLVR-style verifiable domains (Tulu 3, DeepSeek-R1) |
| Strict verifier | Parses exactly one `Answer: N` line outside `<think>`; the format check is separate. It catches hedging, answers placed inside think, full-width digits and absurd lengths. Used by DPO pairs, RL rewards, rejection sampling and eval. | `data/mathworld.py` (`verify`) | DeepSeek-R1 accuracy + format rewards; D9 |
| Memmapped token shards | `uint16/uint32` shards per source and split, read through `np.memmap`; a position-addressable seeded sampler means resume needs no loader state beyond the step. | `data/pretrain_mix.py` | nanoGPT / Megatron-style binary shards |
| Mixture sampling | Per-source weights (`data.pretrain_mix`, `data.midtrain_mix`); per-source losses are logged separately. | `data/pretrain_mix.py`, `train/losses.py` | Data-mixture practice (Llama, OLMo) |
| Document masking | Optional block-diagonal attention plus per-document RoPE positions inside packed rows (`data.doc_mask`, SFT packing). Packing then changes throughput, not the objective. | `data/batch.py`, `model/attention.py` | Llama 3 intra-document masking |

## Model

| Technique | What / why | Code | Source |
|---|---|---|---|
| Pre-norm RMSNorm, no biases | Stable deep training at low cost. | `model/norms.py`, `model/transformer.py` | Zhang & Sennrich 2019; Llama |
| RoPE (+ NTK / linear / YaRN scaling) | Rotary positions. Context extension rescales frequencies; NTK with factor = length ratio is the default (D10). | `model/rope.py` | Su et al. 2021; bloc97 NTK; Peng et al. 2023 (YaRN) |
| Grouped-query attention | `n_kv_heads < n_heads` shrinks the KV cache and bandwidth. | `model/attention.py` | Ainslie et al. 2023 |
| QK-Norm | RMSNorm on queries and keys bounds attention logits, which stabilizes higher learning rates (and Muon). | `model/attention.py` | Dehghani et al. 2023; OLMo 2, Qwen 3 |
| SDPA with fallback | `F.scaled_dot_product_attention` uses Flash / efficient kernels where available. `hardware.sdpa_backend` can force `math`; the chosen backend is logged. | `model/attention.py`, `hardware.py` | PyTorch SDPA |
| Sliding-window / global interleave | `model.sliding_window` + `global_every` gives Gemma/Kimi-style hybrid attention; the KV cache respects windows. | `model/attention.py` | Gemma 2/3, Mistral |
| SwiGLU | Gated FFN. | `model/mlp.py` | Shazeer 2020 |
| Fine-grained MoE + shared expert | Many small routed experts (top-k) plus an always-on shared expert; the first `n_dense_layers` stay dense. More specialization per FLOP. | `model/moe.py` | DeepSeekMoE (Dai et al. 2024), DeepSeek-V3, GLM-4.5 |
| Sigmoid routing, top-k renormalized | Scores `sigmoid(xW)`; selection uses `score + bias`, gates use the unbiased scores. | `model/moe.py` (`route`) | DeepSeek-V3 |
| Aux-loss-free balancing | Per-expert bias nudged by `sign(mean_load - load)` after every step, so no gradient interference. A tiny sequence-wise balance loss guards extremes. | `model/moe.py`, `train/loop.py` (`_moe_step`) | Wang et al. 2024; DeepSeek-V3 |
| Switch aux loss / router z-loss | Alternatives for comparison (`moe.balance: switch`, `moe.z_loss_weight`). | `model/moe.py` | Fedus et al. 2021; ST-MoE (Zoph et al. 2022) |
| Sorted dispatch (loop / padded / auto) | Assignments are sorted by expert. `padded` runs three batched matmuls, has a width cap (`padded_limit`) with exact per-expert overflow, and a worst-case mode for memory probing. `loop` does one grouped matmul per expert. No per-token Python loop. | `model/moe.py` (`_dispatch`) | Megablocks / GShard-style dispatch; D12 |
| Capacity factor (optional) | Drops overflow tokens; off by default. | `model/moe.py` | GShard (Lepikhin et al. 2020) |
| Multi-token prediction | Extra depth-`k` heads predict `t+k`; an auxiliary loss densifies the signal, and `generate_speculative` uses the MTP head as a draft model. | `model/transformer.py` | DeepSeek-V3, GLM-4.5; Gloeckle et al. 2024 |
| Scaled residual init, tied embeddings | `std / sqrt(2 · n_layers)` on residual outputs; optional weight tying. | `model/transformer.py` | GPT-2 |
| Total vs active params | `param_count` reports both, plus active non-embedding parameters (used for MFU). | `model/config.py` | Kaplan et al. 2020 (6N FLOPs/token) |
| KV-cache generation, routing record | Batched left-padded generation with per-token log-probs and recorded expert choices (for RL replay). | `model/transformer.py` (`generate`) | — |
| LoRA (optional SFT) | Low-rank adapters on chosen linears with B=0 init, merged into the weights at stage end. | `model/lora.py` | Hu et al. 2021 |

## Training system

| Technique | What / why | Code | Source |
|---|---|---|---|
| Muon + AdamW | Muon (momentum, then Newton–Schulz orthogonalization) for hidden 2-D matrices, including stacked experts; AdamW for embeddings, norms, router and head. Moonlight-style update scaling. | `train/optim.py` | Jordan et al. 2024; Moonlight (Liu et al. 2025); Kimi K2, GLM-4.5 |
| WSD schedule | Warmup, stable, decay. Pre-training stays in the stable phase (`decay_frac: 0`) and mid-training does the decay on high-quality data (annealing). | `train/schedule.py`, `configs/stage/*.yaml` | MiniCPM (Hu et al. 2024); DeepSeek |
| Derived batch math | `tokens_per_step` plus micro batch → gradient accumulation, from world size and sequence length. The global batch stays exact. | `config.py` (`derive_batch`) | — |
| Autobatch | `micro_batch_size: auto` probes a real optimizer step at its memory worst case: two accumulated passes, an lr-0 optimizer step on zeroed gradients, and MoE padded buffers at their cap. It extrapolates to skip sizes over budget and keeps `autobatch_headroom` of a hard memory cap. | `train/autobatch.py` | D12, D13, D16 |
| Hard memory cap | `hardware.memory_budget_gb` is enforced with per-process memory fractions on MPS/CUDA, so a 16 GB Mac raises OOM instead of swapping. | `hardware.py` (`cap_memory`) | — |
| Precision policy + fp16 loss scaling | `auto` uses bf16 on CUDA and benchmarks fp32 vs fp16 elsewhere in interleaved best-of-3 rounds (D5). The Mac preset pins fp16, because the M1 Pro sits at the threshold (D16). Master weights stay fp32. An in-house dynamic loss scaler skips steps with inf/nan gradients and backs off. | `hardware.py`, `train/loop.py` | Micikevicius et al. 2018 |
| `torch.compile` with fallback | Used only where it is known to work; `hardware.compile=off` forces eager. The decision is logged. | `hardware.py` | D6 |
| Fused / foreach AdamW | Picked from capabilities. | `train/optim.py` | — |
| Gradient clipping, z-loss, update ratio | Global-norm clipping. Optional output z-loss. Logged: per-group grad norms and `‖ΔW‖/‖W‖` sampled on log steps. | `train/loop.py`, `train/losses.py` | PaLM (z-loss) |
| Deterministic resume | Checkpoints hold model, optimizer, scaler, provider state and every RNG; batches are addressed by `(step, micro)`, so a resumed run is bit-identical (tested). | `train/checkpoint.py`, `train/providers.py` | — |
| DDP | `torchrun` works out of the box (gloo/nccl from capabilities). Row addressing makes `W` ranks × `A` accumulation equal 1 rank × `A·W` (tested). | `train/distributed.py`, `train/providers.py` | — |
| MFU estimate | `6·N_active + 12·L·d·T` FLOPs per token against `hardware.peak_tflops`. | `stages/common.py`, `train/loop.py` | PaLM appendix |

## Stages

| Technique | What / why | Code | Source |
|---|---|---|---|
| Pre-training with MTP + aux-free MoE | CE + balance terms + MTP + z-loss on the stories + MathWorld mix. | `stages/pretrain.py` | DeepSeek-V3 recipe |
| Annealing on high-quality data | LR decay phase on worked, harder solutions (`data.midtrain_mix`). | `stages/midtrain.py` | Llama 3, OLMo 2, MiniCPM |
| Context extension | Longer packed rows plus RoPE scaling; long-context validation loss before / after scaling / after training goes to `context_extension.json`. | `stages/midtrain.py` | Code Llama; YaRN; D10 |
| SFT, assistant-only loss, packing | Cold-start `<think>` traces, packed with per-conversation isolation; held-out SFT loss is logged. | `stages/sft.py`, `data/sft_data.py` | DeepSeek-R1 cold start; Tulu 3 |
| DPO / IPO / SimPO | Reference log-probs from a one-off pre-pass, so no second resident model. Pairs are synthetic (verifier-defined) plus mined from the model's own samples. | `stages/dpo.py`, `train/preference.py` | Rafailov et al. 2023; Azar et al. 2023 (IPO); Meng et al. 2024 (SimPO) |
| GRPO | Group-normalized advantages, PPO clipped ratio, k3 KL to reference, several PPO epochs × minibatches on cached old log-probs. Exact minibatch-wide loss aggregation across micro batches and ranks. | `rl/grpo.py`, `rl/rollout.py`, `stages/grpo.py` | DeepSeekMath (Shao et al. 2024) |
| Dr. GRPO | No std normalization and a constant length normalizer, which removes length / difficulty bias. | `rl/grpo.py` (`variant: dr_grpo`) | Liu et al. 2025 |
| DAPO | Clip-higher, token-level aggregation, dynamic sampling (drops groups whose accuracy is all 0 or all 1, with resampling), soft overlong punishment. | `rl/grpo.py`, `rl/rollout.py`, `rl/rewards.py` | Yu et al. 2025 |
| Difficulty calibration | Picks buckets where the starting policy's pass@1 lies in `[low, high]`, so groups have variance. | `stages/grpo.py`, `data/rl_prompts.py` | Common RLVR practice (Kimi k1.5, DAPO) |
| Routing replay | Experts chosen during rollout are forced in the training forward (gates still from the current router). Mismatch rate is logged. | `rl/rollout.py`, `model/moe.py` | MoE RL stability work (Qwen GSPO discussion, GLM-4.5) |
| IcePop | Masks tokens whose train/inference probability ratio leaves a band. | `rl/rollout.py` | GLM-5 / Ling team (2025) |
| Distillation ladder | Every student mode starts from the same base with the same step budget: ground-truth SFT baseline, sequence-level KD on verifier-filtered teacher samples, word-level logit KD (`τ²·KL(T‖S)`), and on-policy full-vocabulary reverse KL on student samples. Results go to an automatic comparison table. | `stages/distill.py`, `train/losses.py` (`token_kl`) | Hinton et al. 2015; Kim & Rush 2016; GKD (Agarwal et al. 2024); MiniLLM (Gu et al. 2024) |

## Evaluation and instrumentation

| Technique | What / why | Code | Source |
|---|---|---|---|
| pass@k (unbiased) | `1 - C(n-c, k)/C(n, k)` per problem, then averaged. | `eval/evaluator.py` | Chen et al. 2021 (Codex) |
| Probe generations | The same prompts are answered at every eval and saved as `generations` snapshots. | `eval/evaluator.py`, `eval/probes.py` | — |
| Internals snapshots | Residual / attention / FFN norms, attention entropy and maps, MoE load / bias / co-activation / specialization / assignments, logit lens, embedding PCA. Captured without changing numerics (no-grad, eval mode restored, RNG preserved; a test checks bit-identical training). | `eval/snapshots.py`, `model/hooks.py` | Logit lens: nostalgebraist 2020 |
| Live viewer | FastAPI reads the run directory (manifest, `metrics.jsonl`, snapshots, artifacts) and streams changes over SSE. The React/TS/ECharts panels cover timeline, scalars, anatomy, attention, MoE, logit lens, embeddings, generations, RL rollouts and the report. | `viewer/server`, `viewer/web` | — |
| Bench | Times real trainer steps (and the real stage losses) on the configured hardware, plus generation and snapshot overhead. Estimates per-stage wall-clock and peak memory; an OOM is reported as "does not fit". | `tools/bench.py` | D11 |
