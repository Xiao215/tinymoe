# Progress

| Milestone | Status | Validated |
|---|---|---|
| M0 scaffold | done | `tinylab info`, config/override/batch-math tests, hardware fallback tests, ruff, mypy |
| M1 tokenizer + data | done | digit splitting, round-trip, chat mask, mathworld determinism + expression semantics, 16 adversarial verifier cases, memmap shards, sampler determinism/mix ratio/doc positions, SFT packing masks, DPO pairs ranked by verifier |
| M2 model | done | shapes, causal mask, doc-mask isolation, KV cache == full forward (global + sliding window), left-padded batch == single, generation logprobs/routing match forward, MoE (loop/padded/auto × sigmoid/softmax) == naive per-token reference, dispatch gradients agree, capacity dropping, aux-free balancing → uniform time-averaged load, aux-loss gradients, routing replay, param-count formula (4 configs), MTP speculative decode == greedy, RoPE ntk/linear/yarn, capture leaves outputs bit-identical |
| M3 trainer | done (bench in M4) | overfit tiny batch (dense/MoE × AdamW/Muon), bit-identical resume after a crash (losses + params), fp32/bf16/fp16 on CPU and fp32/fp16/auto on MPS, loss scaler skip+backoff, compile path (eager backend), Newton–Schulz orthogonality, param roles, WSD/cosine shapes, autobatch search, 2-rank gloo DDP == 1-rank 2× accumulation |
| M4 pre/mid-training | done | pipeline composes (all three pipelines, exact batch math), manifest lineage, loss falls > 1 nat on every source, WSD anneal, context-extension artifact (before / RoPE-scaled / after), eval metrics + probe generations, rerun skips finished stages; NTK vs YaRN measured (D10) |
| M5 instrumentation | done | snapshot contents (layers, attention npz, MoE, logit lens, PCA), training bit-identical with and without hooks, bench smoke, snapshot overhead measured in bench |
| M6 viewer v1 | done | API tests (runs, manifest, metrics offset with a partial last line, snapshots, arrays, artifacts, SSE generator, path traversal), `tsc --noEmit` strict, `vite build` |
| M7 SFT + LoRA | done | packing masks, LoRA B=0 identity + exact merge (incl. MTP and shared-expert targets), LoRA params no_decay, held-out SFT loss logged |
| M8 DPO / IPO / SimPO | done | loss values vs closed forms, gradient prefers chosen, pair encoding / batch layout, reference pre-pass, dpo/* metrics in the smoke run |
| M9 GRPO family | done | batches reach the training device (on MPS too), rewards + DAPO overlong, variant presets, group advantages, clipping + aggregation, k3 KL >= 0, provider covers the buffer once per epoch, replayed forward == old log-probs, IcePop masking, accuracy-based dynamic sampling (D14), provider state round-trip, calibration artifact |
| M10 distillation | done | token KL == definition, KD losses vanish (value + gradient) when student == teacher, tau^2 scaling + CE mix, prompt mix, on-policy rollout layout / caching / statelessness, comparison table + best-mode rule, smoke pipeline produces all four modes with contiguous step spans |
| M11 viewer v2 | done | anatomy, attention (heatmap + arcs), MoE explorer, logit lens, embeddings, RL rollouts, report panels; every panel populated in the browser on the CPU smoke run (MoE and Logit lens on MoE snapshots; the dense distill student shows a dense notice) |
| M12 docs + final validation | done | `make check` green (ruff, ruff format, mypy strict, 177 tests, 1 skip that needs a machine without an accelerator); hard-coding lint test; CPU smoke pipeline 39 s; MPS smoke (`hardware=mac_m1_16gb`, fp16) 99 s; MPS smoke with forced fp32 / no compile / math SDPA / unfused AdamW 102 s; Mac bench 7.3 h, 7.8 GB peak of a 10 GB cap (D16); 2-process gloo DDP == single process; clean clone → `uv sync` → `make check` |

The M4–M11 work landed as one commit (`M4-M11: ...`) rather than one per milestone: it
was validated together, and the pipelines, the bench and the CLI reference every stage, so
earlier split points would not have passed `make check` on their own. That commit was
checked in a clean worktree before the docs commit.
