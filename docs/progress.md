# Progress

| Milestone | Status | Validated |
|---|---|---|
| M0 scaffold | done | `tinylab info`, config/override/batch-math tests, hardware fallback tests, ruff, mypy |
| M1 tokenizer + data | done | digit splitting, round-trip, chat mask, mathworld determinism + expression semantics, 16 adversarial verifier cases, memmap shards, sampler determinism/mix ratio/doc positions, SFT packing masks, DPO pairs ranked by verifier |
| M2 model | done | shapes, causal mask, doc-mask isolation, KV cache == full forward (global + sliding window), left-padded batch == single, generation logprobs/routing match forward, MoE (loop/padded/auto × sigmoid/softmax) == naive per-token reference, dispatch gradients agree, capacity dropping, aux-free balancing → uniform time-averaged load, aux-loss gradients, routing replay, param-count formula (4 configs), MTP speculative decode == greedy, RoPE ntk/linear/yarn, capture leaves outputs bit-identical |
