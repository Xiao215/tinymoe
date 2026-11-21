# TinyMoE

An end-to-end LLM lab: **tokenizer → pre-training → mid-training → SFT → DPO → RL (GRPO
family) → distillation**, on a small modern MoE transformer. It trains on a MacBook (M1
Pro, 16 GB), and a live viewer lets you watch every stage. The same code runs on CPU,
Apple MPS, a CUDA GPU and multi-GPU with `torchrun`. Nothing hardware-specific is
hard-coded: devices, precision, batch sizes and paths all come from `configs/`.

* What each technique is and where it lives: [docs/techniques.md](docs/techniques.md)
* How the pieces fit: [docs/architecture.md](docs/architecture.md)
* Judgment calls and measurements: [docs/decisions.md](docs/decisions.md)
* Status: [docs/progress.md](docs/progress.md)

## Install

```bash
uv sync                      # Python 3.12, torch, FastAPI, dev tools (ruff, mypy, pytest)
make web                     # build the viewer frontend (needs Node 18+), once
uv run tinymoe info          # detected device, precision, compile, SDPA, memory
```

## Quickstart: CPU smoke run (a few minutes)

Every stage on a ~0.8M-parameter MoE with tiny synthetic data, to check the machinery:

```bash
uv run tinymoe pipeline configs/pipeline/smoke.yaml
uv run tinymoe viewer --runs runs          # http://127.0.0.1:8765
```

## Quickstart: MacBook (M1 Pro, 16 GB)

The smoke pipeline on the GPU (MPS, fp16) takes under two minutes:

```bash
uv run tinymoe pipeline configs/pipeline/smoke.yaml hardware=mac_m1_16gb
```

Then run the bench. It times real training steps of every stage on your machine, checks
peak memory against the 10 GB cap in `configs/hardware/mac_m1_16gb.yaml`, and prints a
wall-clock estimate:

```bash
uv run tinymoe bench configs/pipeline/mac.yaml --json runs/bench_mac.json
```

Then start the overnight pipeline, and open the viewer in a second terminal while it trains:

```bash
uv run tinymoe pipeline configs/pipeline/mac.yaml
```

```bash
uv run tinymoe viewer --runs runs
```

The pipeline survives interruptions. Rerunning with the same `run.run_id` skips finished
stages and resumes the current one from its last checkpoint:

```bash
uv run tinymoe pipeline configs/pipeline/mac.yaml run.run_id=<the run's id>
```

### Measured bench (M1 Pro 16 GB, `configs/pipeline/mac.yaml`)

Model: 106.3M total / 29.2M active parameters, fp16 autocast with loss scaling, MPS. Each
row times 6 real training steps (after 2 warm-up steps) at the micro batch autobatch picked;
"est. min" is the stage's full length in the pipeline. Generation runs at about 640 tok/s
(KV cache, batched).

| stage | model | batch | s/step | peak GB | est. min | notes |
|---|---|---|---|---|---|---|
| pretrain | model | 7x4 | 6.332 | 6.75 | 221.4 |  |
| midtrain/anneal | model | 7x4 | 6.702 | 6.77 | 60.9 |  |
| midtrain/context | model | 2x8 | 11.757 | 4.74 | 22.1 |  |
| sft | model | 16x2 | 5.156 | 7.82 | 31.5 |  |
| dpo | model | 8x4 pairs | 5.658 | 6.77 | 22.6 | reference pre-pass 172s |
| grpo | model | 128 rollouts, 4 updates of 32x2 | 3.596 | 6.76 | 39.2 | + reference model; generation 640 tok/s |
| distill/teacher_samples | model | 4000 samples | 0.000 | 1.14 | 10.1 | generation 640 tok/s |
| distill/student_pretrain | student | 32x1 | 0.718 | 4.42 | 8.9 |  |
| distill/baseline | student | 32x1 | 0.730 | 4.42 | 4.6 |  |
| distill/offpolicy | student | 32x1 | 0.710 | 4.42 | 4.5 |  |
| distill/logit | student | 16x2 | 2.057 | 4.46 | 11.3 | + teacher forward per micro batch |
| distill/onpolicy | student | 16 rollouts/step | 1.762 | 2.32 | 2.9 | + teacher forward |

**Total: about 440 min (7.3 h). Peak device memory 7.8 GB against the 10 GB cap; process
RSS 3.4 GB.** That leaves room for a browser with the viewer open. Rerun the bench on your
machine: the numbers move with thermals and with other apps using the GPU.

## Quickstart: GPU box

```bash
uv sync                      # on Linux the PyPI torch wheel ships with CUDA
uv run tinymoe bench configs/pipeline/gpu.yaml
uv run tinymoe pipeline configs/pipeline/gpu.yaml                      # one GPU
torchrun --nproc_per_node 8 -m tinymoe.cli pipeline configs/pipeline/gpu.yaml   # DDP
```

`tokens_per_step` is the global batch: more GPUs means fewer accumulation steps, not a
different run. For a 24 GB card use `hardware=cuda_24gb`.

## Running stages alone

Each stage reads its starting checkpoint from the run's `manifest.json`
(`stage.init_from: auto` means "the latest finished earlier stage"). To run one stage on
an existing run:

```bash
uv run tinymoe stage sft hardware=mac_m1_16gb model=moe_mac data=default run.run_id=<run id>
uv run python -m tinymoe.stages.grpo run.run_id=<run id> grpo.variant=dapo
```

Or start a stage from any checkpoint:

```bash
uv run tinymoe stage distill stage.init_from=runs/<id>/checkpoints/grpo/final.pt run.name=distill-only
```

Overrides are `group=file` (from `configs/<group>/`) or dotted keys (`train.lr=3e-4`). In a
pipeline, `stages.<name>.<key>=value` targets a single stage. Unknown keys, wrong types and
inconsistent batch math are rejected before anything runs.

Other tools:

```bash
uv run tinymoe report runs/<id>                            # cross-stage table (markdown + JSON)
uv run tinymoe chat runs/<id>/checkpoints/grpo/final.pt    # REPL, shows <think>
uv run tinymoe inspect runs/<id>/checkpoints/sft/final.pt  # config, params, router bias
```

## The viewer

`tinymoe viewer --runs runs` serves the API and the built frontend. It works live, with
server-sent events while a run trains, and offline on finished runs. Panels:

| Panel | Shows |
|---|---|
| Overview | the landing page: run status with stage ETA; headline numbers with their change; one card per stage (steps, time, pass@1 vs the previous stage, the number that says whether the stage did its job); key charts; math pass@1 by difficulty; an event feed with evals, stage changes and detected stalls, fp16 overflow skips, loss spikes and dead experts |
| timeline | all stages on one global-step axis; a snapshot scrubber drives the internals panels (Anatomy, Attention, MoE, Logit lens, Embeddings) |
| Metrics | small multiples grouped by area (filter by substring or `/regex/`), x axis as global step or wall clock, linear/log, EMA smoothing; loss per data source, val loss, LR, grad norm, update/weight ratio, throughput, MFU, MoE balance, math pass@1/pass@k per difficulty, think length, DPO margins, RL reward/KL/clip/entropy, distillation; overlay of other runs |
| Anatomy | the layer stack from the model config: heads coloured by attention entropy, experts by load, residual norms; click a layer for details |
| Attention | per layer/head heatmaps on probe prompts and token-to-token arcs |
| MoE | load (layer × expert and over time), tokens coloured by expert, specialisation by token type, co-activation, router bias, dead-expert alerts, RL routing-replay mismatch |
| Logit lens | top-k decoded from every layer at probe positions, and p(target) per layer over training |
| Embeddings | animated PCA of digit/operator/word embeddings |
| Generations | the same probe prompts answered at every eval, `<think>` collapsible, verifier verdicts, word diff between two checkpoints |
| RL rollouts | sampled rollouts per difficulty with reward and advantage |
| Report | eval summary per stage, pass@1 per bucket, distillation table, context-extension and RL-calibration results |

For frontend development: `cd viewer/web && npm run dev` (proxies `/api` to a running viewer).

## Experiments to try

Each is a single override, so compare runs side by side with the viewer's "compare" menu.

| Question | Override(s) |
|---|---|
| aux-loss vs aux-loss-free balancing (or none, to watch collapse) | `model.moe.balance=switch` / `aux_free` / `none` |
| Muon vs AdamW | `train.optimizer=adamw` (on pretrain) |
| GRPO vs Dr. GRPO vs DAPO | `stages.grpo.grpo.variant=dr_grpo` / `dapo` |
| routing replay on/off, IcePop | `stages.grpo.grpo.routing_replay=false`, `stages.grpo.grpo.icepop.enabled=true` |
| distillation modes | `stages.distill.distill.modes=[offpolicy,onpolicy]`, `distill.kd_temperature=2.0` |
| top-1 vs top-2 (vs top-4) routing | `model.moe.top_k=1` |
| shared expert on/off | `model.moe.n_shared_experts=0` |
| sigmoid vs softmax router | `model.moe.score_func=softmax` |
| DPO vs IPO vs SimPO | `stages.dpo.dpo.loss=ipo` / `simpo` |
| full SFT vs LoRA | `stages.sft.sft.lora.enabled=true` |
| context extension method | `midtrain.context_extension.rope_scaling.type=yarn` |
| MTP on/off | `model.mtp.depth=0` |
| dense vs MoE | `model=dense_tiny` (a small dense baseline) |

## Development

```bash
make check       # ruff lint + format check, mypy (strict), pytest
make smoke       # CPU smoke pipeline
make hardcode    # no device names / machine paths outside hardware.py and the config schema
```

Forced fallbacks (each tested and logged when chosen): `hardware.precision=fp32`,
`hardware.compile=off`, `hardware.sdpa_backend=math`, `hardware.fused_optimizer=off`,
`hardware.enforce_memory_budget=false`, `hardware.device=cpu`.

## Known limitations

* **Scale.** The Mac model has ~106M total / ~29M active parameters (with an 8k-token vocabulary) and trains for an
  overnight budget on TinyStories plus synthetic arithmetic. Expect fluent short stories
  and decent pass@1 on easy and medium arithmetic, not general reasoning. The lab is about
  watching the techniques work, not about the final model.
* **Data.** Math and the verifier are synthetic (integer arithmetic with a strict
  `Answer:` format). DPO preferences come from the verifier and from length/format rules,
  not from human labels.
* **MPS.** The M1 has no bf16 tensor support, so the Mac preset uses fp16 autocast with
  a dynamic loss scaler. That is pinned, not `auto`: the measured fp16 speedup on the
  M1 Pro straddles the auto threshold (D16). `hardware.precision=fp32` is the tested
  fallback. `compile: auto` leaves `torch.compile` off outside CUDA, because
  MoE dispatch has data-dependent shapes (D6); `hardware.compile=on` forces it. Kernel-launch overhead dominates on MPS, which is why MoE uses padded
  dispatch (D12). MPS reports out-of-memory late, so the memory cap is enforced
  per process; another heavy app running at the same time can still push the machine into
  swap.
* **Parallelism.** Data-parallel DDP only (tested with 2 gloo processes on CPU, and
  configured for NCCL). No FSDP, tensor or expert parallelism: every rank holds all
  experts, which is fine at this size. Rollouts and teacher sampling are generated per
  rank (rank 0 for the teacher) without a separate inference engine.
* **MoE kernels.** Experts run as one padded batched matmul (plus an exact overflow path),
  or as a per-expert loop. There are no grouped-GEMM or Triton kernels.
* **RL.** GRPO generation uses the training model's own `generate` (no vLLM). IcePop and
  routing replay therefore guard against the training/inference mismatch that MoE routing
  introduces between rollout and update passes, not against a different inference engine.
* **Viewer.** It is a local tool: it has no authentication and binds to 127.0.0.1 by
  default. It polls the run directory for changes, so it reads runs on local or shared
  disks, not object storage.
* **Bench numbers** are from one M1 Pro with nothing else running. Thermals and other apps
  change them; rerun `tinymoe bench` before a long run on another machine.
