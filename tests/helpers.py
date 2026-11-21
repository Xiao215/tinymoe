"""Shared test utilities (importable from tests and the torchrun worker)."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import torch

from tinylab import hardware
from tinylab.config import BatchPlan, HardwareConfig, ModelConfig, TrainConfig, section_from_dict
from tinylab.data.batch import Batch
from tinylab.hardware import Runtime
from tinylab.logging import RunLogger
from tinylab.model.transformer import Transformer
from tinylab.train.distributed import DistInfo
from tinylab.train.loop import Hooks, LossFn, Trainer
from tinylab.train.losses import lm_loss

V = 61


@cache
def cpu_runtime(
    precision: str = "fp32", compile_: str = "off", backend: str = "inductor"
) -> Runtime:
    return hardware.resolve_runtime(
        HardwareConfig(
            device="cpu", precision=precision, compile=compile_, compile_backend=backend,  # type: ignore[arg-type]
            peak_tflops=1.0,
        )
    )  # fmt: skip


def tiny_model(moe: bool, seed: int = 0, **kw: Any) -> Transformer:
    base: dict[str, Any] = {
        "vocab_size": V, "d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "ffn_hidden": 64, "max_seq_len": 32,
        "moe": {"enabled": moe, "n_experts": 4, "top_k": 2, "expert_hidden": 32,
                "n_dense_layers": 1},
        "mtp": {"depth": 1},
    }  # fmt: skip
    base.update(kw)
    torch.manual_seed(seed)
    return Transformer(section_from_dict(ModelConfig, base))


class FixedBatch:
    """Always the same batch (overfitting tests)."""

    def __init__(self, batch: Batch) -> None:
        self.b = batch

    def batch(self, step: int, micro: int) -> Batch:
        return self.b

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class RandomRows:
    """Deterministic random token rows addressed by global row index."""

    def __init__(self, seq_len: int, vocab: int = V) -> None:
        self.seq_len = seq_len
        self.vocab = vocab

    def batch(self, indices: Any) -> Batch:
        rows = []
        for i in indices:
            g = torch.Generator().manual_seed(int(i))
            # a learnable pattern: x_{t+1} = (x_t * 3 + 1) mod vocab
            start = torch.randint(0, self.vocab, (1,), generator=g)
            seq = [int(start)]
            for _ in range(self.seq_len):
                seq.append((seq[-1] * 3 + 1) % self.vocab)
            rows.append(seq)
        t = torch.tensor(rows)
        return Batch(input_ids=t[:, :-1], labels=t[:, 1:])


def make_trainer(
    tmp: Path,
    model: Transformer,
    provider: Any,
    runtime: Runtime | None = None,
    plan: BatchPlan | None = None,
    loss_fn: LossFn | None = None,
    hooks: Hooks | None = None,
    dist: DistInfo | None = None,
    **train: Any,
) -> Trainer:
    cfg = section_from_dict(
        TrainConfig,
        {"max_steps": 20, "warmup_steps": 0, "log_interval": 1, "eval_interval": 10_000,
         "ckpt_interval": 10_000, "lr": 3e-3, "seq_len": 16, **train},
    )  # fmt: skip
    dist = dist or DistInfo()
    return Trainer(
        stage="test",
        model=model,
        runtime=runtime or cpu_runtime(),
        cfg=cfg,
        plan=plan or BatchPlan(micro_batch_size=4, grad_accum=1, world_size=1, unit=16),
        loss_fn=loss_fn or (lambda m, b: lm_loss(m, b, mtp_weight=0.3)),
        provider=provider,
        logger=RunLogger(tmp, enabled=dist.is_main),
        ckpt_dir=tmp / "ckpt",
        dist=dist,
        hooks=hooks,
        flops_per_unit=1.0,
    )


def read_losses(run_dir: Path, key: str = "loss") -> dict[int, float]:
    import json

    out = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if key in rec and rec[key] is not None:
            out[rec["step"]] = rec[key]
    return out
