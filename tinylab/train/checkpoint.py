"""Checkpoints: model, optimizer, step, RNG streams, loss scaler and data/provider state.

Everything needed for a bit-identical resume is stored (tested in tests/test_train.py).
Files are written to a temp name and renamed, so a crash never leaves a torn checkpoint.
Checkpoints are our own trusted files and contain numpy/python RNG state, so they are
loaded with ``weights_only=False``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from tinylab import hardware
from tinylab.config import ModelConfig, section_from_dict, to_dict
from tinylab.model.transformer import Transformer

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = torch.load(path, map_location=hardware.host_device(), weights_only=False)
    return data


def model_payload(model: Transformer) -> dict[str, Any]:
    return {"model": model.state_dict(), "model_config": to_dict(model.cfg)}


def save_model(path: Path, model: Transformer, extra: dict[str, Any] | None = None) -> Path:
    """Weights-only checkpoint (end of a stage): what the next stage initialises from."""
    return save(path, {**model_payload(model), "extra": extra or {}})


def load_model(path: Path, device: torch.device | None = None) -> Transformer:
    data = load(path)
    cfg: ModelConfig = section_from_dict(ModelConfig, data["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(data["model"])
    return model.to(device) if device is not None else model


def latest(directory: Path) -> Path | None:
    found = [
        (int(m.group(1)), p) for p in directory.glob("step_*.pt") if (m := _STEP_RE.search(p.name))
    ]
    return max(found)[1] if found else None


def prune(directory: Path, keep: int) -> None:
    found = sorted(
        (int(m.group(1)), p) for p in directory.glob("step_*.pt") if (m := _STEP_RE.search(p.name))
    )
    for _, p in found[: max(len(found) - keep, 0)]:
        p.unlink()
