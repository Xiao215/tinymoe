"""``tinymoe inspect <checkpoint>``: model config, parameter counts and MoE router state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tinymoe.config import ModelConfig, section_from_dict
from tinymoe.model.config import param_count
from tinymoe.train import checkpoint


def describe(path: Path) -> dict[str, Any]:
    data = checkpoint.load(path)
    cfg: ModelConfig = section_from_dict(ModelConfig, data["model_config"])
    pc = param_count(cfg)
    state = data["model"]
    biases = {k: v.float() for k, v in state.items() if k.endswith("expert_bias")}
    out: dict[str, Any] = {
        "path": str(path),
        "keys": sorted(k for k in data if k != "model"),
        "step": data.get("step", (data.get("extra") or {}).get("step")),
        "params": {
            "total": pc.total,
            "active": pc.active,
            "embedding": pc.embedding,
            "routed_experts": pc.routed_experts,
            "tensors": len(state),
        },
        "model_config": data["model_config"],
    }
    if biases:
        out["router_bias"] = {
            k: {"min": float(v.min()), "max": float(v.max()), "std": float(v.std())}
            for k, v in biases.items()
        }
    return out
