"""LoRA (Hu et al., 2021): low-rank adapters on frozen ``nn.Linear`` projections.

``W x + (alpha / r) * B A x`` with ``A ~ Kaiming``, ``B = 0`` so the adapted model starts
exactly at the base model. Only adapter weights train; at the end of the stage the
adapters are merged (``W <- W + (alpha / r) B A``) so the next stage loads a plain model.

Targets are matched by module *name* (``q_proj``, ``gate_proj``, ...), so attention, dense
MLPs and shared experts can be adapted; routed experts are stored as stacked tensors (see
model/moe.py) and the router is left frozen - SFT on a small budget should not re-learn
routing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from tinylab.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scale
        return out

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        lin = lin.to(self.base.weight.device, self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + self.scale * self.lora_b @ self.lora_a)
        return lin


def _targets(model: nn.Module, names: Sequence[str]) -> list[tuple[nn.Module, str, nn.Linear]]:
    found = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if child_name in names and isinstance(child, nn.Linear):
                found.append((parent, child_name, child))
    return found


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> int:
    """Freeze ``model`` and wrap every target projection; returns the number wrapped."""
    model.requires_grad_(False)
    targets = _targets(model, cfg.targets)
    if not targets:
        raise ValueError(f"no nn.Linear named any of {cfg.targets} in the model")
    for parent, name, lin in targets:
        setattr(parent, name, LoRALinear(lin, cfg.rank, cfg.alpha))
    return len(targets)


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by its merged nn.Linear (in place); returns the count."""
    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
                merged += 1
    model.requires_grad_(True)
    return merged
