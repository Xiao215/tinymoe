"""SwiGLU feed-forward (Shazeer, 2020): ``down(silu(gate(x)) * up(x))``, no biases."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out
