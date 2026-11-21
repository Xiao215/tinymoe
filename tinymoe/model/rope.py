"""Rotary position embeddings (Su et al., 2021) with context-extension scaling.

Scaling options (``model.rope_scaling.type``):

* ``ntk``    - NTK-aware base scaling: ``base' = base * s^(d/(d-2))`` (bloc97, 2023;
  what Code Llama / Qwen do when they raise ``rope_theta`` for long context).
* ``linear`` - position interpolation, positions divided by ``s`` (Chen et al., 2023).
* ``yarn``   - NTK-by-parts interpolation + attention temperature ``0.1 ln s + 1``
  (Peng et al., 2023; used by DeepSeek-V3, Qwen2.5-1M).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tinymoe.config import RopeScalingConfig


def inv_frequencies(
    head_dim: int, base: float, scaling: RopeScalingConfig, max_seq_len: int
) -> tuple[torch.Tensor, float]:
    """Inverse frequencies (head_dim/2,) and the cos/sin magnitude scale."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    s = scaling.factor
    if scaling.type == "ntk" and s != 1.0:
        base = base * s ** (head_dim / (head_dim - 2))
    inv = 1.0 / (base**exponents)
    mscale = 1.0
    if scaling.type == "linear" and s != 1.0:
        inv = inv / s
    elif scaling.type == "yarn" and s != 1.0:
        original = scaling.original_max_seq_len or max(1, int(max_seq_len / s))
        wavelength = 2 * math.pi / inv
        ratio = original / wavelength
        ramp = ((ratio - scaling.beta_slow) / (scaling.beta_fast - scaling.beta_slow)).clamp(0, 1)
        inv = (1 - ramp) * (inv / s) + ramp * inv
        mscale = 0.1 * math.log(s) + 1.0
    return inv.float(), mscale


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables; ``forward(position_ids)`` gathers rows per token."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self, head_dim: int, max_seq_len: int, base: float, scaling: RopeScalingConfig
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.reset(max_seq_len, base, scaling)

    def reset(self, max_seq_len: int, base: float, scaling: RopeScalingConfig) -> None:
        """Rebuild tables, e.g. for context extension (keeps the module's device)."""
        device = self.cos.device if hasattr(self, "cos") else None
        inv, mscale = inv_frequencies(self.head_dim, base, scaling, max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", (freqs.cos() * mscale).to(device), persistent=False)
        self.register_buffer("sin", (freqs.sin() * mscale).to(device), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids (B, T) -> cos, sin of shape (B, 1, T, head_dim/2).

        Bounds are checked by the caller on Python ints (no device sync here)."""
        return self.cos[position_ids].unsqueeze(1), self.sin[position_ids].unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x1, x2) = (first half, second half) of the head dimension."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return out.to(x.dtype)
