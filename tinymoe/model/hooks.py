"""Instrumentation hooks: capture internals on probe inputs for the viewer.

A :class:`Capture` is passed explicitly through the forward pass (no global state, no
``register_forward_hook``), so capturing is opt-in per call and cannot leak into training
steps. Captured tensors are detached and moved to CPU immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Capture:
    """What the model recorded during one forward pass."""

    attention: bool = True
    residual: list[torch.Tensor] = field(default_factory=list)  # embed + after each layer
    attn_probs: dict[int, torch.Tensor] = field(default_factory=dict)  # (B, H, T, T)
    attn_out_norm: dict[int, torch.Tensor] = field(default_factory=dict)  # (B, T)
    ffn_out_norm: dict[int, torch.Tensor] = field(default_factory=dict)  # (B, T)
    routing: dict[int, torch.Tensor] = field(default_factory=dict)  # (B, T, k)
    router_scores: dict[int, torch.Tensor] = field(default_factory=dict)  # (B, T, E)
    expert_out_norm: dict[int, torch.Tensor] = field(default_factory=dict)  # (E,)

    @staticmethod
    def keep(t: torch.Tensor) -> torch.Tensor:
        return t.detach().float().cpu()
