"""Optimizers: Muon for hidden matrices + AdamW for everything else, behind one facade.

Muon (Jordan et al., 2024) orthogonalises the momentum of each 2-D weight with a quintic
Newton-Schulz iteration, i.e. it steps along ``U V^T`` of the update's SVD. Moonshot's
"Muon is Scalable" (Liu et al., 2025; Kimi K2) adds decoupled weight decay and rescales
the update by ``0.2 * sqrt(max(m, n))`` so its RMS matches AdamW's - the same learning
rate then works for both. GLM-4.5 and Kimi K2 train with Muon. Embeddings, the output
head, norms and the MoE router stay on AdamW (Muon is only defined for hidden matrices).
3-D expert tensors ``(E, m, n)`` are orthogonalised per expert in one batched iteration.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from tinylab.config import MuonConfig, TrainConfig
from tinylab.hardware import Runtime

log = logging.getLogger(__name__)
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    """Approximate ``U V^T`` for every trailing (m, n) matrix of ``g``."""
    a, b, c = NS_COEFFS
    x = g.to(dtype)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    out: torch.Tensor = x.to(g.dtype)
    return out


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter] | list[dict[str, Any]],
        lr: float,
        weight_decay: float,
        cfg: MuonConfig,
        ns_dtype: torch.dtype,
    ) -> None:
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": cfg.momentum}
        super().__init__(params, defaults)
        self.cfg = cfg
        self.ns_dtype = ns_dtype

    def _scale(self, shape: torch.Size) -> float:
        m, n = shape[-2], shape[-1]
        if self.cfg.scale == "moonlight":
            return 0.2 * math.sqrt(max(m, n))
        return math.sqrt(max(1.0, m / n))

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(mom).add_(p.grad)
                g = p.grad.add(buf, alpha=mom) if self.cfg.nesterov else buf
                update = newton_schulz(g, self.cfg.ns_steps, self.ns_dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * self._scale(p.shape))


def classify(name: str, p: nn.Parameter) -> str:
    """'muon' (hidden matrices), 'decay' (other matrices), 'no_decay' (vectors, router, LoRA
    adapters: low-rank factors are neither "hidden matrices" for Muon nor decayed)."""
    if p.ndim < 2 or "lora_" in name:
        return "no_decay"
    if name.endswith("embed.weight") or "router" in name:
        return "embed" if name.endswith("embed.weight") else "no_decay"
    if name.endswith("lm_head.weight"):
        return "decay"
    return "muon"


class Optimizers:
    """Facade over one or two torch optimizers with per-group base learning rates."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        for opt in optimizers:
            for g in opt.param_groups:
                g.setdefault("base_lr", g["lr"])

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    def set_lr_multiplier(self, mult: float) -> None:
        for g in self.param_groups:
            g["lr"] = g["base_lr"] * mult

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def state_dict(self) -> list[dict[str, Any]]:
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, states: list[dict[str, Any]]) -> None:
        for opt, st in zip(self.optimizers, states, strict=True):
            opt.load_state_dict(st)


def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
    runtime: Runtime,
    lr: float | None = None,
    quiet: bool = False,
) -> Optimizers:
    """Param groups by role; Muon or AdamW for hidden matrices per ``cfg.optimizer``."""
    lr = cfg.lr if lr is None else lr
    buckets: dict[str, list[nn.Parameter]] = {"muon": [], "decay": [], "no_decay": [], "embed": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[classify(name, p)].append(p)
    use_muon = cfg.optimizer == "muon"
    adam_groups = [
        {"params": buckets["decay"], "weight_decay": cfg.weight_decay, "lr": lr},
        {"params": buckets["no_decay"], "weight_decay": 0.0, "lr": lr},
        {
            "params": buckets["embed"],
            "weight_decay": cfg.weight_decay if cfg.decay_embeddings else 0.0,
            "lr": lr,
        },
    ]
    if not use_muon:
        adam_groups.append({"params": buckets["muon"], "weight_decay": cfg.weight_decay, "lr": lr})
    adam_groups = [g for g in adam_groups if g["params"]]
    kwargs: dict[str, Any] = {"betas": tuple(cfg.betas), "eps": cfg.eps}
    if runtime.fused_optimizer:
        kwargs["fused"] = True
    else:
        kwargs["foreach"] = True
    opts: list[torch.optim.Optimizer] = []
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, lr=lr, **kwargs))
    if use_muon and buckets["muon"]:
        ns_dtype = _ns_dtype(cfg.muon, runtime)
        muon_lr = cfg.muon.lr if cfg.muon.lr is not None else lr
        opts.append(Muon(buckets["muon"], muon_lr, cfg.weight_decay, cfg.muon, ns_dtype))
    counts = {k: sum(p.numel() for p in v) for k, v in buckets.items()}
    if not quiet:
        log.info("[optim] %s; params per group %s", "muon+adamw" if use_muon else "adamw", counts)
    return Optimizers(opts)


def _ns_dtype(cfg: MuonConfig, runtime: Runtime) -> torch.dtype:
    if cfg.ns_dtype == "fp32":
        return torch.float32
    if cfg.ns_dtype == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if runtime.caps.native_bf16 else torch.float32
