"""Grouped-query attention with QK-Norm, RoPE, SDPA and a preallocated KV cache.

* GQA (Ainslie et al., 2023): ``n_kv_heads < n_heads`` shrinks the KV cache; each KV head
  serves ``n_heads / n_kv_heads`` query heads (Llama-2/3, Qwen, GLM).
* QK-Norm (Dehghani et al., 2023; OLMo-2, Qwen3, Gemma-3): RMSNorm on per-head q and k
  bounds attention logits and removes a classic source of loss spikes.
* ``F.scaled_dot_product_attention`` dispatches to Flash/efficient kernels when the
  backend has them and to the math kernel otherwise (forced via ``hardware.sdpa_backend``).
* Sliding-window layers (Mistral; Gemma-2/3 and Kimi-style local/global interleaving) are
  just a banded mask here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tinylab.config import ModelConfig, head_dim
from tinylab.model.hooks import Capture
from tinylab.model.norms import RMSNorm
from tinylab.model.rope import apply_rope


class KVCache:
    """Per-layer K/V buffers ``(B, n_kv, max_len, head_dim)`` plus key validity.

    ``pos`` is the number of filled slots; layers write at ``[pos, pos+T)`` during a
    forward and the model calls :meth:`advance` once afterwards. ``valid`` marks real
    (non-padding) keys. :meth:`truncate` rolls back (speculative decoding).
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        max_len: int,
        hd: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch, n_kv_heads, max_len, hd)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
        self.max_len = max_len
        self.pos = 0

    def write(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.pos + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer][:, :, self.pos : end] = k.to(self.k[layer].dtype)
        self.v[layer][:, :, self.pos : end] = v.to(self.v[layer].dtype)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int) -> None:
        self.pos += n

    def truncate(self, pos: int) -> None:
        self.valid[:, pos:] = False
        self.pos = pos


@dataclass
class AttnMask:
    """Boolean masks (True = may attend) for global and sliding-window layers.

    ``None`` means "plain causal over the whole sequence" (the SDPA ``is_causal`` path).
    """

    full: torch.Tensor | None
    local: torch.Tensor | None


def build_mask(
    q_slots: torch.Tensor,
    k_len: int,
    key_valid: torch.Tensor | None,
    doc_q: torch.Tensor | None,
    doc_k: torch.Tensor | None,
    window: int | None,
) -> torch.Tensor:
    """(B, 1, Tq, Tk) mask from slot indices; every query may always see itself."""
    k_idx = torch.arange(k_len, device=q_slots.device)
    q = q_slots[:, :, None]  # (B, Tq, 1)
    mask = k_idx[None, None, :] <= q
    if window is not None:
        mask = mask & (k_idx[None, None, :] > q - window)
    if doc_q is not None and doc_k is not None:
        mask = mask & (doc_q[:, :, None] == doc_k[:, None, :])
    if key_valid is not None:
        mask = mask & key_valid[:, None, :k_len]
    mask = mask | (k_idx[None, None, :] == q)  # never an all-False row (NaN-free softmax)
    return mask.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = head_dim(cfg)
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        local = cfg.sliding_window is not None and (layer_idx + 1) % cfg.global_every != 0
        self.window = cfg.sliding_window if local else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: AttnMask,
        cache: KVCache | None = None,
        capture: Capture | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.write(self.layer_idx, k, v)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        m = mask.local if self.window is not None else mask.full
        if m is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        if capture is not None and capture.attention:
            capture.attn_probs[self.layer_idx] = Capture.keep(self._probs(q, k, m))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.hd)
        result: torch.Tensor = self.o_proj(out)
        return result

    @staticmethod
    def _probs(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Explicit softmax(QK^T) for inspection only; the SDPA output is what's used."""
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if mask is None:
            tq, tk = scores.shape[-2:]
            mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(tk - tq)
        return scores.masked_fill(~mask, float("-inf")).softmax(-1)
