"""Fine-grained Mixture-of-Experts with shared experts (DeepSeekMoE / DeepSeek-V3 / GLM-4.5).

Routing
    ``scores = sigmoid(x W_r)`` (V3) or ``softmax`` (V2/Mixtral). Top-k selection uses
    ``scores + bias`` but the gate weights use the *unbiased* scores, re-normalised over
    the selected experts and multiplied by ``routed_scaling``.

Load balancing (``moe.balance``)
    * ``aux_free``: DeepSeek-V3 auxiliary-loss-free balancing. After every optimizer step
      ``bias_i += rate * sign(mean_load - load_i)``; no gradient interference with the LM
      loss. A tiny *sequence-wise* balance loss (``seq_aux_weight``) guards extreme
      imbalance within one sequence, as in V3.
    * ``switch``: classic Switch-Transformer loss ``E * sum_i f_i * P_i`` (Fedus et al.).
    * ``none``: no balancing (watch the router collapse - a useful experiment).
    Router z-loss (ST-MoE, Zoph et al., 2022) is available on top of any of these.

Dispatch
    Token->expert assignments are sorted by expert id. ``loop`` runs one grouped matmul per
    expert on its contiguous slice; ``padded`` scatters into an ``(E, C, d)`` tensor and
    runs three batched matmuls (no Python loop; best when load is balanced). The padded width
    is capped at ``padded_limit`` x the balanced load, so a collapsed router cannot blow up
    memory (an early-training spike once OOMed the Mac preset, docs/decisions.md D12); the
    assignments past the cap - contiguous in sorted order - run through the per-expert path
    for just the overloaded experts, so results stay exact. ``worst_case`` (set while
    autobatch probes memory) pins the width at that cap: a fresh router is near-balanced,
    and training must not later outgrow the micro batch the probe chose. ``auto`` picks
    ``padded`` when padding waste is small. Results return via ``index_add_``. An optional
    capacity factor drops overflow tokens (GShard); the default never drops.

Routing replay
    ``routing_override`` forces the expert choice (e.g. replaying what the inference engine
    chose during an RL rollout) while gate weights still come from the current router, so
    the router keeps receiving gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tinymoe.config import ModelConfig
from tinymoe.model.hooks import Capture
from tinymoe.model.mlp import SwiGLU


@dataclass
class MoEAux:
    """Auxiliary losses (already weighted) and the routing used by this forward."""

    loss: torch.Tensor
    seq_aux: torch.Tensor
    switch_aux: torch.Tensor
    z_loss: torch.Tensor
    topk: torch.Tensor  # (B, T, k)


class MoE(nn.Module):
    expert_bias: torch.Tensor
    load_counts: torch.Tensor

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        m = cfg.moe
        self.layer_idx = layer_idx
        self.n_experts = m.n_experts
        self.top_k = m.top_k
        self.cfg = m
        self.worst_case = False  # memory probing: padded buffers at their cap
        d, h = cfg.d_model, m.expert_hidden
        self.router = nn.Linear(d, m.n_experts, bias=False)
        self.w_gate = nn.Parameter(torch.empty(m.n_experts, d, h))
        self.w_up = nn.Parameter(torch.empty(m.n_experts, d, h))
        self.w_down = nn.Parameter(torch.empty(m.n_experts, h, d))
        self.shared = SwiGLU(d, h * m.n_shared_experts) if m.n_shared_experts > 0 else None
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, 0.0, cfg.init_std)  # Transformer.reset_parameters refines this
        self.register_buffer("expert_bias", torch.zeros(m.n_experts))
        self.register_buffer("load_counts", torch.zeros(m.n_experts), persistent=False)

    # ----------------------------------------------------------------- routing
    def route(
        self, h: torch.Tensor, override: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (N, d) -> logits (N, E) fp32, scores (N, E), topk idx (N, k), gates (N, k)."""
        logits = F.linear(h.float(), self.router.weight.float())
        scores = logits.sigmoid() if self.cfg.score_func == "sigmoid" else logits.softmax(-1)
        if override is not None:
            topk = override
        else:
            select = scores + self.expert_bias if self.cfg.balance == "aux_free" else scores
            topk = select.topk(self.top_k, dim=-1).indices
        gates = scores.gather(-1, topk)
        if self.cfg.score_func == "sigmoid" or self.cfg.norm_topk_prob:
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)
        return logits, scores, topk, gates * self.cfg.routed_scaling

    # ---------------------------------------------------------------- experts
    def _expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]

    def _dispatch(self, h: torch.Tensor, topk: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        n, k = topk.shape
        flat_e = topk.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[order]
        token = order // k
        counts = torch.bincount(flat_e, minlength=self.n_experts)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(n * k, device=h.device) - starts[sorted_e]
        weight = gates.reshape(-1)[order].unsqueeze(-1)
        full_counts = counts.tolist()  # one host sync per layer
        counts_list = full_counts
        keep: torch.Tensor | None = None
        if self.cfg.capacity_factor is not None:
            cap = math.ceil(self.cfg.capacity_factor * n * k / self.n_experts)
            keep = rank < cap
            counts_list = [min(c, cap) for c in full_counts]
        max_count = max(counts_list)
        mode = self.cfg.dispatch
        if mode == "auto":
            balanced = max_count * self.n_experts <= 2 * n * k
            mode = "padded" if balanced or self.worst_case else "loop"
        xs = h[token]
        if mode == "padded":
            limit = math.ceil(self.cfg.padded_limit * n * k / self.n_experts)
            width = max(limit if self.worst_case else min(max_count, limit), 1)
            # width <= max_count <= cap, so rank < width already implies keep
            sel = None if width >= max_count and keep is None else rank < width
            pad = h.new_zeros(self.n_experts, width, h.shape[-1])
            if sel is None:
                pad[sorted_e, rank] = xs
            else:
                pad[sorted_e[sel], rank[sel]] = xs[sel]
            act = F.silu(torch.bmm(pad, self.w_gate)) * torch.bmm(pad, self.w_up)
            out_pad = torch.bmm(act, self.w_down)
            if sel is None:
                ys = out_pad[sorted_e, rank]
            else:
                ys_parts = [out_pad[sorted_e[sel], rank[sel]]]
                tok_parts, w_parts = [token[sel]], [weight[sel]]
                start = 0
                for e, (c_full, c) in enumerate(zip(full_counts, counts_list, strict=True)):
                    if c > width:  # overloaded expert: its tail runs unpadded
                        tail = slice(start + width, start + c)
                        ys_parts.append(self._expert(xs[tail], e))
                        tok_parts.append(token[tail])
                        w_parts.append(weight[tail])
                    start += c_full
                ys, token, weight = torch.cat(ys_parts), torch.cat(tok_parts), torch.cat(w_parts)
        else:
            if keep is not None:
                xs, token, weight, sorted_e = xs[keep], token[keep], weight[keep], sorted_e[keep]
            chunks = []
            offset = 0
            for e, c in enumerate(counts_list):
                if c:
                    chunks.append(self._expert(xs[offset : offset + c], e))
                offset += c
            ys = torch.cat(chunks) if chunks else xs[:0]
        ys = ys * weight.to(ys.dtype)
        return torch.zeros(n, h.shape[-1], dtype=ys.dtype, device=h.device).index_add_(0, token, ys)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        override: torch.Tensor | None = None,
        capture: Capture | None = None,
    ) -> tuple[torch.Tensor, MoEAux]:
        b, t, d = x.shape
        h = x.reshape(-1, d)
        flat_override = override.reshape(-1, self.top_k) if override is not None else None
        logits, scores, topk, gates = self.route(h, flat_override)
        y = self._dispatch(h, topk, gates)
        if self.shared is not None:
            y = y + self.shared(h).to(y.dtype)
        if self.training:
            self.load_counts += torch.bincount(topk.reshape(-1), minlength=self.n_experts).float()
        aux = self._aux(logits, scores, topk, b, t)
        if capture is not None:
            capture.routing[self.layer_idx] = topk.view(b, t, -1).detach().cpu()
            capture.router_scores[self.layer_idx] = Capture.keep(scores.view(b, t, -1))
            capture.expert_out_norm[self.layer_idx] = Capture.keep(self._expert_norms(h, topk))
        return y.view(b, t, d).to(x.dtype), aux

    def _aux(
        self, logits: torch.Tensor, scores: torch.Tensor, topk: torch.Tensor, b: int, t: int
    ) -> MoEAux:
        zero = logits.new_zeros(())
        e, k = self.n_experts, self.top_k
        seq_aux = switch_aux = z_loss = zero
        use_seq = self.cfg.balance == "aux_free" and self.cfg.seq_aux_weight > 0
        if use_seq or self.cfg.balance == "switch":
            # selection indicators carry no gradient; P (normalised scores) does
            onehot = torch.zeros_like(scores).scatter_(-1, topk, 1.0)  # (N, E)
            probs = scores / scores.sum(-1, keepdim=True).clamp_min(1e-9)
            if use_seq:
                f = onehot.view(b, t, e).sum(1) * (e / (k * t))  # (B, E)
                p = probs.view(b, t, e).mean(1)
                seq_aux = self.cfg.seq_aux_weight * (f * p).sum(-1).mean()
            else:
                f = onehot.mean(0) / k
                switch_aux = self.cfg.switch_aux_weight * e * (f * probs.mean(0)).sum()
        if self.cfg.z_loss_weight > 0:
            z_loss = self.cfg.z_loss_weight * torch.logsumexp(logits, -1).pow(2).mean()
        return MoEAux(
            seq_aux + switch_aux + z_loss, seq_aux, switch_aux, z_loss, topk.view(b, t, k)
        )

    @torch.no_grad()
    def _expert_norms(self, h: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        """Mean output norm per expert over the tokens routed to it (0 if unused)."""
        norms = torch.zeros(self.n_experts)
        for e in sorted(set(topk.flatten().tolist())):
            rows = (topk == e).any(-1)
            norms[e] = self._expert(h[rows], e).float().norm(dim=-1).mean().cpu()
        return norms

    # ------------------------------------------------------------ balancing
    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor) -> None:
        """DeepSeek-V3 aux-free update: push overloaded experts' bias down, idle ones up."""
        if self.cfg.balance != "aux_free":
            return
        load = counts / counts.sum().clamp_min(1)  # all-zero counts -> sign(0) -> no change
        self.expert_bias += self.cfg.bias_update_rate * torch.sign(load.mean() - load)

    def reset_counts(self) -> torch.Tensor:
        counts = self.load_counts.clone()
        self.load_counts.zero_()
        return counts
