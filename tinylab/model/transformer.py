"""Decoder-only transformer: pre-norm blocks, dense or MoE FFNs, MTP heads, generation.

Architecture follows the modern open-weight recipe (Llama-3 / Qwen3 / DeepSeek-V3):
pre-norm RMSNorm, RoPE, GQA + QK-Norm, SwiGLU, no biases, optional tied embeddings, the
first ``moe.n_dense_layers`` layers dense and the rest fine-grained MoE.

Multi-token prediction (DeepSeek-V3 §2.2): module ``k`` combines the previous depth's
hidden state at position ``i`` with the embedding of token ``i+k`` and predicts token
``i+k+1`` through one extra transformer block and the shared output head. It densifies the
training signal and doubles as a draft model for self-speculative decoding
(:func:`generate_speculative`).

Initialisation: N(0, init_std) everywhere; residual-output projections (attention
``o_proj``, FFN/expert ``down``) are scaled by ``1/sqrt(2 * n_layers)`` (GPT-2 /
Megatron) so the residual stream variance does not grow with depth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tinylab.config import ModelConfig, RopeScalingConfig, head_dim
from tinylab.model.attention import Attention, AttnMask, KVCache, build_mask
from tinylab.model.hooks import Capture
from tinylab.model.mlp import SwiGLU
from tinylab.model.moe import MoE, MoEAux
from tinylab.model.norms import RMSNorm
from tinylab.model.rope import RotaryEmbedding


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor
    aux: dict[str, torch.Tensor]
    mtp_logits: list[torch.Tensor] = field(default_factory=list)
    hidden: torch.Tensor | None = None  # last block output (pre final norm)
    routing: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> (B, T, k)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, use_moe: bool) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn: SwiGLU | MoE = (
            MoE(cfg, layer_idx) if use_moe else SwiGLU(cfg.d_model, cfg.ffn_hidden)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: AttnMask,
        cache: KVCache | None,
        override: torch.Tensor | None,
        capture: Capture | None,
    ) -> tuple[torch.Tensor, MoEAux | None]:
        a = self.attn(self.attn_norm(x), cos, sin, mask, cache, capture)
        x = x + a
        aux = None
        if isinstance(self.ffn, MoE):
            f, aux = self.ffn(self.ffn_norm(x), override, capture)
        else:
            f = self.ffn(self.ffn_norm(x))
        if capture is not None:
            capture.attn_out_norm[self.layer_idx] = Capture.keep(a.norm(dim=-1))
            capture.ffn_out_norm[self.layer_idx] = Capture.keep(f.norm(dim=-1))
        return x + f, aux


class MTPModule(nn.Module):
    """One DeepSeek-V3 MTP depth: RMSNorm(h) ++ RMSNorm(emb) -> proj -> block -> norm."""

    def __init__(self, cfg: ModelConfig, depth: int) -> None:
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, depth - 1, use_moe=False)  # layer_idx = its cache slot
        self.block.attn.window = None
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: AttnMask,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb)], dim=-1))
        out: torch.Tensor = self.block(x, cos, sin, mask, cache, None, None)[0]
        return out


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.vocab_size < 1:
            raise ValueError("model.vocab_size must be set (taken from the tokenizer)")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        n_dense = cfg.moe.n_dense_layers if cfg.moe.enabled else cfg.n_layers
        self.layers = nn.ModuleList(
            Block(cfg, i, use_moe=i >= n_dense) for i in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = (
            None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        )
        self.mtp = nn.ModuleList(MTPModule(cfg, k + 1) for k in range(cfg.mtp.depth))
        self.rope = RotaryEmbedding(head_dim(cfg), cfg.max_seq_len, cfg.rope_base, cfg.rope_scaling)
        self.has_local = any(b.attn.window is not None for b in self.blocks)
        self.reset_parameters()

    # ------------------------------------------------------------------ setup
    @property
    def blocks(self) -> list[Block]:
        return [b for b in self.layers if isinstance(b, Block)]

    def moe_layers(self) -> list[MoE]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.ndim < 2:
                continue  # norm weights stay at 1
            residual_out = name.endswith(("o_proj.weight", "down_proj.weight", "w_down"))
            nn.init.normal_(p, 0.0, out_std if residual_out else std)
        for moe in self.moe_layers():
            nn.init.normal_(moe.router.weight, 0.0, self.cfg.moe.router_init_std or std)

    def extend_context(self, max_seq_len: int, scaling: RopeScalingConfig) -> None:
        """Context extension: rebuild RoPE tables for a longer window with ``scaling``."""
        if scaling.original_max_seq_len is None:
            scaling = replace(scaling, original_max_seq_len=self.cfg.max_seq_len)
        self.cfg.max_seq_len = max_seq_len
        self.cfg.rope_scaling = scaling
        self.rope.reset(max_seq_len, self.cfg.rope_base, scaling)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            return F.linear(h, self.embed.weight)
        out: torch.Tensor = self.lm_head(h)
        return out

    def new_cache(self, batch: int, max_len: int, n_layers: int | None = None) -> KVCache:
        p = self.embed.weight
        return KVCache(
            n_layers or self.cfg.n_layers, batch, self.cfg.n_kv_heads, max_len, head_dim(self.cfg),
            p.device, torch.float32,
        )  # fmt: skip

    # ---------------------------------------------------------------- forward
    def _masks(
        self,
        b: int,
        t: int,
        device: torch.device,
        doc_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        cache: KVCache | None,
    ) -> AttnMask:
        need_full = cache is not None or doc_ids is not None or attention_mask is not None
        if not need_full and not self.has_local:
            return AttnMask(None, None)
        offset = cache.pos if cache is not None else 0
        q_slots = (offset + torch.arange(t, device=device)).expand(b, t)
        key_valid = attention_mask
        if cache is not None:
            if doc_ids is not None:
                raise ValueError("doc_ids are not supported with a KV cache")
            new = (
                attention_mask
                if attention_mask is not None
                else torch.ones(b, t, dtype=torch.bool, device=device)
            )
            cache.valid[:, offset : offset + t] = new
            key_valid = cache.valid
        k_len = offset + t
        window = self.cfg.sliding_window

        def make(w: int | None) -> torch.Tensor:
            return build_mask(q_slots, k_len, key_valid, doc_ids, doc_ids, w)

        return AttnMask(make(None) if need_full else None, make(window) if self.has_local else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        routing_override: dict[int, torch.Tensor] | None = None,
        capture: Capture | None = None,
        compute_mtp: bool | None = None,
    ) -> ModelOutput:
        b, t = input_ids.shape
        offset = cache.pos if cache is not None else 0
        if offset + t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {offset + t} > max_seq_len {self.cfg.max_seq_len}")
        if position_ids is None:
            position_ids = (offset + torch.arange(t, device=input_ids.device)).expand(b, t)
        cos, sin = self.rope(position_ids)
        mask = self._masks(b, t, input_ids.device, doc_ids, attention_mask, cache)
        x = self.embed(input_ids)
        if capture is not None:
            capture.residual.append(Capture.keep(x))
        zero = x.new_zeros((), dtype=torch.float32)
        aux_terms = {"seq_aux": zero, "switch_aux": zero, "z_loss": zero}
        aux_loss = zero
        routing: dict[int, torch.Tensor] = {}
        for block in self.blocks:
            override = routing_override.get(block.layer_idx) if routing_override else None
            x, aux = block(x, cos, sin, mask, cache, override, capture)
            if aux is not None:
                aux_loss = aux_loss + aux.loss
                aux_terms = {
                    "seq_aux": aux_terms["seq_aux"] + aux.seq_aux.detach(),
                    "switch_aux": aux_terms["switch_aux"] + aux.switch_aux.detach(),
                    "z_loss": aux_terms["z_loss"] + aux.z_loss.detach(),
                }
                routing[block.layer_idx] = aux.topk
            if capture is not None:
                capture.residual.append(Capture.keep(x))
        if cache is not None:
            cache.advance(t)
        logits = self.head(self.norm(x))
        out = ModelOutput(logits, aux_loss, aux_terms, hidden=x, routing=routing)
        if compute_mtp is None:
            compute_mtp = self.training
        if compute_mtp and len(self.mtp) and cache is None:
            out.mtp_logits = self._mtp_forward(x, input_ids, cos, sin, mask)
        return out

    def _mtp_forward(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: AttnMask,
    ) -> list[torch.Tensor]:
        """Depth-k logits aligned with ``labels[:, k:]`` (they predict token i+k+1)."""
        outs = []
        t = input_ids.shape[1]
        for k, module in enumerate(self.mtp, start=1):
            assert isinstance(module, MTPModule)
            if t - k < 1:
                break
            n = t - k
            sub = AttnMask(
                mask.full[..., :n, :n] if mask.full is not None else None,
                None,
            )
            h = module(h[:, :n], self.embed(input_ids[:, k:]), cos[:, :, :n], sin[:, :, :n], sub)
            outs.append(self.head(module.out_norm(h)))
        return outs

    # ------------------------------------------------------------ balancing
    def load_counts(self) -> torch.Tensor | None:
        """(n_moe_layers, E) routed-token counts since the last reset (then reset)."""
        layers = self.moe_layers()
        if not layers:
            return None
        return torch.stack([m.reset_counts() for m in layers])

    def update_router_bias(self, counts: torch.Tensor) -> None:
        for moe, c in zip(self.moe_layers(), counts, strict=True):
            moe.update_bias(c)


# ------------------------------------------------------------------ generation


@dataclass
class Generation:
    sequences: list[list[int]]  # completion tokens (stop token included if hit)
    logprobs: list[list[float]]  # per completion token, under the tempered distribution
    finished: list[bool]  # True if a stop token ended the completion
    routing: list[dict[int, np.ndarray]] | None = None  # per sample: layer -> (L, k)


def sample_next(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row on CPU (deterministic across devices). Returns ids, logprobs."""
    logits = logits.float().cpu()
    if temperature <= 0:
        ids = logits.argmax(-1)
        return ids, logits.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
    logp = (logits / temperature).log_softmax(-1)
    probs = logp.exp()
    if top_p < 1.0:
        sorted_p, idx = probs.sort(-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
    ids = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return ids, logp.gather(-1, ids[:, None]).squeeze(-1)


@torch.no_grad()
def generate(
    model: Transformer,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_ids: Sequence[int] = (),
    pad_id: int = 0,
    generator: torch.Generator | None = None,
    record_routing: bool = False,
) -> Generation:
    """Batched sampling with a KV cache. Prompts are left-padded; RoPE positions and key
    masks make padding invisible, so each row decodes exactly as if alone."""
    was_training = model.training
    model.eval()
    device = model.embed.weight.device
    b = len(prompts)
    plen = max(len(p) for p in prompts)
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - plen)
    ids = torch.full((b, plen), pad_id, dtype=torch.long)
    valid = torch.zeros(b, plen, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, plen - len(p) :] = torch.tensor(list(p), dtype=torch.long)
        valid[i, plen - len(p) :] = True
    positions = (valid.long().cumsum(1) - 1).clamp_min(0)
    ids, valid, positions = ids.to(device), valid.to(device), positions.to(device)
    cache = model.new_cache(b, plen + max_new_tokens)
    out = model(ids, position_ids=positions, attention_mask=valid, cache=cache)
    routes: list[dict[int, torch.Tensor]] = [out.routing] if record_routing else []
    next_pos = positions[:, -1] + 1
    logits = out.logits[:, -1]
    stop = set(stop_ids)
    seqs: list[list[int]] = [[] for _ in range(b)]
    lps: list[list[float]] = [[] for _ in range(b)]
    done = [False] * b
    for step in range(max_new_tokens):
        tok, lp = sample_next(logits, temperature, top_p, generator)
        for i in range(b):
            if not done[i]:
                seqs[i].append(int(tok[i]))
                lps[i].append(float(lp[i]))
                done[i] = int(tok[i]) in stop
        if all(done) or step == max_new_tokens - 1:
            break
        out = model(tok.to(device)[:, None], position_ids=next_pos[:, None], cache=cache)
        if record_routing:
            routes.append(out.routing)
        next_pos = next_pos + 1
        logits = out.logits[:, -1]
    model.train(was_training)
    routing = _collect_routing(routes, prompts, plen, seqs) if record_routing else None
    return Generation(seqs, lps, done, routing)


def _collect_routing(
    routes: list[dict[int, torch.Tensor]],
    prompts: Sequence[Sequence[int]],
    plen: int,
    seqs: list[list[int]],
) -> list[dict[int, np.ndarray]]:
    """Per sample, routing of every token that was *fed* (prompt + completion[:-1])."""
    layers = routes[0].keys()
    full = {layer: torch.cat([r[layer] for r in routes], dim=1).cpu().numpy() for layer in layers}
    out = []
    for i, prompt in enumerate(prompts):
        start = plen - len(prompt)
        end = plen + max(len(seqs[i]) - 1, 0)
        out.append({layer: full[layer][i, start:end] for layer in layers})
    return out


@torch.no_grad()
def generate_speculative(
    model: Transformer, prompt: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = ()
) -> tuple[list[int], dict[str, int]]:
    """Greedy self-speculative decoding with the first MTP module as the drafter.

    Each step verifies ``[pending, draft]`` with one 2-token forward of the main model. On
    acceptance two tokens are emitted; on rejection the cache is rolled back one slot. The
    output is identical to plain greedy decoding (tested); the acceptance rate is reported.
    """
    if not len(model.mtp):
        raise ValueError("speculative decoding needs model.mtp.depth >= 1")
    was_training = model.training
    model.eval()
    device = model.embed.weight.device
    mtp = model.mtp[0]
    assert isinstance(mtp, MTPModule)
    n = len(prompt)
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - n - 1)
    cache = model.new_cache(1, n + max_new_tokens + 2)
    mcache = model.new_cache(1, n + max_new_tokens + 2, n_layers=1)
    stop = set(stop_ids)

    def mtp_draft(h: torch.Tensor, next_tokens: list[int]) -> int:
        """Feed MTP slots (h_j, emb(x_{j+1})) and return the draft for the last slot."""
        t = len(next_tokens)
        pos = (mcache.pos + torch.arange(t, device=device))[None]
        cos, sin = model.rope(pos)
        mask = model._masks(1, t, device, None, None, mcache)
        emb = model.embed(torch.tensor([next_tokens], device=device))
        hm = mtp(h, emb, cos, sin, mask, mcache)
        mcache.advance(t)
        return int(model.head(mtp.out_norm(hm))[0, -1].argmax())

    out = model(torch.tensor([list(prompt)], device=device), cache=cache)
    assert out.hidden is not None
    pending = int(out.logits[0, -1].argmax())
    emitted = [pending]
    draft = mtp_draft(out.hidden, [*prompt[1:], pending])
    stats = {"drafted": 0, "accepted": 0}
    while len(emitted) < max_new_tokens and emitted[-1] not in stop:
        base = cache.pos
        out = model(torch.tensor([[pending, draft]], device=device), cache=cache)
        assert out.hidden is not None
        verify = out.logits[0].argmax(-1).tolist()
        stats["drafted"] += 1
        if verify[0] == draft:
            stats["accepted"] += 1
            emitted += [draft, int(verify[1])][: max_new_tokens - len(emitted)]
            if draft in stop:
                emitted = emitted[: emitted.index(draft) + 1]
                break
            pending = int(verify[1])
            draft = mtp_draft(out.hidden, [draft, pending])
        else:
            cache.truncate(base + 1)
            pending = int(verify[0])
            emitted.append(pending)
            draft = mtp_draft(out.hidden[:, :1], [pending])
    model.train(was_training)
    return emitted[:max_new_tokens], stats
