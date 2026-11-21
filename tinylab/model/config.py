"""Model sizing: closed-form parameter counts (total vs active per token).

Kept separate from the modules so presets can be sized without building a model, and so
a test can check the formula against ``sum(p.numel())`` of the real module tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinylab.config import ModelConfig, head_dim


@dataclass(frozen=True)
class ParamCount:
    total: int
    active: int
    embedding: int
    routed_experts: int

    @property
    def total_non_embedding(self) -> int:
        return self.total - self.embedding

    @property
    def active_non_embedding(self) -> int:
        return self.active - self.embedding

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "active": self.active,
            "embedding": self.embedding,
            "routed_experts": self.routed_experts,
            "total_non_embedding": self.total_non_embedding,
            "active_non_embedding": self.active_non_embedding,
        }


def param_count(cfg: ModelConfig) -> ParamCount:
    d, hd = cfg.d_model, head_dim(cfg)
    attn = d * hd * cfg.n_heads * 2 + d * hd * cfg.n_kv_heads * 2
    attn += 2 * hd if cfg.qk_norm else 0
    norms = 2 * d
    dense_ffn = 3 * d * cfg.ffn_hidden
    m = cfg.moe
    n_moe = cfg.n_layers - m.n_dense_layers if m.enabled else 0
    expert = 3 * d * m.expert_hidden
    moe_layer = m.n_experts * expert + d * m.n_experts + m.n_shared_experts * expert
    layers = cfg.n_layers * (attn + norms) + (cfg.n_layers - n_moe) * dense_ffn + n_moe * moe_layer
    embedding = cfg.vocab_size * d
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    mtp = cfg.mtp.depth * (3 * d + 2 * d * d + attn + norms + dense_ffn)
    total = embedding + layers + d + head + mtp
    inactive = n_moe * (m.n_experts - m.top_k) * expert
    return ParamCount(total, total - inactive, embedding, n_moe * m.n_experts * expert)
