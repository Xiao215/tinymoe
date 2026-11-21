"""Group-relative policy optimisation: GRPO and the variants that fix its biases.

Per prompt, ``G`` responses are sampled and scored; the advantage of response ``i`` is
``A_i = r_i - mean(r)`` (divided by ``std(r)`` when ``std_normalize``). Every token of a
response shares its advantage. The policy loss is PPO's clipped surrogate on token
ratios ``rho = pi / pi_old``:

    l_t = -min(rho_t A, clip(rho_t, 1 - clip_low, 1 + clip_high) A)  (+ kl_coef * KL_t)

Variants (``grpo.variant``; ``custom`` keeps the knobs as configured):

==========  =============  ===============  ==============  =======  ==========
variant     std normalise  aggregation      clip low/high   KL       extras
==========  =============  ===============  ==============  =======  ==========
grpo        yes            seq_mean         0.2 / 0.2       0.04     -
dr_grpo     no             constant         0.2 / 0.2       0        -
dapo        yes            token_mean       0.2 / 0.28      0        dyn. sampling, overlong
==========  =============  ===============  ==============  =======  ==========

* GRPO (Shao et al., 2024, DeepSeekMath) averages tokens within each sequence, then over
  sequences (``seq_mean``): long wrong answers are penalised *less* per token, a length bias.
* Dr. GRPO (Liu et al., 2025, "Understanding R1-Zero-like training") removes the std
  normalisation (difficulty bias) and divides by a constant ``G * max_new_tokens``.
* DAPO (Yu et al., 2025) uses token-level averaging, a higher upper clip ("clip-higher",
  against entropy collapse), dynamic sampling (drop groups whose rewards are all equal:
  they carry no signal) and the soft overlong penalty (rl/rewards.py).

KL uses the k3 estimator ``exp(q - p) - (q - p) - 1`` with ``q = log ref``, ``p = log pi``
(Schulman, 2020), unbiased and always >= 0.

The aggregation is done *exactly* over a minibatch that spans several micro-batches and
ranks: every micro-batch carries the minibatch-wide denominator and the trainer's
``1 / (accum * world)`` gradient averaging is undone (``scale``), so the gradient equals
that of one big batch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from tinymoe.config import GRPOConfig

VARIANTS: dict[str, dict[str, object]] = {
    "grpo": {
        "std_normalize": True,
        "loss_agg": "seq_mean",
        "clip_low": 0.2,
        "clip_high": 0.2,
        "kl_coef": 0.04,
        "dynamic_sampling": False,
    },
    "dr_grpo": {
        "std_normalize": False,
        "loss_agg": "constant",
        "clip_low": 0.2,
        "clip_high": 0.2,
        "kl_coef": 0.0,
        "dynamic_sampling": False,
    },
    "dapo": {
        "std_normalize": True,
        "loss_agg": "token_mean",
        "clip_low": 0.2,
        "clip_high": 0.28,
        "kl_coef": 0.0,
        "dynamic_sampling": True,
    },
}


def resolve_variant(cfg: GRPOConfig) -> GRPOConfig:
    """Apply the variant preset to the loss knobs (``custom`` leaves them as configured)."""
    if cfg.variant == "custom":
        return cfg
    overlong = replace(cfg.overlong, enabled=cfg.variant == "dapo")
    return replace(cfg, **VARIANTS[cfg.variant], overlong=overlong)  # type: ignore[arg-type]


def group_advantages(rewards: torch.Tensor, std_normalize: bool, eps: float = 1e-6) -> torch.Tensor:
    """(P, G) rewards -> (P, G) advantages."""
    centred = rewards - rewards.mean(-1, keepdim=True)
    if not std_normalize:
        return centred
    return centred / (rewards.std(-1, keepdim=True, unbiased=False) + eps)


def row_weights(lengths: torch.Tensor, loss_agg: str) -> torch.Tensor:
    """Per-row token weight w_i such that loss = sum_i w_i sum_t l_it / denominator."""
    if loss_agg == "seq_mean":
        return 1.0 / lengths.clamp_min(1)
    return torch.ones_like(lengths, dtype=torch.float32)


def denominator(lengths: torch.Tensor, valid: torch.Tensor, loss_agg: str, max_new: int) -> float:
    """Minibatch-wide normaliser over the *valid* rows (dynamic sampling drops rows)."""
    if loss_agg == "token_mean":
        return float((lengths * valid).sum().clamp_min(1))
    if loss_agg == "seq_mean":
        return float(valid.sum().clamp_min(1))
    return float(valid.numel() * max_new)  # Dr. GRPO: constant G * L_max per sequence


@dataclass
class PolicyLoss:
    loss: torch.Tensor
    clip_frac: torch.Tensor
    kl: torch.Tensor
    ratio_mean: torch.Tensor


def policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    row_weight: torch.Tensor,
    denom: float,
    scale: float,
    cfg: GRPOConfig,
) -> PolicyLoss:
    """All per-token tensors are (B, T); advantages / row_weight are (B,).

    ``token_mask`` is 1 on completion tokens that count (0 on prompt, padding, IcePop-
    masked tokens and dropped groups)."""
    adv = advantages[:, None]
    ratio = torch.exp(new_logp - old_logp)
    clipped = ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high)
    per_tok = -torch.minimum(ratio * adv, clipped * adv)
    kl = torch.zeros_like(per_tok)
    if ref_logp is not None and cfg.kl_coef > 0:
        diff = ref_logp - new_logp
        kl = torch.exp(diff) - diff - 1
        per_tok = per_tok + cfg.kl_coef * kl
    w = token_mask * row_weight[:, None]
    loss = (per_tok * w).sum() / denom * scale
    n = token_mask.sum().clamp_min(1)
    is_clipped = ((ratio - clipped).abs() > 1e-6).float()
    return PolicyLoss(
        loss,
        (is_clipped * token_mask).sum().detach() / n,
        (kl.detach() * token_mask).sum() / n,
        (ratio.detach() * token_mask).sum() / n,
    )
