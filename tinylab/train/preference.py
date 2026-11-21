"""Preference-optimisation losses on summed sequence log-probabilities.

With ``h = (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))``:

* **DPO** (Rafailov et al., 2023): ``-log sigmoid(beta * h)``; ``label_smoothing`` eps gives
  conservative DPO (``-(1-eps) log s(beta h) - eps log s(-beta h)``) for noisy labels.
* **IPO** (Azar et al., 2023): ``(h - 1 / (2 beta))^2`` - a squared loss that keeps the
  margin finite (DPO can push ``h`` to infinity on deterministic preferences).
* **SimPO** (Meng et al., 2024): reference-free, length-normalised
  ``-log sigmoid(beta * (logp_w / |y_w| - logp_l / |y_l|) - gamma)``.

The implicit rewards ``beta * (log pi - log ref)`` are reported as in the DPO paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tinylab.config import DPOConfig


@dataclass
class PrefLoss:
    loss: torch.Tensor  # scalar
    per_pair: torch.Tensor  # (B,)
    chosen_reward: torch.Tensor  # (B,)
    rejected_reward: torch.Tensor  # (B,)


def preference_loss(
    cfg: DPOConfig,
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    len_chosen: torch.Tensor,
    len_rejected: torch.Tensor,
) -> PrefLoss:
    """All inputs are (B,) tensors of summed log-probs / token counts."""
    if cfg.loss == "simpo":
        pc = policy_chosen / len_chosen.clamp_min(1)
        pr = policy_rejected / len_rejected.clamp_min(1)
        logits = cfg.beta * (pc - pr) - cfg.simpo_gamma
        per = -F.logsigmoid(logits)
        return PrefLoss(per.mean(), per, cfg.beta * pc.detach(), cfg.beta * pr.detach())
    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    h = chosen_ratio - rejected_ratio
    if cfg.loss == "ipo":
        per = (h - 1.0 / (2.0 * cfg.beta)) ** 2
    else:
        eps = cfg.label_smoothing
        per = -(1 - eps) * F.logsigmoid(cfg.beta * h) - eps * F.logsigmoid(-cfg.beta * h)
    return PrefLoss(
        per.mean(), per, cfg.beta * chosen_ratio.detach(), cfg.beta * rejected_ratio.detach()
    )
