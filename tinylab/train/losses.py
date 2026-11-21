"""Shared losses: next-token CE (+ MoE aux, MTP, z-loss), per-source breakdown, log-probs,
and the token-level KL used by distillation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from tinylab.data.batch import IGNORE_INDEX, Batch
from tinylab.model.transformer import ModelOutput

COUNT_PREFIX = "_count/"


@dataclass
class LossOutput:
    """``loss`` is backpropagated; ``metrics`` are detached scalars averaged over micro-steps.

    A metric ``m`` with a companion ``_count/m`` is a sum: the trainer reports
    ``sum(m) / sum(_count/m)`` over the step instead of a mean of per-micro values.
    """

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def forward_batch(model: nn.Module, batch: Batch, **kwargs: object) -> ModelOutput:
    out: ModelOutput = model(
        batch.input_ids,
        position_ids=batch.position_ids,
        doc_ids=batch.doc_ids,
        attention_mask=batch.attention_mask,
        **kwargs,
    )
    return out


def token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token cross entropy in fp32, 0 where labels are ignored. Shape = labels.shape."""
    flat = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    return flat.view(labels.shape)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def lm_loss(
    model: nn.Module,
    batch: Batch,
    mtp_weight: float = 0.0,
    z_loss_weight: float = 0.0,
    source_names: Sequence[str] | None = None,
    aux_scale: float = 1.0,
) -> LossOutput:
    """Next-token CE + MoE balance losses + DeepSeek-V3 MTP loss + PaLM z-loss."""
    out = forward_batch(model, batch)
    mask = (batch.labels != IGNORE_INDEX).float()
    ce_tok = token_ce(out.logits, batch.labels)
    ce = masked_mean(ce_tok, mask)
    total = ce + aux_scale * out.aux_loss
    metrics: dict[str, torch.Tensor] = {"loss": ce.detach(), "aux_loss": out.aux_loss.detach()}
    if out.mtp_logits and mtp_weight > 0:
        mtp_losses = []
        for k, lg in enumerate(out.mtp_logits, start=1):
            labels_k = batch.labels[:, k:]
            mtp_losses.append(
                masked_mean(token_ce(lg, labels_k), (labels_k != IGNORE_INDEX).float())
            )
        mtp = torch.stack(mtp_losses).mean()
        total = total + mtp_weight * mtp
        metrics["mtp_loss"] = mtp.detach()
    if z_loss_weight > 0:
        z = masked_mean(torch.logsumexp(out.logits.float(), -1).pow(2), mask)
        total = total + z_loss_weight * z
        metrics["z_loss"] = z.detach()
    if source_names is not None and batch.source_ids is not None:
        row_ce = (ce_tok * mask).sum(1) / mask.sum(1).clamp_min(1)
        for i, name in enumerate(source_names):
            sel = (batch.source_ids == i).float()
            # (sum, count) pair: the trainer divides totals over all micro-steps
            metrics[f"loss/{name}"] = (row_ce * sel).sum().detach()
            metrics[f"{COUNT_PREFIX}loss/{name}"] = sel.sum()
    return LossOutput(total, metrics)


def sequence_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Per-token log p(label) (0 at ignored positions). Shape = labels.shape."""
    logp = (logits.float() / temperature).log_softmax(-1)
    safe = labels.clamp_min(0)
    out = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return out * (labels != IGNORE_INDEX)


def token_kl(
    p_logits: torch.Tensor, q_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Per-position ``KL(p || q)`` over the full vocabulary at ``temperature``, in fp32.

    ``p = teacher`` is forward KL (mean-seeking word-level KD, Hinton et al., 2015);
    ``p = student`` is reverse KL (mode-seeking, MiniLLM / GKD). Shape = logits.shape[:-1].
    """
    lp = (p_logits.float() / temperature).log_softmax(-1)
    lq = (q_logits.float() / temperature).log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)
