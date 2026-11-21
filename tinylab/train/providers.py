"""Batch providers: map (optimizer step, micro step) to a deterministic micro-batch.

Rows are addressed globally as ``((step * accum + micro) * world + rank) * mbs + i``, so a
single process with ``accum = A * W`` sees exactly the rows that ``W`` DDP ranks with
``accum = A`` see together - the property the multi-process test checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch

from tinylab.config import BatchPlan


class Movable(Protocol):
    """Any batch type (``Batch``, DPO pair batches, RL rollouts) that can change device."""

    def to(self, device: torch.device) -> Any: ...


class RowSource(Protocol):
    def batch(self, indices: Sequence[int]) -> Movable: ...


def row_indices(step: int, micro: int, plan: BatchPlan, rank: int, offset: int = 0) -> range:
    base = ((step * plan.grad_accum + micro) * plan.world_size + rank) * plan.micro_batch_size
    return range(offset + base, offset + base + plan.micro_batch_size)


class RowProvider:
    """Stateless provider over any row-indexed source (memmap sampler, packed SFT rows)."""

    def __init__(
        self, source: RowSource, plan: BatchPlan, rank: int, device: torch.device, offset: int = 0
    ) -> None:
        self.source = source
        self.plan = plan
        self.rank = rank
        self.device = device
        self.offset = offset

    def batch(self, step: int, micro: int) -> Any:
        rows = row_indices(step, micro, self.plan, self.rank, self.offset)
        return self.source.batch(rows).to(self.device)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None
