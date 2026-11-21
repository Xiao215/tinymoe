"""The one batch type every stage feeds to the model and the trainer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

IGNORE_INDEX = -100


@dataclass
class Batch:
    """Token batch. Optional fields switch on model features:

    * ``position_ids`` - per-token RoPE positions (reset per packed document).
    * ``doc_ids``      - block-diagonal attention: tokens attend only within their doc.
    * ``attention_mask`` - key padding mask (True = real token), for padded rows.
    * ``source_ids``   - per-row data source index, for per-source losses.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor | None = None
    doc_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    source_ids: torch.Tensor | None = None

    def to(self, device: torch.device) -> Batch:
        moved = {
            f.name: value.to(device)
            for f in fields(self)
            if isinstance(value := getattr(self, f.name), torch.Tensor)
        }
        return replace(self, **moved)

    @property
    def n_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum().item())


def shuffled_index(index: int, n: int, seed: int) -> int:
    """Map a global sample index to an item: a fresh permutation of ``range(n)`` per epoch,
    seeded by ``(seed, epoch)``, so any row is computable without iterator state."""
    epoch, pos = divmod(index, n)
    return int(np.random.default_rng([seed, epoch]).permutation(n)[pos])


def doc_layout(tokens: np.ndarray, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Doc ids and per-doc positions for a packed row (a new doc starts at each BOS)."""
    starts = tokens == bos_id
    doc_ids = np.cumsum(starts)
    idx = np.arange(len(tokens))
    last_start = np.maximum.accumulate(np.where(starts, idx, 0))
    return doc_ids.astype(np.int64), (idx - last_start).astype(np.int64)


def stack_rows(
    inputs: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    doc_ids: Sequence[np.ndarray] | None = None,
    positions: Sequence[np.ndarray] | None = None,
    valid: Sequence[np.ndarray] | None = None,
    sources: Sequence[int] | None = None,
) -> Batch:
    def t(rows: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    return Batch(
        input_ids=t(inputs),
        labels=t(labels),
        doc_ids=t(doc_ids) if doc_ids is not None else None,
        position_ids=t(positions) if positions is not None else None,
        attention_mask=torch.from_numpy(np.stack(valid)) if valid is not None else None,
        source_ids=torch.tensor(list(sources), dtype=torch.long) if sources is not None else None,
    )
