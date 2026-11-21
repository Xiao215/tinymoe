"""``micro_batch_size: auto`` - the largest micro batch whose train step fits.

Doubling then binary search; a size "fits" if one forward+backward completes without an
out-of-memory error *and* (on accelerators) device memory stays under
``hardware.memory_budget_gb``. Unified-memory Macs rarely raise OOM before swapping, so
the budget check is what keeps a 16 GB machine responsive.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable

import torch

from tinylab import hardware

log = logging.getLogger(__name__)


def fits(
    step: Callable[[int], None], size: int, device: torch.device, budget: float | None
) -> bool:
    gc.collect()
    hardware.empty_cache(device)
    hardware.reset_peak_memory(device)
    try:
        step(size)
        hardware.synchronize(device)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if not hardware.is_oom(exc):
            raise
        log.info("[autobatch] micro batch %d: out of memory", size)
        return False
    finally:
        gc.collect()
    if budget is None or not hardware.has_memory_stats(device):
        return True
    peak = hardware.memory_bytes(device)["peak"]
    hardware.empty_cache(device)
    ok = peak <= budget
    log.info(
        "[autobatch] micro batch %d: %.2f GB (%s)", size, peak / 1e9, "ok" if ok else "over budget"
    )
    return ok


def find_micro_batch(
    step: Callable[[int], None], max_size: int, device: torch.device, budget_gb: float | None
) -> int:
    budget = budget_gb * 1e9 if budget_gb else None
    if not fits(step, 1, device, budget):
        raise RuntimeError("even micro_batch_size=1 does not fit in memory")
    good, size = 1, 2
    while size <= max_size and fits(step, size, device, budget):
        good, size = size, size * 2
    bad = min(size, max_size + 1)
    while bad - good > 1:
        mid = (good + bad) // 2
        if fits(step, mid, device, budget):
            good = mid
        else:
            bad = mid
    log.info("[autobatch] selected micro_batch_size=%d", good)
    return good
