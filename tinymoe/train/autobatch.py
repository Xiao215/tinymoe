"""``micro_batch_size: auto`` - the largest micro batch whose train step fits.

Doubling then binary search; a size "fits" if one forward+backward completes without an
out-of-memory error *and* (on accelerators) device memory stays under
``hardware.memory_budget_gb``. Unified-memory Macs rarely raise OOM before swapping, so
the budget check is what keeps a 16 GB machine responsive.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from collections.abc import Callable, Iterator

import torch

from tinymoe import hardware
from tinymoe.config import HardwareConfig, TrainConfig
from tinymoe.data.batch import Batch
from tinymoe.hardware import Runtime
from tinymoe.model.transformer import Transformer
from tinymoe.train.losses import lm_loss
from tinymoe.train.optim import build_optimizer

log = logging.getLogger(__name__)


def fits(
    step: Callable[[int], None], size: int, device: torch.device, budget: float | None
) -> tuple[bool, float | None]:
    """(fits, measured peak bytes or None where the backend reports no device memory)."""
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
        return False, None
    finally:
        gc.collect()
    if not hardware.has_memory_stats(device):
        return True, None
    peak = float(hardware.memory_bytes(device)["peak"])
    hardware.empty_cache(device)
    ok = budget is None or peak <= budget
    log.info(
        "[autobatch] micro batch %d: %.2f GB (%s)", size, peak / 1e9, "ok" if ok else "over budget"
    )
    return ok, peak


def find_micro_batch(
    step: Callable[[int], None], max_size: int, device: torch.device, budget_gb: float | None
) -> int:
    """Largest size in [1, max_size] whose step fits.

    Doubling, then bisection. Where peak memory is measurable, a size whose *projected*
    peak (linear in the batch from the last two measurements) exceeds the budget is not
    even tried, so the search never pushes a unified-memory machine into swap.
    """
    budget = budget_gb * 1e9 if budget_gb else None
    peaks: dict[int, float] = {}

    def attempt(size: int) -> bool:
        if budget is not None and len(peaks) >= 2:
            (s0, p0), (s1, p1) = sorted(peaks.items())[-2:]
            projected = p1 + (p1 - p0) / (s1 - s0) * (size - s1)
            if projected > budget:
                log.info(
                    "[autobatch] micro batch %d: projected %.2f GB > budget, skipped",
                    size,
                    projected / 1e9,
                )
                return False
        ok, peak = fits(step, size, device, budget)
        if ok and peak is not None:
            peaks[size] = peak
        return ok

    if not attempt(1):
        raise RuntimeError("even micro_batch_size=1 does not fit in memory")
    good, size = 1, 2
    while size <= max_size and attempt(size):
        good, size = size, size * 2
    bad = min(size, max_size + 1)
    while bad - good > 1:
        mid = (good + bad) // 2
        if attempt(mid):
            good = mid
        else:
            bad = mid
    log.info("[autobatch] selected micro_batch_size=%d", good)
    return good


@contextlib.contextmanager
def worst_case_routing(model: Transformer) -> Iterator[None]:
    """Padded MoE buffers at their cap while probing (see model/moe.py ``worst_case``)."""
    layers = model.moe_layers()
    for m in layers:
        m.worst_case = True
    try:
        yield
    finally:
        for m in layers:
            m.worst_case = False


def model_micro_batch(
    model: Transformer,
    runtime: Runtime,
    seq_len: int,
    requested: int | str,
    max_size: int,
    hw: HardwareConfig,
    probe_loss: Callable[[Transformer, Batch], torch.Tensor] | None = None,
    train: TrainConfig | None = None,
) -> int:
    """Resolve ``micro_batch_size`` (an int, or ``"auto"``) for a training step on ``model``.

    The probe mirrors a real optimizer step at its memory worst case: MoE padded buffers at
    their cap (:func:`worst_case_routing`), two accumulated forward/backward passes (the
    second backward runs while the first's gradients are resident) of the real loss (CE +
    MoE aux + MTP + z-loss, or ``probe_loss`` for stages whose loss costs more, e.g. a
    distillation teacher forward), then - given ``train`` - a step of the real optimizer
    (Muon / AdamW, with its moments and Newton-Schulz workspace). That step runs at lr 0 on
    zeroed gradients, so the weights are untouched; its state is discarded. Without
    ``train`` two fp32 moments per trainable parameter are reserved instead. Only
    ``autobatch_headroom`` of the budget may be used: fragmentation and generation need
    slack. (Measured on the Mac preset: forward/backward alone under-estimated the real
    step's peak by ~2 GB at micro batch 8, which OOMed - docs/decisions.md D12.)
    """
    if isinstance(requested, int):
        return requested
    budget_gb = None
    if hw.memory_budget_gb:
        reserved = 0.0
        if train is None:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            reserved = 8 * n_params / 1e9
        budget_gb = max(hw.memory_budget_gb * hw.autobatch_headroom - reserved, 0.1)
        log.info(
            "[autobatch] budget %.2f GB = %.1f GB x %.2f headroom - %.2f GB reserved "
            "(optimizer %s)",
            budget_gb,
            hw.memory_budget_gb,
            hw.autobatch_headroom,
            reserved,
            "probed" if train is not None else "estimated",
        )

    def step(size: int) -> None:
        with worst_case_routing(model):
            probe_step(size)

    def probe_step(size: int) -> None:
        x = torch.randint(0, model.cfg.vocab_size, (size, seq_len + 1), device=runtime.device)
        batch = Batch(x[:, :-1], x[:, 1:])
        for _ in range(2):
            with runtime.autocast(), runtime.sdpa_context():
                if probe_loss is None:
                    loss = lm_loss(model, batch, model.cfg.mtp.loss_weight, 1e-4).loss
                else:
                    loss = probe_loss(model, batch)
            torch.autograd.backward(loss)
        if train is not None:
            opt = build_optimizer(model, train, runtime, quiet=True)
            opt.set_lr_multiplier(0.0)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            opt.step()
            del opt
        model.zero_grad(set_to_none=True)
        model.load_counts()

    return find_micro_batch(step, max_size, runtime.device, budget_gb)
