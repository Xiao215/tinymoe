"""Learning-rate schedules as multipliers of the base LR.

WSD - warmup / stable / decay (MiniCPM, Hu et al., 2024; DeepSeek-V3; "Scaling Laws and
Compute-Optimal Training Beyond Fixed Training Durations", Hägele et al., 2024). The long
constant phase means any stable checkpoint can be *continued*, and the short decay is where
mid-training "anneals" on high-quality data. ``decay_shape: sqrt`` is the 1-sqrt cooldown
that Hägele et al. found best. Cosine (Loshchilov & Hutter) is kept for comparison.
"""

from __future__ import annotations

import math

from tinylab.config import TrainConfig


def lr_multiplier(step: int, cfg: TrainConfig) -> float:
    """Multiplier in [min_lr_ratio, 1] for optimizer step ``step`` (0-based)."""
    total = max(cfg.max_steps, 1)
    floor = cfg.min_lr_ratio
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "cosine":
        span = max(total - cfg.warmup_steps, 1)
        progress = min((step - cfg.warmup_steps) / span, 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    decay_steps = round(total * cfg.decay_frac)
    decay_start = total - decay_steps
    if decay_steps == 0 or step < decay_start:
        return 1.0
    progress = min((step - decay_start + 1) / decay_steps, 1.0)
    if cfg.decay_shape == "linear":
        frac = 1 - progress
    elif cfg.decay_shape == "sqrt":
        frac = 1 - math.sqrt(progress)
    else:
        frac = 0.5 * (1 + math.cos(math.pi * progress))
    return floor + (1 - floor) * frac
