"""Difficulty-bucketed RL prompt sets and bucket calibration.

RL with group-relative advantages only learns from groups whose rewards *differ*: a
bucket the policy always solves (or never solves) yields zero advantage. Calibration
therefore keeps the buckets where the SFT model's pass@1 sits inside ``[low, high]``
(the "zone of proximal development" filtering used by DAPO, Kimi k1.5, Skywork-OR1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tinylab.config import MathWorldConfig
from tinylab.data import mathworld
from tinylab.data.mathworld import Bucket, Problem


def rl_problems(
    cfg: MathWorldConfig, buckets: Sequence[Bucket], n_per_bucket: int, seed: int, salt: str = "rl"
) -> dict[str, list[Problem]]:
    return mathworld.problem_set(cfg, seed, salt, n_per_bucket, buckets)


def select_buckets(
    pass_rates: Mapping[str, float], low: float, high: float, max_buckets: int
) -> list[str]:
    """Buckets with pass@1 in ``[low, high]``, closest to the band centre first.

    If none qualify, fall back to the buckets nearest the band (preferring ones the model
    solves at least sometimes), so RL always has something to train on.
    """
    if not pass_rates:
        raise ValueError("no pass rates to calibrate from")
    centre = (low + high) / 2
    inside = sorted((abs(r - centre), b) for b, r in pass_rates.items() if low <= r <= high)
    if inside:
        return [b for _, b in inside[:max_buckets]]

    def distance(rate: float) -> float:
        return low - rate if rate < low else rate - high

    ranked = sorted(pass_rates.items(), key=lambda kv: (kv[1] == 0.0, distance(kv[1])))
    return [b for b, _ in ranked[:max_buckets]]
