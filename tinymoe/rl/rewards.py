"""Verifiable rewards for MathWorld rollouts (RLVR, as in DeepSeek-R1 / Tulu-3).

``reward = reward_correct * correct + reward_format * format_ok + overlong``

* correctness and format are separate verifier signals (data/mathworld.py), as in
  DeepSeek-R1's accuracy + format rewards: "correct" means the single ``Answer:`` line
  outside the reasoning matches (an answer inside ``<think>`` never counts); "format"
  additionally requires the ``<think> ... </think>`` layout, so skipping the reasoning
  forfeits only the format bonus;
* DAPO soft overlong punishment (Yu et al., 2025): with ``L = max_new_tokens`` and
  ``C = overlong.cache_len``, a response of length ``n`` gets ``0`` if ``n <= L - C``,
  ``-penalty * (n - (L - C)) / C`` inside the cache window, and ``-penalty`` if truncated.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinymoe.config import GRPOConfig
from tinymoe.data import mathworld


@dataclass(frozen=True)
class Reward:
    total: float
    correct: bool
    format_ok: bool
    overlong: float


def overlong_penalty(n_tokens: int, finished: bool, max_new: int, cfg: GRPOConfig) -> float:
    o = cfg.overlong
    if not o.enabled:
        return 0.0
    if not finished:
        return -o.penalty
    soft_start = max_new - o.cache_len
    if n_tokens <= soft_start:
        return 0.0
    return -o.penalty * min((n_tokens - soft_start) / max(o.cache_len, 1), 1.0)


def score(
    text: str, answer: int, n_tokens: int, finished: bool, max_new: int, cfg: GRPOConfig
) -> Reward:
    v = mathworld.verify(text, answer)
    over = overlong_penalty(n_tokens, finished, max_new, cfg)
    total = cfg.reward_correct * v.correct + cfg.reward_format * v.format_ok + over
    return Reward(total, v.correct, v.format_ok, over)
