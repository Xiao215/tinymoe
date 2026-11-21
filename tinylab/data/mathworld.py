"""MathWorld: a synthetic arithmetic "world" a 20-50M-active-parameter model can master.

Why synthetic: open-domain reasoning is out of reach at this scale, but multi-step
integer arithmetic with explicit worked traces is learnable, verifiable exactly, and has
a tunable difficulty (digits x steps x operations). That makes every post-training stage
(SFT format learning, DPO preferences, RL with verifiable rewards, distillation) produce a
visible, measurable change - the same reason labs use GSM8K/MATH-style verifiable data.

Everything is generated deterministically from ``(seed, salt, index)``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from tinylab.config import MathWorldConfig

NAMES = ("Tom", "Lily", "Sam", "Mia", "Ben", "Anna", "Max", "Zoe", "Leo", "Ella")
ITEMS = ("apple", "ball", "book", "coin", "star", "flower", "cookie", "stone", "shell", "toy")


@dataclass(frozen=True)
class Bucket:
    """Difficulty bucket: operand digit count x number of operations."""

    digits: int
    steps: int

    @property
    def name(self) -> str:
        return f"d{self.digits}s{self.steps}"

    @classmethod
    def parse(cls, name: str) -> Bucket:
        match = re.fullmatch(r"d(\d+)s(\d+)", name)
        if not match:
            raise ValueError(f"bad bucket name '{name}'")
        return cls(int(match.group(1)), int(match.group(2)))


def buckets_from(cfg: MathWorldConfig) -> list[Bucket]:
    return [Bucket(d, s) for d in cfg.digits for s in cfg.steps]


@dataclass(frozen=True)
class Problem:
    """A verifiable problem with its worked solution."""

    question: str
    answer: int
    steps: tuple[str, ...]
    bucket: str
    kind: str
    start: int = 0
    chain: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    def trace(self) -> str:
        return "\n".join(self.steps)


def rng_for(seed: int, salt: str, index: int) -> random.Random:
    """Independent, reproducible stream per (seed, salt, index) - order independent."""
    return random.Random(f"{salt}:{seed}:{index}")


def _operand(rng: random.Random, digits: int) -> int:
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(lo, 10**digits - 1)


def _apply(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def run_chain(start: int, chain: Sequence[tuple[str, int]]) -> tuple[int, list[str]]:
    """Evaluate a chain and produce one worked line per operation."""
    value = start
    lines = []
    for op, b in chain:
        new = _apply(value, op, b)
        lines.append(f"{value} {op} {b} = {new}")
        value = new
    return value, lines


def expression(start: int, chain: Sequence[tuple[str, int]]) -> str:
    """Infix string with parentheses wherever left-to-right differs from precedence."""
    expr = str(start)
    prev_low = False  # does expr end in a +/- at top level?
    for op, b in chain:
        if op == "*" and prev_low:
            expr = f"({expr})"
        expr = f"{expr} {op} {b}"
        prev_low = op in "+-"
    return expr


def _arithmetic(rng: random.Random, bucket: Bucket, cfg: MathWorldConfig) -> Problem:
    start = _operand(rng, bucket.digits)
    chain = _make_chain(rng, start, bucket, cfg)
    answer, lines = run_chain(start, chain)
    question = f"What is {expression(start, chain)}?"
    return Problem(question, answer, tuple(lines), bucket.name, "arithmetic", start, tuple(chain))


def _make_chain(
    rng: random.Random, start: int, bucket: Bucket, cfg: MathWorldConfig
) -> list[tuple[str, int]]:
    value = start
    chain: list[tuple[str, int]] = []
    for _ in range(bucket.steps):
        op = rng.choice(cfg.ops)
        if op == "*":
            b = rng.randint(2, max(2, min(cfg.max_mul_operand, 10**bucket.digits - 1)))
        elif op == "-":
            b = rng.randint(0, min(value, 10**bucket.digits - 1))
        else:
            b = _operand(rng, bucket.digits)
        chain.append((op, b))
        value = _apply(value, op, b)
    return chain


def _word(rng: random.Random, bucket: Bucket, cfg: MathWorldConfig) -> Problem:
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    start = _operand(rng, bucket.digits)
    chain = _make_chain(rng, start, bucket, cfg)
    sentences = [f"{name} has {start} {item}s."]
    for op, b in chain:
        if op == "+":
            sentences.append(f"{name} finds {b} more {item}s.")
        elif op == "-":
            sentences.append(f"{name} gives away {b} {item}s.")
        else:
            sentences.append(f"Then the number of {item}s becomes {b} times as many.")
    sentences.append(f"How many {item}s does {name} have now?")
    answer, lines = run_chain(start, chain)
    return Problem(
        " ".join(sentences), answer, tuple(lines), bucket.name, "word", start, tuple(chain)
    )


def _facts(rng: random.Random, bucket: Bucket, cfg: MathWorldConfig) -> Problem:
    n = _operand(rng, bucket.digits)
    choice = rng.randrange(3)
    if choice == 0:
        question, op, b = f"What number comes after {n}?", "+", 1
    elif choice == 1 and n > 0:
        question, op, b = f"What number comes before {n}?", "-", 1
    else:
        question, op, b = f"What is double {n}?", "*", 2
    answer, lines = run_chain(n, [(op, b)])
    return Problem(question, answer, tuple(lines), bucket.name, "facts", n, ((op, b),))


def _algebra(rng: random.Random, bucket: Bucket, cfg: MathWorldConfig) -> Problem:
    x = _operand(rng, bucket.digits)
    steps: tuple[str, ...]
    if bucket.steps >= 2 and "*" in cfg.ops:
        m = rng.randint(2, max(2, min(cfg.max_mul_operand, 9)))
        c = _operand(rng, bucket.digits)
        rhs = x * m + c
        question = f"If x * {m} + {c} = {rhs}, what is x?"
        steps = (f"x * {m} = {rhs} - {c} = {rhs - c}", f"x = {rhs - c} / {m} = {x}")
    else:
        c = _operand(rng, bucket.digits)
        rhs = x + c
        question = f"If x + {c} = {rhs}, what is x?"
        steps = (f"x = {rhs} - {c} = {x}",)
    return Problem(question, x, steps, bucket.name, "algebra")


GENERATORS = {"arithmetic": _arithmetic, "word": _word, "facts": _facts, "algebra": _algebra}
SOLVABLE_KINDS = ("arithmetic", "word")  # kinds that span every (digits, steps) bucket


def make_problem(
    rng: random.Random, bucket: Bucket, cfg: MathWorldConfig, kinds: Sequence[str] = SOLVABLE_KINDS
) -> Problem:
    kind = rng.choice(list(kinds))
    return GENERATORS[kind](rng, bucket, cfg)


def problem_set(
    cfg: MathWorldConfig, seed: int, salt: str, n_per_bucket: int, buckets: Sequence[Bucket]
) -> dict[str, list[Problem]]:
    """Fixed problems per bucket (eval sets, RL prompts, SFT data)."""
    out: dict[str, list[Problem]] = {}
    for bucket in buckets:
        out[bucket.name] = [
            make_problem(rng_for(seed, f"{salt}:{bucket.name}", i), bucket, cfg)
            for i in range(n_per_bucket)
        ]
    return out


# ------------------------------------------------------------ pre-training text


def _weighted_kind(rng: random.Random, weights: dict[str, float]) -> str:
    kinds = [k for k, w in weights.items() if w > 0 and k in GENERATORS]
    return rng.choices(kinds, weights=[weights[k] for k in kinds])[0]


def pretrain_document(
    cfg: MathWorldConfig, seed: int, index: int, split: str, hq: bool = False
) -> str:
    """One pre-training document: several problems in prose with worked solutions.

    ``hq=True`` produces the mid-training "annealing" flavour: harder buckets only and
    always fully worked, step-by-step solutions (Llama-3 / OLMo-2 style high-quality
    upsampling during the LR decay).
    """
    rng = rng_for(seed, f"doc:{split}:{'hq' if hq else 'base'}", index)
    min_steps = cfg.hq_min_steps if hq else 1
    steps = [s for s in cfg.steps if s >= min_steps] or [max(cfg.steps)]
    parts = []
    for _ in range(rng.randint(2, 4)):
        bucket = Bucket(rng.choice(cfg.digits), rng.choice(steps))
        kind = rng.choice(SOLVABLE_KINDS) if hq else _weighted_kind(rng, cfg.kind_weights)
        problem = GENERATORS[kind](rng, bucket, cfg)
        style = 0 if hq else rng.randrange(3)
        if style == 0:
            work = "\n".join(f"Step {i + 1}: {line}." for i, line in enumerate(problem.steps))
            parts.append(f"Question: {problem.question}\n{work}\nAnswer: {problem.answer}")
        elif style == 1:
            parts.append(f"Question: {problem.question}\nAnswer: {problem.answer}")
        else:
            parts.append(". ".join(problem.steps) + ".")
    return "\n\n".join(parts)


# --------------------------------------------------------------------- verifier


ANSWER_RE = re.compile(r"^\s*Answer:\s*(-?[0-9]{1,12})\s*\.?\s*$")  # 12 digits max
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@dataclass(frozen=True)
class Verdict:
    correct: bool
    format_ok: bool
    parsed: int | None
    reasons: tuple[str, ...] = ()


def extract_answer(response: str) -> int | None:
    """The single ``Answer: N`` line after the reasoning, else None.

    Lines inside ``<think>`` (closed or not) never count, and two answer lines (hedging)
    yield None.
    """
    tail = response.rsplit(THINK_CLOSE, 1)[-1] if THINK_CLOSE in response else response
    if THINK_OPEN in tail:  # an unclosed <think>: everything after it is still reasoning
        return None
    found = [m.group(1) for line in tail.splitlines() if (m := ANSWER_RE.match(line))]
    if len(found) != 1:
        return None
    return int(found[0])


def format_problems(response: str) -> list[str]:
    """Format check: <think>...</think> first, then exactly one answer line, nothing else."""
    reasons = []
    text = response.strip()
    if text.count(THINK_OPEN) != 1 or text.count(THINK_CLOSE) != 1:
        reasons.append("needs exactly one <think> and one </think>")
        return reasons
    if not text.startswith(THINK_OPEN):
        reasons.append("response must start with <think>")
    if text.index(THINK_OPEN) > text.index(THINK_CLOSE):
        reasons.append("</think> before <think>")
        return reasons
    tail = text.split(THINK_CLOSE, 1)[1]
    lines = [line for line in tail.splitlines() if line.strip()]
    if len(lines) != 1 or not ANSWER_RE.match(lines[0]):
        reasons.append("after </think> there must be exactly one 'Answer: N' line")
    return reasons


def verify(response: str, answer: int) -> Verdict:
    parsed = extract_answer(response)
    reasons = format_problems(response)
    return Verdict(parsed == answer, not reasons, parsed, tuple(reasons))


def solution_text(problem: Problem, think: bool = True) -> str:
    """Canonical assistant response used by SFT/DPO (the verifier accepts it)."""
    body = problem.trace() if think else ""
    return f"{THINK_OPEN}\n{body}\n{THINK_CLOSE}\nAnswer: {problem.answer}"
