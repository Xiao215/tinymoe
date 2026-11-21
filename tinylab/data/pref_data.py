"""Preference pairs for DPO: chosen = correct + well formatted, rejected = a specific flaw.

Synthetic rejections mirror the failure modes the verifier detects:

* ``wrong_answer`` - one intermediate step is miscomputed and the error is *propagated*,
  so the trace stays internally consistent but the answer is wrong (a hard negative);
* ``no_format``    - right numbers, wrong output format (no think tags / no answer line);
* ``rambling``     - repetitive filler that never commits to the right answer.

Pairs can also be mined from the SFT model's own samples, ranked by the verifier (the
"on-policy preference data" recipe used by Tulu-3 / Llama-3 rejection sampling).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tinylab.config import DataConfig, DPOConfig
from tinylab.data import mathworld, stories
from tinylab.data.mathworld import Problem, solution_text
from tinylab.data.sft_data import math_prompt, story_instruction
from tinylab.data.tokenizer import Message


@dataclass(frozen=True)
class PrefPair:
    prompt: tuple[Message, ...]
    chosen: str
    rejected: str
    kind: str


def wrong_answer(problem: Problem, rng: random.Random) -> str:
    if problem.chain:
        chain = list(problem.chain)
        i = rng.randrange(len(chain))
        value = problem.start
        lines = []
        for j, (op, b) in enumerate(chain):
            new = mathworld.run_chain(value, [(op, b)])[0]
            if j == i:
                new += rng.choice([-3, -2, -1, 1, 2, 3, 10])
            lines.append(f"{value} {op} {b} = {new}")
            value = new
        if value != problem.answer:
            return f"<think>\n{chr(10).join(lines)}\n</think>\nAnswer: {value}"
    wrong = problem.answer + rng.choice([-2, -1, 1, 2, 10])
    return f"<think>\n{problem.trace()}\n</think>\nAnswer: {wrong}"


def no_format(problem: Problem, rng: random.Random) -> str:
    variants = [
        f"The answer is {problem.answer}.",
        f"{problem.trace()}\nAnswer: {problem.answer}",
        f"<think>\n{problem.trace()}\nAnswer: {problem.answer}",
        f"Answer: {problem.answer}\nAnswer: {problem.answer}",
    ]
    return rng.choice(variants)


def rambling(problem: Problem, rng: random.Random) -> str:
    filler = ["Let me think.", "Hmm, numbers.", "I am not sure.", "Let me count again."]
    lines = [rng.choice(filler) for _ in range(rng.randint(4, 8))]
    tail = rng.choice(["", f"Answer: {problem.answer + rng.randint(1, 9)}"])
    return f"<think>\n{chr(10).join(lines)}\n</think>\n{tail}".rstrip()


REJECTORS = {"wrong_answer": wrong_answer, "no_format": no_format, "rambling": rambling}


def build_pairs(cfg: DataConfig, dpo: DPOConfig, seed: int) -> list[PrefPair]:
    """Synthetic math pairs (+ optional story pairs) - deterministic for a given seed."""
    buckets = mathworld.buckets_from(cfg.mathworld)
    kinds = [k for k, w in dpo.rejected_kinds.items() if w > 0]
    weights = [dpo.rejected_kinds[k] for k in kinds]
    n_story = int(dpo.n_pairs * dpo.story_pairs_fraction)
    pairs = []
    for i in range(dpo.n_pairs - n_story):
        rng = mathworld.rng_for(seed, "dpo", i)
        problem = mathworld.make_problem(rng, buckets[i % len(buckets)], cfg.mathworld)
        kind = rng.choices(kinds, weights=weights)[0]
        rejected = REJECTORS[kind](problem, rng)
        pairs.append(PrefPair(math_prompt(problem), solution_text(problem), rejected, kind))
    if n_story:
        pairs += story_pairs(cfg, n_story, seed)
    return pairs


def story_pairs(cfg: DataConfig, n: int, seed: int) -> list[PrefPair]:
    """Chosen = story about the requested subject; rejected = story about another one."""
    texts = stories.load_stories(cfg.stories, "train", n * 4, Path(cfg.cache_dir), cfg.seed)
    by_subject: dict[str, list[str]] = {}
    for text in texts:
        subject = stories.find_subject(text)
        if subject:
            by_subject.setdefault(subject, []).append(text)
    subjects = sorted(by_subject)
    pairs: list[PrefPair] = []
    if len(subjects) < 2:
        return pairs
    rng = random.Random(f"story-pairs:{seed}")
    for i in range(n):
        subject = subjects[i % len(subjects)]
        other = rng.choice([s for s in subjects if s != subject])
        chosen = rng.choice(by_subject[subject])
        rejected = rng.choice(
            [t for t in by_subject[other] if not stories.mentions(t, subject)] or [""]
        )
        if rejected:
            prompt = (Message("user", story_instruction(subject)),)
            pairs.append(PrefPair(prompt, chosen, rejected, "story_off_topic"))
    return pairs


def pairs_from_samples(
    problems: Sequence[Problem], samples: Sequence[Sequence[str]]
) -> list[PrefPair]:
    """Verifier-ranked pairs from model samples: first passing vs first failing response."""
    pairs = []
    for problem, responses in zip(problems, samples, strict=True):
        verdicts = [mathworld.verify(r, problem.answer) for r in responses]
        good = [r for r, v in zip(responses, verdicts, strict=True) if v.correct and v.format_ok]
        bad = [
            r for r, v in zip(responses, verdicts, strict=True) if not (v.correct and v.format_ok)
        ]
        if good and bad:
            pairs.append(PrefPair(math_prompt(problem), good[0], bad[0], "model_sampled"))
    return pairs
