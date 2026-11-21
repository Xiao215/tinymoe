"""Chat-formatted SFT data with assistant-only loss masks, packed or padded.

Packing concatenates several short conversations into one row; ``doc_ids`` give each one
its own attention block and positions restart per example ("packing with per-example
attention isolation", as in Llama-3 / the HF ``DataCollatorWithFlattening``). With
``packing: false`` every example gets its own right-padded row; right padding under a
causal mask needs no extra attention mask because real tokens never see later pads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinylab.config import DataConfig
from tinylab.data import mathworld, stories
from tinylab.data.batch import IGNORE_INDEX, Batch, stack_rows
from tinylab.data.mathworld import Problem
from tinylab.data.tokenizer import Message, Tokenizer, render_chat


@dataclass(frozen=True)
class ChatExample:
    messages: tuple[Message, ...]
    kind: str  # "math" | "story"
    answer: int | None = None
    bucket: str | None = None
    subject: str | None = None


def math_prompt(problem: Problem) -> tuple[Message, ...]:
    return (Message("user", problem.question),)


def math_example(problem: Problem, think: bool = True) -> ChatExample:
    reply = Message("assistant", mathworld.solution_text(problem, think))
    return ChatExample((*math_prompt(problem), reply), "math", problem.answer, problem.bucket)


def story_instruction(subject: str) -> str:
    return f"Write a short story about a {subject}."


def story_examples(texts: Sequence[str]) -> list[ChatExample]:
    out = []
    for text in texts:
        subject = stories.find_subject(text)
        if subject is None:
            continue
        msgs = (Message("user", story_instruction(subject)), Message("assistant", text))
        out.append(ChatExample(msgs, "story", subject=subject))
    return out


def math_examples(
    cfg: DataConfig, n: int, seed: int, salt: str, think: bool = True
) -> list[ChatExample]:
    """``n`` problems cycling through every difficulty bucket."""
    buckets = mathworld.buckets_from(cfg.mathworld)
    return [
        math_example(
            mathworld.make_problem(
                mathworld.rng_for(seed, salt, i), buckets[i % len(buckets)], cfg.mathworld
            ),
            think,
        )
        for i in range(n)
    ]


def build_sft_examples(
    cfg: DataConfig, n_math: int, n_story: int, think: bool, seed: int
) -> list[ChatExample]:
    texts = stories.load_stories(cfg.stories, "train", n_story * 2, Path(cfg.cache_dir), cfg.seed)
    story = story_examples(texts)[:n_story]
    return math_examples(cfg, n_math, seed, "sft", think) + story


def encode(
    tok: Tokenizer, examples: Sequence[ChatExample], max_len: int
) -> list[tuple[list[int], list[int]]]:
    """Render + tokenise; examples longer than ``max_len + 1`` tokens are dropped."""
    out = []
    for ex in examples:
        ids, mask = render_chat(tok, ex.messages)
        if len(ids) <= max_len + 1:
            out.append((ids, mask))
    return out


@dataclass
class Row:
    tokens: np.ndarray  # (seq_len + 1,)
    mask: np.ndarray  # (seq_len + 1,) 1 = learn this token
    doc_ids: np.ndarray  # (seq_len + 1,) 0 = padding
    positions: np.ndarray  # (seq_len + 1,)


def _make_row(chunks: Sequence[tuple[list[int], list[int]]], length: int, pad_id: int) -> Row:
    tokens = np.full(length, pad_id, dtype=np.int64)
    mask = np.zeros(length, dtype=np.int64)
    doc_ids = np.zeros(length, dtype=np.int64)
    positions = np.zeros(length, dtype=np.int64)
    cursor = 0
    for doc, (ids, m) in enumerate(chunks, start=1):
        n = len(ids)
        tokens[cursor : cursor + n] = ids
        mask[cursor : cursor + n] = m
        doc_ids[cursor : cursor + n] = doc
        positions[cursor : cursor + n] = np.arange(n)
        cursor += n
    positions[cursor:] = np.arange(length - cursor)
    return Row(tokens, mask, doc_ids, positions)


def pack(
    encoded: Sequence[tuple[list[int], list[int]]], seq_len: int, pad_id: int, seed: int
) -> list[Row]:
    """Greedy packing in a seeded random order into rows of ``seq_len + 1`` tokens."""
    order = np.random.default_rng([seed, 7]).permutation(len(encoded))
    rows: list[Row] = []
    current: list[tuple[list[int], list[int]]] = []
    used = 0
    for idx in order:
        item = encoded[int(idx)]
        if current and used + len(item[0]) > seq_len + 1:
            rows.append(_make_row(current, seq_len + 1, pad_id))
            current, used = [], 0
        current.append(item)
        used += len(item[0])
    if current:
        rows.append(_make_row(current, seq_len + 1, pad_id))
    return rows


def pad(encoded: Sequence[tuple[list[int], list[int]]], seq_len: int, pad_id: int) -> list[Row]:
    return [_make_row([item], seq_len + 1, pad_id) for item in encoded]


class RowDataset:
    """Deterministic, epoch-shuffled access to fixed rows by global row index."""

    def __init__(self, rows: Sequence[Row], seed: int, isolate_docs: bool) -> None:
        if not rows:
            raise ValueError("RowDataset needs at least one row")
        self.rows = list(rows)
        self.seed = seed
        self.isolate_docs = isolate_docs

    def __len__(self) -> int:
        return len(self.rows)

    def _row(self, index: int) -> Row:
        epoch, pos = divmod(index, len(self.rows))
        perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.rows))
        return self.rows[int(perm[pos])]

    def batch(self, indices: Sequence[int]) -> Batch:
        rows = [self._row(i) for i in indices]
        return rows_to_batch(rows, self.isolate_docs)


def rows_to_batch(rows: Sequence[Row], isolate_docs: bool) -> Batch:
    inputs = [r.tokens[:-1] for r in rows]
    labels = [np.where(r.mask[1:] == 1, r.tokens[1:], IGNORE_INDEX) for r in rows]
    if not isolate_docs:
        return stack_rows(inputs, labels)
    return stack_rows(
        inputs,
        labels,
        doc_ids=[r.doc_ids[:-1] for r in rows],
        positions=[r.positions[:-1] for r in rows],
    )


def make_rows(
    tok: Tokenizer, examples: Sequence[ChatExample], seq_len: int, packing: bool, seed: int
) -> list[Row]:
    encoded = encode(tok, examples, seq_len)
    if packing:
        return pack(encoded, seq_len, tok.pad_id, seed)
    return pad(encoded, seq_len, tok.pad_id)
