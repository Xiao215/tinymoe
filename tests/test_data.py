from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from tinylab.config import DataConfig, DPOConfig, MathWorldConfig
from tinylab.data import mathworld, pref_data, rl_prompts, sft_data, stories
from tinylab.data.batch import IGNORE_INDEX
from tinylab.data.mathworld import Bucket, verify
from tinylab.data.pretrain_mix import MixtureSampler, build_shards, open_shard, write_shard
from tinylab.data.tokenizer import Message, Tokenizer, render_chat

# ------------------------------------------------------------------ tokenizer


def test_digits_tokenized_individually(tok: Tokenizer) -> None:
    for text in ("12345", "x = 2024", "Tom has 987 apples"):
        ids = tok.encode(text)
        digits = [tok.decode([i]) for i in ids if tok.decode([i]).strip().isdigit()]
        assert all(len(d.strip()) == 1 for d in digits)
        assert "".join(d.strip() for d in digits) == "".join(c for c in text if c.isdigit())


@pytest.mark.parametrize(
    "text",
    [
        "Once upon a time, there was a cat.",
        "What is (12 + 7) * 3?\n<think>\n12 + 7 = 19\n</think>\nAnswer: 57",
        "unicode: café ☕ 日本 🙂",
        "  leading spaces and\ttabs\n\nnewlines  ",
        "",
    ],
)
def test_roundtrip(tok: Tokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens(tok: Tokenizer) -> None:
    ids = tok.encode("<think>hi</think>")
    assert ids[0] == tok.think_id and tok.think_end_id in ids
    assert len(set(tok.special_ids.values())) == len(tok.special_ids)


def test_chat_template_mask(tok: Tokenizer) -> None:
    msgs = [Message("system", "be nice"), Message("user", "hi"), Message("assistant", "hello")]
    ids, mask = render_chat(tok, msgs)
    assert ids[0] == tok.bos_id and len(ids) == len(mask)
    learned = [i for i, m in zip(ids, mask, strict=True) if m]
    assert tok.decode(learned) == "hello<|end|>"
    prompt, pmask = render_chat(tok, msgs[:2], add_generation_prompt=True)
    assert prompt[-1] == tok.assistant_id and sum(pmask) == 0
    assert ids[: len(prompt)] == prompt  # prompt is a prefix of the full conversation


# ------------------------------------------------------------------ mathworld


def test_mathworld_deterministic() -> None:
    cfg = MathWorldConfig()
    a = [mathworld.pretrain_document(cfg, 1, i, "train") for i in range(20)]
    b = [mathworld.pretrain_document(cfg, 1, i, "train") for i in range(20)]
    c = [mathworld.pretrain_document(cfg, 2, i, "train") for i in range(20)]
    assert a == b and a != c
    assert mathworld.problem_set(cfg, 3, "x", 5, [Bucket(2, 2)]) == mathworld.problem_set(
        cfg, 3, "x", 5, [Bucket(2, 2)]
    )


def test_answers_match_expression_semantics() -> None:
    cfg = MathWorldConfig(digits=[1, 2, 3], steps=[1, 2, 3, 4])
    for i in range(300):
        rng = random.Random(i)
        bucket = Bucket(rng.choice(cfg.digits), rng.choice(cfg.steps))
        p = mathworld._arithmetic(rng, bucket, cfg)
        expr = p.question.removeprefix("What is ").removesuffix("?")
        assert eval(expr) == p.answer, (expr, p.answer)  # parentheses encode the order
        assert len(p.steps) == bucket.steps
        assert verify(mathworld.solution_text(p), p.answer).correct


def test_every_kind_is_verifiable() -> None:
    cfg = MathWorldConfig()
    for kind, gen in mathworld.GENERATORS.items():
        for i in range(30):
            p = gen(random.Random(i), Bucket(2, 2), cfg)
            v = verify(mathworld.solution_text(p), p.answer)
            assert v.correct and v.format_ok, (kind, p)


def test_bucket_parse() -> None:
    assert Bucket.parse("d3s2") == Bucket(3, 2) and Bucket(3, 2).name == "d3s2"
    with pytest.raises(ValueError):
        Bucket.parse("hard")


# ------------------------------------------------------------------- verifier

GOOD = "<think>\n2 + 3 = 5\n</think>\nAnswer: 5"


@pytest.mark.parametrize(
    ("response", "parsed", "format_ok"),
    [
        (GOOD, 5, True),
        ("  " + GOOD + "\n\n", 5, True),
        ("<think>\n2 + 3 = 5\n</think>\nAnswer: 5.", 5, True),
        ("<think>x</think>\nAnswer: -7", -7, True),
        ("<think>\nAnswer: 5\n</think>\n", None, False),  # answer only inside think
        ("<think>\n</think>\nAnswer: 5\nAnswer: 6", None, False),  # hedging
        ("<think>\n</think>\nAnswer: 5 or 6", None, False),
        ("<think>\n</think>\nanswer: 5", None, False),  # case sensitive
        ("<think>\n</think>\nAnswer: ５", None, False),  # full-width digit
        ("<think>\n</think>\nAnswer: 5\nThanks!", 5, False),  # trailing chatter
        ("Answer: 5", 5, False),  # no think tags
        ("</think>\n<think>\nAnswer: 5", None, False),  # reversed tags
        ("<think><think></think>\nAnswer: 5", 5, False),  # duplicated tag
        ("<think>\n</think>\nAnswer: 1234567890123456", None, False),  # absurd length
        ("<think>\n</think>\nAnswer: 5</think>", None, False),
        ("", None, False),
    ],
)
def test_verifier_adversarial(response: str, parsed: int | None, format_ok: bool) -> None:
    v = verify(response, 5)
    assert v.parsed == parsed
    assert v.format_ok == format_ok
    assert v.correct == (parsed == 5)


# ---------------------------------------------------------------- shards/mix


def test_shard_memmap_roundtrip(tok: Tokenizer, tmp_path: Path) -> None:
    docs = ["hello world", "What is 2 + 2?", "a\nb"]
    meta = write_shard(tmp_path / "s.bin", docs, tok)
    arr = open_shard(tmp_path / "s.bin")
    assert isinstance(arr, np.memmap) and arr.dtype == np.uint16
    assert len(arr) == meta["n_tokens"]
    bos = np.where(arr == tok.bos_id)[0]
    decoded = [tok.decode(arr[s + 1 : e].tolist()).removesuffix("<|eos|>") for s, e in
               zip(bos, [*bos[1:], len(arr)], strict=True)]  # fmt: skip
    assert decoded == docs


def test_build_shards_cached(smoke_data: DataConfig, tok: Tokenizer) -> None:
    paths = build_shards(smoke_data, tok, ["math"])
    first = paths["math"]["train"].stat().st_mtime_ns
    again = build_shards(smoke_data, tok, ["math"])
    assert again == paths and paths["math"]["train"].stat().st_mtime_ns == first


def test_mixture_sampler(tok: Tokenizer) -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(10, 100, 5000)
    a[::50] = tok.bos_id
    b = rng.integers(10, 100, 5000)
    s = MixtureSampler({"x": a, "y": b}, {"x": 0.7, "y": 0.3}, 32, seed=1, bos_id=tok.bos_id)
    assert (s.row(5)[1] == s.row(5)[1]).all()
    batch = s.batch(range(400))
    assert batch.input_ids.shape == (400, 32) == batch.labels.shape
    assert (batch.input_ids[:, 1:] == batch.labels[:, :-1]).all()
    frac_x = float((batch.source_ids == 0).float().mean())
    assert 0.6 < frac_x < 0.8
    dm = MixtureSampler({"x": a}, {"x": 1.0}, 64, seed=1, bos_id=tok.bos_id, doc_mask=True)
    row = dm.batch([3])
    assert row.position_ids is not None and row.doc_ids is not None
    starts = (row.input_ids[0] == tok.bos_id).nonzero().flatten().tolist()
    for start in starts:
        assert row.position_ids[0, start] == 0


# ------------------------------------------------------------------ sft/pref


def test_sft_packing_masks(tok: Tokenizer, smoke_data: DataConfig) -> None:
    examples = sft_data.build_sft_examples(smoke_data, n_math=20, n_story=5, think=True, seed=0)
    assert {e.kind for e in examples} == {"math", "story"}
    rows = sft_data.make_rows(tok, examples, seq_len=256, packing=True, seed=0)
    assert len(rows) < len(examples)  # packing put several examples per row
    batch = sft_data.RowDataset(rows, seed=0, isolate_docs=True).batch([0, 1])
    assert batch.doc_ids is not None and batch.position_ids is not None
    for r in range(2):
        labels = batch.labels[r]
        learned = labels[labels != IGNORE_INDEX].tolist()
        text = tok.decode(learned)
        assert "<|user|>" not in text and "<|assistant|>" not in text
        assert "Answer:" in text or len(text) > 20
        # every learned label is predicted from inside the same document
        inp_docs = batch.doc_ids[r]
        assert (inp_docs[labels != IGNORE_INDEX] > 0).all()
    padded = sft_data.make_rows(tok, examples, seq_len=256, packing=False, seed=0)
    assert len(padded) == len(sft_data.encode(tok, examples, 256))


def test_pref_pairs_are_ranked_by_verifier(smoke_data: DataConfig) -> None:
    pairs = pref_data.build_pairs(smoke_data, DPOConfig(n_pairs=60), seed=0)
    assert len(pairs) == 60 and {p.kind for p in pairs} >= {"wrong_answer", "no_format"}
    for pair in pairs:
        problem_answer = verify(pair.chosen, 0).parsed
        assert problem_answer is not None
        v_chosen = verify(pair.chosen, problem_answer)
        v_rej = verify(pair.rejected, problem_answer)
        assert v_chosen.correct and v_chosen.format_ok
        assert not (v_rej.correct and v_rej.format_ok), pair


def test_story_pairs(smoke_data: DataConfig) -> None:
    pairs = pref_data.story_pairs(smoke_data, 10, seed=0)
    assert pairs
    for p in pairs:
        subject = p.prompt[0].content.split()[-1].rstrip(".")
        assert stories.mentions(p.chosen, subject) and not stories.mentions(p.rejected, subject)


def test_select_buckets() -> None:
    rates = {"d1s1": 0.95, "d1s2": 0.5, "d2s2": 0.3, "d3s3": 0.0}
    assert rl_prompts.select_buckets(rates, 0.1, 0.6, 4) == ["d2s2", "d1s2"]  # nearest 0.35
    assert rl_prompts.select_buckets({"a": 0.0, "b": 0.9}, 0.1, 0.6, 1) == ["b"]


def test_synthetic_story_and_subject() -> None:
    s = stories.synthetic_story(0, 1, "train")
    assert stories.find_subject(s) is not None
    assert stories.clean("Hi’ there  \n\n\n\nyo", 100) == "Hi' there \n\nyo"
    assert stories.clean("日本", 100) is None
