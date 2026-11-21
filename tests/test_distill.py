"""M10: distillation losses, on-policy rollouts, prompt mix and the comparison table."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from helpers import V, cpu_runtime, tiny_model

from tinylab.config import BatchPlan, load_stage_config
from tinylab.data.batch import IGNORE_INDEX, Batch
from tinylab.data.tokenizer import Tokenizer
from tinylab.eval.evaluator import Evaluator
from tinylab.logging import RunLogger
from tinylab.rl.rollout import pad_completions
from tinylab.stages.distill import (
    STORY_EVERY,
    ModeResult,
    OnPolicyRollouts,
    TeacherSample,
    best,
    comparison,
    distill_prompts,
    logit_kd_loss,
    reverse_kl_loss,
    select_teacher_samples,
)
from tinylab.train.losses import token_kl


def test_token_kl_matches_definition() -> None:
    torch.manual_seed(0)
    p, q = torch.randn(3, 5, 11), torch.randn(3, 5, 11)
    ref = F.kl_div(q.log_softmax(-1), p.log_softmax(-1), log_target=True, reduction="none")
    torch.testing.assert_close(token_kl(p, q), ref.sum(-1))
    assert torch.all(token_kl(p, q) >= -1e-6)
    torch.testing.assert_close(token_kl(p, p), torch.zeros(3, 5), atol=1e-6, rtol=0)
    # temperature flattens both distributions: divergence shrinks
    assert float(token_kl(p, q, 4.0).mean()) < float(token_kl(p, q).mean())


def _batch(seed: int) -> Batch:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (2, 9), generator=g)
    labels = x[:, 1:].clone()
    labels[:, :3] = IGNORE_INDEX  # prompt positions
    return Batch(x[:, :-1], labels)


@pytest.mark.parametrize("make", ["logit", "reverse"])
def test_kd_losses_vanish_when_student_equals_teacher(make: str) -> None:
    teacher = tiny_model(False, seed=1)
    teacher.requires_grad_(False)
    student = tiny_model(False, seed=1)
    loss_fn = logit_kd_loss(teacher, 2.0, 0.0) if make == "logit" else reverse_kl_loss(teacher)
    out = loss_fn(student, _batch(0))
    assert float(out.loss.detach()) == pytest.approx(0.0, abs=1e-5)
    out.loss.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads) < 1e-5
    other = tiny_model(False, seed=2)
    assert float(loss_fn(other, _batch(0)).loss.detach()) > 1e-3


def test_logit_kd_tau_squared_and_ce_mix() -> None:
    teacher = tiny_model(False, seed=1)
    student = tiny_model(False, seed=2)
    b = _batch(1)
    plain = logit_kd_loss(teacher, 1.0, 0.0)(student, b)
    mixed = logit_kd_loss(teacher, 1.0, 0.5)(student, b)
    assert float(mixed.loss.detach()) == pytest.approx(
        float(plain.metrics["kd/kl"]) + 0.5 * float(plain.metrics["kd/ce"]), rel=1e-5
    )
    hot = logit_kd_loss(teacher, 3.0, 0.0)(student, b)
    with torch.no_grad():
        s, t = student(b.input_ids).logits, teacher(b.input_ids).logits
    mask = (b.labels != IGNORE_INDEX).float()
    expect = 9.0 * float((token_kl(t, s, 3.0) * mask).sum() / mask.sum())
    assert float(hot.metrics["kd/kl"]) == pytest.approx(expect, rel=1e-5)


def test_prompt_mix(tok: Tokenizer) -> None:
    cfg = load_stage_config(["data=smoke"])
    prompts = distill_prompts(cfg, tok, 12, "t")
    kinds = [p.kind for p in prompts]
    assert kinds.count("story") == 12 // STORY_EVERY and kinds[STORY_EVERY - 1] == "story"
    assert all(p.answer is not None for p in prompts if p.kind == "math")
    again = distill_prompts(cfg, tok, 12, "t")
    assert [p.ids for p in again] == [p.ids for p in prompts]
    assert [p.ids for p in distill_prompts(cfg, tok, 12, "u")] != [p.ids for p in prompts]


def test_onpolicy_rollouts_split_and_regenerate(tok: Tokenizer, tmp_path: Path) -> None:
    cfg = load_stage_config(["data=smoke", "eval.batch_size=8"])
    rt = cpu_runtime()
    ev = Evaluator(cfg, tok, rt, RunLogger(tmp_path, enabled=False), "distill")
    student = tiny_model(False, seed=0, vocab_size=tok.vocab_size, max_seq_len=128)
    prompts = distill_prompts(cfg, tok, 16, "t")
    plan = BatchPlan(micro_batch_size=2, grad_accum=2, world_size=1, unit=1)
    prov = OnPolicyRollouts(
        student, ev, prompts, plan, 0, 0, 8, 1.0, tok.pad_id, torch.device("cpu")
    )
    a0, a1 = prov.batch(0, 0), prov.batch(0, 1)
    assert a0.input_ids.shape[0] == 2 and a1.input_ids.shape[0] == 2
    for b, i in ((a0, 0), (a1, 2)):  # rows follow the prompt order, labels on the completion
        p = prompts[i].ids
        assert b.input_ids[0, : len(p)].tolist() == p
        assert torch.all(b.labels[0, : len(p) - 1] == IGNORE_INDEX)
        assert int((b.labels[0] != IGNORE_INDEX).sum()) >= 1
    assert prov.batch(0, 0).input_ids.equal(a0.input_ids)  # same step: cached rollout
    nxt = prov.batch(1, 0)
    assert nxt.input_ids[0, : len(prompts[4].ids)].tolist() == prompts[4].ids
    fresh = OnPolicyRollouts(
        student, ev, prompts, plan, 0, 0, 8, 1.0, tok.pad_id, torch.device("cpu")
    )
    assert fresh.batch(0, 0).input_ids.equal(a0.input_ids)  # stateless: same weights, same rollout
    assert student.training  # sampling restores train mode


def test_pad_completions_layout() -> None:
    ids, labels = pad_completions([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], 0)
    assert ids.tolist() == [[1, 2, 3, 5], [4, 7, 8, 0]]
    assert labels.tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 5, 6],
        [7, 8, 9, IGNORE_INDEX],
    ]


def test_comparison_table_and_best() -> None:
    def row(mode: str, acc: float, loss: float) -> ModeResult:
        metrics = {"eval/math/pass1": acc, "val/loss/sft": loss}
        return ModeResult(mode, "student base", "d", "o", (0, 1), metrics, None)

    rows = [row("baseline", 0.2, 1.0), row("offpolicy", 0.4, 1.2), row("logit", 0.4, 1.1)]
    assert best(rows).mode == "logit"  # tie on pass@1 -> lower held-out loss
    table = comparison({"distill/teacher/eval/math/pass1": 0.6}, rows)
    assert table["teacher"] == {"eval/math/pass1": 0.6}
    lines = table["markdown"].splitlines()
    assert len(lines) == 2 + 1 + len(rows) and lines[2].startswith("| teacher")
    assert "0.400" in lines[4] and lines[3].count("|") == lines[0].count("|")


def test_teacher_samples_rejection_and_fallback() -> None:
    """Keep finished, verified, unique samples; with none, fall back to all non-empty ones."""
    good = TeacherSample([1, 2], [3, 4], "math", True)
    samples = [
        good,
        TeacherSample([1, 2], [3, 4], "math", True),  # duplicate
        TeacherSample([1, 2], [5], "math", False),  # wrong or unfinished
        TeacherSample([7], [8, 9], "story", True),
        TeacherSample([7], [], "story", False),  # empty
    ]
    kept, filtered = select_teacher_samples(samples)
    assert filtered
    assert [(s["prompt"], s["completion"]) for s in kept] == [([1, 2], [3, 4]), ([7], [8, 9])]
    weak = [TeacherSample([1], [2], "math", False), TeacherSample([1], [], "math", False)]
    kept, filtered = select_teacher_samples(weak)
    assert not filtered and kept == [{"prompt": [1], "completion": [2], "kind": "math"}]
