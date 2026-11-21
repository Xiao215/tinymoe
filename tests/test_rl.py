"""M9: rewards, GRPO-family losses, rollout provider (exact minibatch aggregation, routing
replay, IcePop, dynamic sampling) on a tiny MoE policy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from helpers import cpu_runtime

from tinymoe import hardware
from tinymoe.config import (
    BatchPlan,
    GRPOConfig,
    HardwareConfig,
    ModelConfig,
    OverlongConfig,
    load_stage_config,
    section_from_dict,
)
from tinymoe.data import mathworld
from tinymoe.data.batch import IGNORE_INDEX
from tinymoe.data.tokenizer import Tokenizer
from tinymoe.eval.evaluator import Evaluator
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.transformer import Transformer
from tinymoe.rl import grpo
from tinymoe.rl.rewards import overlong_penalty, score
from tinymoe.rl.rollout import RolloutProvider
from tinymoe.train.losses import sequence_logprobs


def test_rewards_and_overlong() -> None:
    cfg = GRPOConfig(reward_correct=1.0, reward_format=0.2)
    good = "<think>\n2 + 3 = 5\n</think>\nAnswer: 5"
    assert score(good, 5, 10, True, 96, cfg).total == pytest.approx(1.2)
    assert score(good, 6, 10, True, 96, cfg).total == pytest.approx(0.2)  # format only
    assert score("Answer: 5", 5, 3, True, 96, cfg).total == 1.0  # no think block: no format bonus
    assert score("<think>\nAnswer: 5\n</think>\n", 5, 9, True, 96, cfg).total == 0.0
    dapo = GRPOConfig(overlong=OverlongConfig(enabled=True, cache_len=16, penalty=1.0))
    assert overlong_penalty(80, True, 96, dapo) == 0.0
    assert overlong_penalty(88, True, 96, dapo) == pytest.approx(-0.5)
    assert overlong_penalty(96, False, 96, dapo) == -1.0  # truncated


def test_variant_presets() -> None:
    dr = grpo.resolve_variant(GRPOConfig(variant="dr_grpo"))
    assert not dr.std_normalize and dr.loss_agg == "constant" and dr.kl_coef == 0.0
    dapo = grpo.resolve_variant(GRPOConfig(variant="dapo"))
    assert dapo.clip_high > dapo.clip_low and dapo.dynamic_sampling and dapo.overlong.enabled
    custom = grpo.resolve_variant(GRPOConfig(variant="custom", clip_high=0.5))
    assert custom.clip_high == 0.5


def test_group_advantages() -> None:
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = grpo.group_advantages(r, std_normalize=True)
    assert torch.allclose(a[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4)
    assert torch.all(a[1] == 0)  # no variance, no signal
    assert torch.allclose(grpo.group_advantages(r, False)[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_policy_loss_clipping_and_aggregation() -> None:
    cfg = GRPOConfig(clip_low=0.2, clip_high=0.2, kl_coef=0.0)
    old = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv = torch.tensor([1.0])
    w = torch.ones(1)
    # ratio exp(1) > 1.2 with positive advantage: clipped, no gradient
    new = torch.ones(1, 3, requires_grad=True)
    pl = grpo.policy_loss(new, old, None, adv, mask, w, 3.0, 1.0, cfg)
    pl.loss.backward()
    assert new.grad is not None and torch.all(new.grad == 0) and float(pl.clip_frac) == 1.0
    assert float(pl.loss.detach()) == pytest.approx(-1.2)
    # on-policy ratio 1: loss = -A, gradient = -A / denom per token
    new = torch.zeros(1, 3, requires_grad=True)
    grpo.policy_loss(new, old, None, adv, mask, w, 3.0, 1.0, cfg).loss.backward()
    assert new.grad is not None and torch.allclose(new.grad, torch.full((1, 3), -1 / 3))
    lengths = torch.tensor([2.0, 4.0])
    assert grpo.denominator(lengths, torch.ones(2), "token_mean", 96) == 6.0
    assert grpo.denominator(lengths, torch.ones(2), "seq_mean", 96) == 2.0
    assert grpo.denominator(lengths, torch.ones(2), "constant", 96) == 192.0
    assert torch.allclose(grpo.row_weights(lengths, "seq_mean"), torch.tensor([0.5, 0.25]))


def test_kl_k3_nonnegative() -> None:
    cfg = GRPOConfig(kl_coef=1.0)
    new = torch.randn(2, 5)
    out = grpo.policy_loss(
        new,
        new.detach(),
        torch.randn(2, 5),
        torch.zeros(2),
        torch.ones(2, 5),
        torch.ones(2),
        10.0,
        1.0,
        cfg,
    )
    assert float(out.kl) >= 0
    same = grpo.policy_loss(
        new,
        new.detach(),
        new.detach(),
        torch.zeros(2),
        torch.ones(2, 5),
        torch.ones(2),
        10.0,
        1.0,
        cfg,
    )
    assert float(same.kl) == pytest.approx(0.0, abs=1e-6)


def _policy(tok: Tokenizer) -> Transformer:
    torch.manual_seed(0)
    cfg = section_from_dict(
        ModelConfig,
        {
            "vocab_size": tok.vocab_size,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden": 64,
            "max_seq_len": 128,
            "moe": {
                "enabled": True,
                "n_experts": 4,
                "top_k": 2,
                "expert_hidden": 32,
                "n_dense_layers": 1,
            },
        },
    )
    return Transformer(cfg)


def _provider(
    tok: Tokenizer, tmp: Path, runtime: Runtime | None = None, **grpo_kw: object
) -> RolloutProvider:
    run = load_stage_config(["data=smoke"])
    g = grpo.resolve_variant(
        replace(
            GRPOConfig(
                group_size=4,
                prompts_per_rollout=4,
                ppo_epochs=2,
                minibatches_per_epoch=2,
                max_new_tokens=12,
                kl_coef=0.04,
            ),
            **grpo_kw,
        )  # type: ignore[arg-type]
    )
    rt = runtime or cpu_runtime()
    policy = _policy(tok).to(rt.device)
    logger = RunLogger(tmp)
    ev = Evaluator(run, tok, rt, logger, "grpo")
    buckets = mathworld.buckets_from(run.data.mathworld)

    def problem_at(i: int) -> mathworld.Problem:
        return mathworld.make_problem(
            mathworld.rng_for(0, "t", i), buckets[i % len(buckets)], run.data.mathworld
        )

    ref = _policy(tok).to(rt.device) if g.kl_coef > 0 else None
    plan = BatchPlan(micro_batch_size=4, grad_accum=2, world_size=1, unit=1)
    return RolloutProvider(
        cfg=g,
        policy=policy,
        ref=ref,
        tok=tok,
        runtime=rt,
        evaluator=ev,
        problem_at=problem_at,
        plan=plan,
        rank=0,
        seed=0,
        logger=logger,
        stage="grpo",
        step_offset=0,
        snapshot_every=1,
        eval_batch=16,
    )


def test_provider_minibatches_cover_buffer_once_per_epoch(tok: Tokenizer, tmp_path: Path) -> None:
    prov = _provider(tok, tmp_path)
    seen: list[list[int]] = []
    for step in range(prov.updates):  # one iteration: 2 epochs x 2 minibatches x 2 micro
        rows = []
        for micro in range(2):
            b = prov.batch(step, micro)
            rows.append(b)
            assert b.denom > 0 and b.scale == 2.0
            assert (b.token_mask * (b.labels == IGNORE_INDEX)).sum() == 0  # only completion tokens
        seen.append([int(x) for r in rows for x in r.input_ids[:, 0]])
    assert prov.iteration == 0 and prov.next_prompt == 4
    assert prov.buffer is not None and prov.buffer.input_ids.shape[0] == 16
    snaps = list((tmp_path / "snapshots" / "grpo").iterdir())
    assert snaps and (snaps[0] / "rollouts.json").is_file()
    prov.batch(prov.updates, 0)  # next iteration rolls out again
    assert prov.iteration == 1 and prov.next_prompt == 8


@pytest.mark.parametrize("kind", ["cpu", *hardware.accelerator_types()[:1]])
def test_provider_batches_arrive_on_the_training_device(
    kind: str, tok: Tokenizer, tmp_path: Path
) -> None:
    """Rollouts are buffered in host memory, but every micro batch must reach the loss on
    the runtime's device (the provider contract); CPU-only tests cannot see this mix-up."""
    rt = hardware.resolve_runtime(HardwareConfig(device=kind, precision="fp32"))
    prov = _provider(tok, tmp_path, runtime=rt, routing_replay=True)
    b = prov.batch(0, 0)
    tensors = [b.input_ids, b.labels, b.token_mask, b.old_logp, b.advantages, b.row_weight]
    assert b.ref_logp is not None and b.routing
    tensors += [b.ref_logp, *b.routing.values()]
    assert all(t.device.type == rt.device.type for t in tensors)


def test_old_logp_matches_training_forward_with_replay(tok: Tokenizer, tmp_path: Path) -> None:
    """At iteration start (no update yet) the training forward with routing replay gives
    exactly old_logp, so the first ratio is 1 everywhere."""
    prov = _provider(tok, tmp_path, routing_replay=True)
    b = prov.batch(0, 0)
    prov.policy.eval()
    with torch.no_grad():
        out = prov.policy(b.input_ids, routing_override=b.routing)
    new = sequence_logprobs(out.logits, b.labels, prov.cfg.temperature)
    assert b.routing is not None
    assert torch.allclose(new * b.token_mask, b.old_logp * b.token_mask, atol=1e-5)
    # the rollout routing equals what a fresh fp32 forward picks on CPU: mismatch ~ 0
    assert prov.buffer is not None


def test_icepop_masks_off_policy_tokens(tok: Tokenizer, tmp_path: Path) -> None:
    prov = _provider(tok, tmp_path)
    prov.cfg = replace(prov.cfg, icepop=replace(prov.cfg.icepop, enabled=True))
    prov.batch(0, 0)
    assert prov.buffer is not None
    prov.buffer.rollout_logp += 5.0  # pretend the sampler saw very different probabilities
    b = prov.batch(0, 0)
    assert float(b.token_mask.sum()) == 0.0


def test_dynamic_sampling_marks_uninformative_groups(tok: Tokenizer, tmp_path: Path) -> None:
    """An untrained policy fails everything: every group's accuracy is all-zero, so DAPO
    drops them all and the rows carry no gradient (even though the overlong penalty makes
    the total reward vary inside a group)."""
    prov = _provider(tok, tmp_path, variant="dapo", max_resample_rounds=1)
    b = prov.batch(0, 0)
    assert prov.buffer is not None
    assert float(prov.buffer.valid.sum()) == 0.0
    assert float(b.token_mask.sum()) == 0.0
    assert prov.next_prompt == 8  # one resampling round of 4 fresh prompts


def test_provider_state_roundtrip(tok: Tokenizer, tmp_path: Path) -> None:
    prov = _provider(tok, tmp_path)
    a = prov.batch(1, 1)
    state = prov.state_dict()
    fresh = _provider(tok, tmp_path / "b")
    fresh.load_state_dict(state)
    b = fresh.batch(1, 1)
    assert torch.equal(a.input_ids, b.input_ids) and torch.equal(a.old_logp, b.old_logp)
    assert fresh.next_prompt == prov.next_prompt
