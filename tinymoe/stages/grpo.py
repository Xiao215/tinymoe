"""Stage 5 - RL with verifiable rewards: GRPO / Dr. GRPO / DAPO (``grpo.variant``).

1. *Calibration*: estimate the starting policy's pass@1 per difficulty bucket and keep the
   buckets inside ``[low, high]`` (data/rl_prompts.py), so groups have reward variance.
2. *Iterations*: rollouts -> verifier rewards -> group advantages -> clipped policy
   updates (rl/rollout.py, rl/grpo.py), with MoE routing replay and IcePop available.

Step counts in ``train`` (``max_steps``, ``warmup_steps``, ``eval_interval``,
``ckpt_interval``, ``log_interval``) and ``snapshot.interval`` are in *iterations* for
this stage; each iteration is ``ppo_epochs * minibatches_per_epoch`` optimizer steps.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from tinymoe.config import ConfigError, derive_batch
from tinymoe.data import mathworld, rl_prompts
from tinymoe.data.mathworld import Problem
from tinymoe.eval.evaluator import Evaluator
from tinymoe.model.transformer import Transformer
from tinymoe.rl.grpo import policy_loss, resolve_variant
from tinymoe.rl.rollout import RolloutBatch, RolloutProvider
from tinymoe.stages.common import StageContext, stage_main
from tinymoe.train.losses import LossOutput, sequence_logprobs

log = logging.getLogger(__name__)


def calibrate(ctx: StageContext, model: Transformer, evaluator: Evaluator) -> list[str]:
    cfg, g = ctx.cfg, ctx.cfg.grpo
    mw = cfg.data.mathworld
    buckets = mathworld.buckets_from(mw)
    if not g.calibration.enabled:
        return [b.name for b in buckets]
    c = g.calibration
    problems = rl_prompts.rl_problems(mw, buckets, c.samples_per_bucket, cfg.run.seed, "calib")
    flat = [p for ps in problems.values() for p in ps]
    scored = evaluator.sample_problems(
        model, flat, 1, g.temperature, g.max_new_tokens, cfg.run.seed + 17
    )
    rates = {
        b: sum(s[0].correct for s in scored if s[0].problem.bucket == b) / max(len(ps), 1)
        for b, ps in problems.items()
    }
    selected = rl_prompts.select_buckets(rates, c.low, c.high, c.max_buckets)
    ctx.logger.write_artifact(
        "calibration.json",
        {"pass_rates": rates, "band": [c.low, c.high], "selected": selected},
    )
    ctx.logger.log_metrics(ctx.stage, 0, {f"rl/calibration/{b}": r for b, r in rates.items()})
    log.info("[grpo] calibration pass@1 %s -> buckets %s", rates, selected)
    return selected


def frozen_copy(ctx: StageContext) -> Transformer:
    ref = ctx.build_model()
    ref.requires_grad_(False)
    ref.eval()
    return ref


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, t = ctx.cfg, ctx.cfg.train
    g = resolve_variant(cfg.grpo)
    log.info(
        "[grpo] variant=%s std_norm=%s agg=%s clip=(%.2f,%.2f) kl=%.3f dyn=%s overlong=%s "
        "replay=%s icepop=%s",
        g.variant,
        g.std_normalize,
        g.loss_agg,
        g.clip_low,
        g.clip_high,
        g.kl_coef,
        g.dynamic_sampling,
        g.overlong.enabled,
        g.routing_replay,
        g.icepop.enabled,
    )
    model = ctx.build_model()
    ref = frozen_copy(ctx) if g.kl_coef > 0 else None
    evaluator = ctx.evaluator()
    selected = calibrate(ctx, model, evaluator)
    bucket_map = {b.name: b for b in mathworld.buckets_from(cfg.data.mathworld)}
    chosen = [bucket_map[b] for b in selected]

    def problem_at(i: int) -> Problem:
        rng = mathworld.rng_for(cfg.run.seed, "grpo", i)
        return mathworld.make_problem(rng, chosen[i % len(chosen)], cfg.data.mathworld)

    n_rows = g.prompts_per_rollout * g.group_size
    world = ctx.dist.world_size
    if n_rows % g.minibatches_per_epoch or (n_rows // g.minibatches_per_epoch) % world:
        raise ConfigError("prompts_per_rollout * group_size must split into minibatches x ranks")
    rows_per_mb = n_rows // g.minibatches_per_epoch
    mbs = ctx.micro_batch(model, t)
    plan = derive_batch(rows_per_mb, 1, min(mbs, rows_per_mb // world), world)
    updates = g.ppo_epochs * g.minibatches_per_epoch
    steps = replace(
        t,
        max_steps=t.max_steps * updates,
        warmup_steps=t.warmup_steps * updates,
        eval_interval=t.eval_interval * updates,
        ckpt_interval=t.ckpt_interval * updates,
        log_interval=max(t.log_interval, 1) * updates,
    )
    provider = RolloutProvider(
        cfg=g,
        policy=model,
        ref=ref,
        tok=ctx.tok,
        runtime=ctx.runtime,
        evaluator=evaluator,
        problem_at=problem_at,
        plan=plan,
        rank=ctx.dist.rank,
        seed=cfg.run.seed,
        logger=ctx.logger,
        stage=ctx.stage,
        step_offset=0,
        snapshot_every=cfg.snapshot.interval if cfg.snapshot.enabled else 0,
        eval_batch=cfg.eval.batch_size,
    )
    replay = provider.replay

    def loss_fn(m: nn.Module, batch: RolloutBatch) -> LossOutput:
        out = m(batch.input_ids, routing_override=batch.routing if replay else None)
        new_logp = sequence_logprobs(out.logits, batch.labels, g.temperature)
        pl = policy_loss(
            new_logp,
            batch.old_logp,
            batch.ref_logp,
            batch.advantages,
            batch.token_mask,
            batch.row_weight,
            batch.denom,
            batch.scale,
            g,
        )
        with torch.no_grad():
            logp = (out.logits.float() / g.temperature).log_softmax(-1)
            ent = -(logp.exp() * logp).sum(-1)
            n = batch.token_mask.sum().clamp_min(1)
            entropy = (ent * batch.token_mask).sum() / n
        metrics = {
            "loss": pl.loss.detach(),
            "aux_loss": out.aux_loss.detach(),
            "rl/clip_frac": pl.clip_frac,
            "rl/kl": pl.kl,
            "rl/ratio": pl.ratio_mean,
            "rl/entropy": entropy,
            "rl/active_tokens": batch.token_mask.sum(),
        }
        return LossOutput(pl.loss + g.moe_balance_scale * out.aux_loss, metrics)

    hooks, bind = ctx.hooks(model, lambda step: evaluator.run(model, step))
    hooks.snapshot_interval *= updates
    trainer = ctx.trainer(
        model, plan, loss_fn, provider, hooks, train=steps, units_name="sequences"
    )
    bind(trainer)
    log.info(
        "[grpo] %d iterations x %d updates, %d rollouts/iteration, plan %s",
        t.max_steps,
        updates,
        n_rows,
        plan,
    )
    result = trainer.fit()
    summary = {**result.eval_metrics, "rl/buckets": len(selected)}
    return ctx.finish(result.checkpoint, summary)


if __name__ == "__main__":
    stage_main("grpo")
