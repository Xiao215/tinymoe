"""The one training loop every stage uses.

A stage supplies a *provider* (``batch(step, micro)`` -> anything its loss understands)
and a *loss function* ``(model, batch) -> LossOutput``; the trainer owns everything else:
gradient accumulation (with DDP ``no_sync``), autocast + dynamic loss scaling, clipping,
LR schedule, MoE router-bias updates, metrics, eval/snapshot hooks and checkpoints with
bit-identical resume.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from tinymoe import hardware
from tinymoe.config import BatchPlan, TrainConfig
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.transformer import Transformer
from tinymoe.train import checkpoint, distributed
from tinymoe.train.distributed import DistInfo
from tinymoe.train.losses import COUNT_PREFIX, LossOutput
from tinymoe.train.optim import Optimizers, build_optimizer
from tinymoe.train.schedule import lr_multiplier

log = logging.getLogger(__name__)


class Provider(Protocol):
    """Deterministic source of micro-batches, addressed by (step, micro)."""

    def batch(self, step: int, micro: int) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


LossFn = Callable[[nn.Module, Any], LossOutput]


@dataclass
class Hooks:
    evaluate: Callable[[int], dict[str, float]] | None = None
    snapshot: Callable[[int], None] | None = None
    snapshot_interval: int = 0
    eval_at_start: bool = False


class LossScaler:
    """Dynamic loss scaling for fp16 (Micikevicius et al., 2018), device-agnostic.

    Non-finite gradients -> skip the step and halve the scale; ``growth_interval`` clean
    steps in a row -> double it. Disabled it is the identity.
    """

    def __init__(
        self, enabled: bool, init_scale: float = 2.0**15, growth_interval: int = 500
    ) -> None:
        self.enabled = enabled
        self.scale = init_scale if enabled else 1.0
        self.growth_interval = growth_interval
        self.good_steps = 0

    def update(self, finite: bool) -> bool:
        """Returns True if the optimizer should step."""
        if not self.enabled:
            return finite
        if not finite:
            self.scale = max(self.scale / 2, 1.0)
            self.good_steps = 0
            return False
        self.good_steps += 1
        if self.good_steps % self.growth_interval == 0:
            self.scale *= 2
        return True

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "good_steps": self.good_steps}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])
        self.good_steps = int(state["good_steps"])


@dataclass
class TrainResult:
    step: int
    checkpoint: Path | None
    metrics: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    step_times: list[float] = field(default_factory=list)


def _backward(loss: torch.Tensor) -> None:
    torch.autograd.backward(loss)


class Trainer:
    def __init__(
        self,
        *,
        stage: str,
        model: Transformer,
        runtime: Runtime,
        cfg: TrainConfig,
        plan: BatchPlan,
        loss_fn: LossFn,
        provider: Provider,
        logger: RunLogger,
        ckpt_dir: Path,
        dist: DistInfo,
        hooks: Hooks | None = None,
        lr: float | None = None,
        units_name: str = "tokens",
        flops_per_unit: float | None = None,
        save_final: bool = True,
        step_offset: int = 0,
    ) -> None:
        self.stage = stage
        self.raw_model = model
        self.runtime = runtime
        self.cfg = cfg
        self.plan = plan
        self.loss_fn = loss_fn
        self.provider = provider
        self.logger = logger
        self.ckpt_dir = ckpt_dir
        self.dist = dist
        self.hooks = hooks or Hooks()
        self.units_name = units_name
        self.flops_per_unit = flops_per_unit
        self.save_final = save_final
        self.step_offset = step_offset  # logged step = offset + local step (multi-phase stages)
        self.optim: Optimizers = build_optimizer(model, cfg, runtime, lr)
        self.scaler = LossScaler(runtime.use_loss_scaling)
        self.params = [p for p in model.parameters() if p.requires_grad]
        wrapped = distributed.wrap(model, runtime.device)
        self.model: nn.Module = wrapped
        if runtime.compile:
            compiled: Any = torch.compile(wrapped, backend=runtime.compile_backend)
            self.model = compiled
        self.start_step = 0
        self.units_seen = 0
        self.recent_load: torch.Tensor | None = None  # MoE counts since the last snapshot

    # ----------------------------------------------------------------- resume
    def state(self, step: int) -> dict[str, Any]:
        return {
            **checkpoint.model_payload(self.raw_model),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "provider": self.provider.state_dict(),
            "rng": hardware.rng_state(),
            "step": step,
            "units_seen": self.units_seen,
            "stage": self.stage,
        }

    def maybe_resume(self) -> None:
        if self.cfg.resume == "never":
            return
        path = checkpoint.latest(self.ckpt_dir)
        if path is None:
            if self.cfg.resume == "must":
                raise FileNotFoundError(f"resume=must but no checkpoint in {self.ckpt_dir}")
            return
        data = checkpoint.load(path)
        self.raw_model.load_state_dict(data["model"])
        self.optim.load_state_dict(data["optim"])
        self.scaler.load_state_dict(data["scaler"])
        self.provider.load_state_dict(data["provider"])
        hardware.set_rng_state(data["rng"])
        self.start_step = int(data["step"])
        self.units_seen = int(data["units_seen"])
        log.info("[%s] resumed from %s (step %d)", self.stage, path, self.start_step)

    def save_checkpoint(self, step: int) -> None:
        if self.dist.is_main:
            checkpoint.save(self.ckpt_dir / f"step_{step:08d}.pt", self.state(step))
            checkpoint.prune(self.ckpt_dir, self.cfg.keep_ckpts)
        distributed.barrier()

    # ------------------------------------------------------------------- step
    def _micro_context(self, micro: int) -> contextlib.AbstractContextManager[Any]:
        last = micro == self.plan.grad_accum - 1
        no_sync = getattr(self.model, "no_sync", None)
        if not last and no_sync is not None:
            ctx: contextlib.AbstractContextManager[Any] = no_sync()
            return ctx
        return contextlib.nullcontext()

    def _accumulate(self, step: int) -> dict[str, torch.Tensor]:
        sums: dict[str, torch.Tensor] = {}
        accum = self.plan.grad_accum
        for micro in range(accum):
            batch = self.provider.batch(step, micro)
            with self._micro_context(micro):
                with self.runtime.autocast(), self.runtime.sdpa_context():
                    out = self.loss_fn(self.model, batch)
                _backward(out.loss * (self.scaler.scale / accum))
            for k, v in out.metrics.items():
                v = v.detach().float()
                sums[k] = sums[k] + v if k in sums else v
        result: dict[str, torch.Tensor] = {}
        for k, v in sums.items():
            if k.startswith(COUNT_PREFIX):
                continue
            count = sums.get(COUNT_PREFIX + k)
            result[k] = v / count.clamp_min(1) if count is not None else v / accum
        return result

    def _grad_norms(self, per_group: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scaler.enabled:
            inv = 1.0 / self.scaler.scale
            for p in self.params:
                if p.grad is not None:
                    p.grad.mul_(inv)
        groups = {}
        for i, g in enumerate(self.optim.param_groups if per_group else []):
            grads = [p.grad for p in g["params"] if p.grad is not None]
            if grads:
                groups[f"g{i}"] = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(x.float()) for x in grads])
                )
        clip = self.cfg.grad_clip if self.cfg.grad_clip > 0 else math.inf
        total = torch.nn.utils.clip_grad_norm_(self.params, clip)
        return total, groups

    def _update_ratio_probe(self) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Sample <= 4 tensors per group to estimate ||dW|| / ||W|| (cheap on log steps)."""
        probe = []
        for i, g in enumerate(self.optim.param_groups):
            for p in g["params"][:4]:
                probe.append((i, p, p.detach().clone()))
        return probe

    def _moe_step(self, with_stats: bool) -> dict[str, float]:
        """Aux-free bias update every step (no host sync); load statistics on log steps."""
        counts = self.raw_model.load_counts()
        if counts is None:
            return {}
        distributed.all_reduce_sum_(counts)
        self.raw_model.update_router_bias(counts)
        self.recent_load = counts if self.recent_load is None else self.recent_load + counts
        if not with_stats:
            return {}
        frac = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        n_exp = counts.shape[-1]
        ent = -(frac * frac.clamp_min(1e-12).log()).sum(-1) / math.log(n_exp)
        biases = torch.stack([m.expert_bias for m in self.raw_model.moe_layers()])
        return {
            "moe/load_max": float(frac.max(-1).values.mean() * n_exp),
            "moe/load_entropy": float(ent.mean()),
            "moe/dead_experts": float((counts == 0).sum()),
            "moe/bias_absmax": float(biases.abs().max()),
        }

    # -------------------------------------------------------------------- fit
    def fit(self) -> TrainResult:
        self.maybe_resume()
        self.raw_model.train()
        result = TrainResult(step=self.start_step, checkpoint=None)
        if self.hooks.eval_at_start and self.start_step == 0 and self.hooks.evaluate:
            result.eval_metrics = self._evaluate(self.step_offset)
        t_last = time.perf_counter()
        units_at_last = self.units_seen
        for step in range(self.start_step, self.cfg.max_steps):
            t0 = time.perf_counter()
            mult = lr_multiplier(step, self.cfg)
            self.optim.set_lr_multiplier(mult)
            is_log = (step + 1) % self.cfg.log_interval == 0 or step + 1 == self.cfg.max_steps
            metrics = self._accumulate(step)
            grad_norm, group_norms = self._grad_norms(per_group=is_log)
            finite = bool(torch.isfinite(grad_norm))
            probe = self._update_ratio_probe() if is_log and finite else []
            stepped = self.scaler.update(finite)
            if stepped:
                self.optim.step()
            self.optim.zero_grad()
            moe_stats = self._moe_step(with_stats=is_log)
            self.units_seen += self.plan.units_per_step
            result.step = step + 1
            result.step_times.append(time.perf_counter() - t0)
            if is_log:
                hardware.synchronize(self.runtime.device)
                now = time.perf_counter()
                rate = (self.units_seen - units_at_last) / max(now - t_last, 1e-9)
                t_last, units_at_last = now, self.units_seen
                values = torch.stack(list(metrics.values())) if metrics else torch.zeros(0)
                distributed.all_reduce_mean_(values)
                logged = {k: float(v) for k, v in zip(metrics, values.tolist(), strict=True)}
                logged.update(moe_stats)
                logged.update(
                    {
                        "lr": self.optim.param_groups[0]["base_lr"] * mult,
                        "lr_mult": mult,
                        "grad_norm": float(grad_norm),
                        "skipped": float(not stepped),
                        self.units_name: float(self.units_seen),
                        f"{self.units_name}_per_s": rate,
                        "step_time": result.step_times[-1],
                        "mem_gb": hardware.memory_bytes(self.runtime.device)["peak"] / 1e9,
                    }
                )
                if self.scaler.enabled:
                    logged["loss_scale"] = self.scaler.scale
                for name, norm in group_norms.items():
                    logged[f"grad_norm/{name}"] = float(norm)
                for i, p, before in probe:
                    ratio = float((p.detach() - before).norm() / before.norm().clamp_min(1e-12))
                    key = f"update_ratio/g{i}"
                    logged[key] = max(logged.get(key, 0.0), ratio)
                mfu = self._mfu(rate)
                if mfu is not None:
                    logged["mfu"] = mfu
                self.logger.log_metrics(self.stage, self.step_offset + step + 1, logged)
                result.metrics = logged
                log.info(
                    "[%s] step %d/%d loss %.4f lr %.2e gnorm %.2f %s/s %.0f",
                    self.stage,
                    step + 1,
                    self.cfg.max_steps,
                    logged.get("loss", float("nan")),
                    logged["lr"],
                    logged["grad_norm"],
                    self.units_name,
                    rate,
                )
            done = step + 1 == self.cfg.max_steps
            if (
                self.hooks.snapshot
                and self.hooks.snapshot_interval
                and ((step + 1) % self.hooks.snapshot_interval == 0 or done)
            ):
                if self.dist.is_main:
                    self.hooks.snapshot(self.step_offset + step + 1)
                self.recent_load = None
                distributed.barrier()
            if self.hooks.evaluate and ((step + 1) % self.cfg.eval_interval == 0 or done):
                result.eval_metrics = self._evaluate(self.step_offset + step + 1)
            if (step + 1) % self.cfg.ckpt_interval == 0 and not done:
                self.save_checkpoint(step + 1)
        if self.save_final:
            result.checkpoint = self.ckpt_dir / "final.pt"
            if self.dist.is_main:
                checkpoint.save_model(result.checkpoint, self.raw_model, {"step": result.step})
            distributed.barrier()
        return result

    def _evaluate(self, step: int) -> dict[str, float]:
        assert self.hooks.evaluate is not None
        metrics: dict[str, float] = {}
        if self.dist.is_main:
            with hardware.preserved_rng():
                metrics = self.hooks.evaluate(step)
            self.logger.log_metrics(self.stage, step, metrics)
        self.raw_model.train()
        distributed.barrier()
        return metrics

    def _mfu(self, units_per_s: float) -> float | None:
        if self.flops_per_unit is None:
            return None
        peak = self.runtime.peak_tflops or self.runtime.measured_tflops
        if not peak:
            return None
        world = max(self.dist.world_size, 1)
        return units_per_s * self.flops_per_unit / (peak * 1e12 * world)
