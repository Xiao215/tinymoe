"""Everything stages share: run directory, tokenizer, lineage, model building, batch plans,
trainer construction, eval/snapshot hooks, and stage bookkeeping in the manifest.

Lineage: a stage's initial weights come from the run manifest (``stage.init_from: auto``
means "the latest finished stage before me"), never from a literal path in a config.
"""

from __future__ import annotations

import copy
import importlib
import logging
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinymoe import hardware
from tinymoe.config import (
    STAGE_ORDER,
    BatchPlan,
    ModelConfig,
    RunConfig,
    TrainConfig,
    derive_batch,
    load_stage_config,
    to_dict,
)
from tinymoe.data.batch import Batch
from tinymoe.data.pretrain_mix import build_shards, build_tokenizer, open_sources
from tinymoe.data.tokenizer import Tokenizer
from tinymoe.eval.evaluator import Evaluator
from tinymoe.eval.snapshots import Snapshotter
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger, new_run_id, setup_logging
from tinymoe.model.config import param_count
from tinymoe.model.transformer import Transformer
from tinymoe.train import checkpoint, distributed
from tinymoe.train.autobatch import model_micro_batch
from tinymoe.train.distributed import DistInfo
from tinymoe.train.loop import Hooks, LossFn, Provider, Trainer, TrainResult
from tinymoe.train.losses import lm_loss
from tinymoe.train.providers import RowProvider, RowSource

log = logging.getLogger(__name__)

ProbeLoss = Callable[[Transformer, Batch], torch.Tensor]


def run_dir_for(cfg: RunConfig) -> Path:
    run_id = cfg.run.run_id or new_run_id(cfg.run.name)
    return Path(cfg.run.out_dir) / run_id


@dataclass
class StageContext:
    cfg: RunConfig
    stage: str
    run_dir: Path
    logger: RunLogger
    runtime: Runtime
    dist: DistInfo
    tok: Tokenizer
    init_ckpt: Path | None

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints" / self.stage

    # ------------------------------------------------------------------ model
    def model_config(self, which: str = "model") -> ModelConfig:
        base: ModelConfig = copy.deepcopy(getattr(self.cfg, which))
        base.vocab_size = base.vocab_size or self.tok.vocab_size
        if base.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"{which}.vocab_size={base.vocab_size} != tokenizer {self.tok.vocab_size}"
            )
        return base

    def build_model(self, which: str = "model", init: Path | str | None = "auto") -> Transformer:
        """Fresh model from config, or the lineage checkpoint (``init='auto'``)."""
        path = self.init_ckpt if init == "auto" else (Path(init) if isinstance(init, str) else init)
        if path is not None:
            model = checkpoint.load_model(path, self.device)
            log.info("[%s] initialised from %s", self.stage, path)
        else:
            hardware.seed_everything(self.cfg.run.seed)
            model = Transformer(self.model_config(which)).to(self.device)
        pc = param_count(model.cfg)
        log.info(
            "[%s] model: %.2fM total / %.2fM active params",
            self.stage,
            pc.total / 1e6,
            pc.active / 1e6,
        )
        return model

    def flops_per_token(self, model: Transformer, seq_len: int) -> float:
        """6 * N_active (fwd+bwd) + attention term 12 * L * d * T (PaLM, Chowdhery et al.)."""
        c = model.cfg
        return 6 * param_count(c).active_non_embedding + 12 * c.n_layers * c.d_model * seq_len

    # ---------------------------------------------------------------- batches
    def micro_batch(
        self,
        model: Transformer,
        train: TrainConfig,
        seq_len: int | None = None,
        probe_loss: ProbeLoss | None = None,
    ) -> int:
        """``train.micro_batch_size`` resolved for ``model`` (autobatch probes ``train``'s
        optimizer and ``probe_loss``; ``seq_len`` defaults to ``train.seq_len``)."""
        return model_micro_batch(
            model,
            self.runtime,
            seq_len or train.seq_len,
            train.micro_batch_size,
            train.auto_batch_max,
            self.cfg.hardware,
            probe_loss,
            train,
        )

    def token_plan(
        self,
        model: Transformer,
        train: TrainConfig | None = None,
        probe_loss: ProbeLoss | None = None,
    ) -> BatchPlan:
        t = train or self.cfg.train
        mbs = self.micro_batch(model, t, probe_loss=probe_loss)
        tokens = t.tokens_per_step or mbs * t.seq_len * self.dist.world_size
        return derive_batch(tokens, t.seq_len, mbs, self.dist.world_size)

    def example_plan(self, model: Transformer, rows_per_example: int = 1) -> BatchPlan:
        t = self.cfg.train
        mbs = self.micro_batch(model, t)
        examples = t.examples_per_step or mbs * self.dist.world_size
        return derive_batch(examples, 1, max(mbs // rows_per_example, 1), self.dist.world_size)

    def shards(self, sources: Sequence[str]) -> dict[str, dict[str, Path]]:
        return build_shards(self.cfg.data, self.tok, sources)

    def val_shards(self, sources: Sequence[str] = ("stories", "math")) -> dict[str, np.ndarray]:
        return open_sources({k: v for k, v in self.shards(sources).items()}, "val")

    # ------------------------------------------------------------ eval/hooks
    def evaluator(self, val_sources: Sequence[str] = ("stories", "math")) -> Evaluator:
        return Evaluator(
            self.cfg, self.tok, self.runtime, self.logger, self.stage, self.val_shards(val_sources)
        )

    def hooks(
        self, model: Transformer, evaluate: Callable[[int], dict[str, float]] | None
    ) -> tuple[Hooks, Callable[[Trainer], None]]:
        """Eval + snapshot hooks; the returned binder attaches the trainer (for MoE load)."""
        holder: dict[str, Trainer] = {}
        snap = Snapshotter(self.cfg, self.tok, self.runtime, self.logger, self.stage)

        def snapshot(step: int) -> None:
            trainer = holder.get("t")
            snap(model, step, trainer.recent_load if trainer else None)

        hooks = Hooks(
            evaluate=evaluate,
            snapshot=snapshot if self.cfg.snapshot.enabled else None,
            snapshot_interval=self.cfg.snapshot.interval,
        )
        return hooks, lambda t: holder.__setitem__("t", t)

    def trainer(
        self,
        model: Transformer,
        plan: BatchPlan,
        loss_fn: LossFn,
        provider: Provider,
        hooks: Hooks | None,
        train: TrainConfig | None = None,
        ckpt_subdir: str | None = None,
        **kwargs: Any,
    ) -> Trainer:
        return Trainer(
            stage=self.stage,
            model=model,
            runtime=self.runtime,
            cfg=train or self.cfg.train,
            plan=plan,
            loss_fn=loss_fn,
            provider=provider,
            logger=self.logger,
            ckpt_dir=self.ckpt_dir / ckpt_subdir if ckpt_subdir else self.ckpt_dir,
            dist=self.dist,
            hooks=hooks,
            **kwargs,
        )

    def fit_lm(
        self,
        model: Transformer,
        source: RowSource,
        source_names: Sequence[str] | None,
        evaluate: Callable[[int], dict[str, float]] | None,
        train: TrainConfig | None = None,
        ckpt_subdir: str | None = None,
        step_offset: int = 0,
        aux_scale: float = 1.0,
        loss_fn: LossFn | None = None,
    ) -> TrainResult:
        """Training on any row source: next-token CE by default (pretrain, midtrain, SFT,
        sequence-level KD), or ``loss_fn`` (logit KD), which then also sizes the batch."""
        t = train or self.cfg.train
        mtp_w = model.cfg.mtp.loss_weight
        probe: ProbeLoss | None = None
        if loss_fn is None:

            def loss_fn(m: torch.nn.Module, batch: Any) -> Any:
                return lm_loss(m, batch, mtp_w, t.z_loss_weight, source_names, aux_scale)

        else:
            custom = loss_fn

            def probe(m: Transformer, batch: Batch) -> torch.Tensor:
                return custom(m, batch).loss

        plan = self.token_plan(model, t, probe)
        provider = RowProvider(source, plan, self.dist.rank, self.device)

        hooks, bind = self.hooks(model, evaluate)
        trainer = self.trainer(
            model,
            plan,
            loss_fn,
            provider,
            hooks,
            train=t,
            ckpt_subdir=ckpt_subdir,
            flops_per_unit=self.flops_per_token(model, t.seq_len),
            step_offset=step_offset,
        )
        bind(trainer)
        log.info("[%s] batch plan %s (%d tokens/step)", self.stage, plan, plan.units_per_step)
        return trainer.fit()

    def finish(self, ckpt: Path | None, summary: Mapping[str, Any]) -> dict[str, Any]:
        self.logger.stage_finished(self.stage, ckpt, summary)
        return dict(summary)


def _resolve_init(cfg: RunConfig, logger: RunLogger) -> Path | None:
    init = cfg.stage.init_from
    if init == "scratch":
        return None
    if init.endswith(".pt"):
        return Path(init)
    if init == "auto":
        prev = logger.last_finished_stage(cfg.stage.name, STAGE_ORDER)
        if prev is None:
            if cfg.stage.name != "pretrain":
                raise RuntimeError(
                    f"stage '{cfg.stage.name}' needs a finished earlier stage in the manifest"
                )
            return None
        init = prev
    path = logger.checkpoint_of(init)
    if path is None:
        raise RuntimeError(f"no finished '{init}' checkpoint in {logger.manifest_path}")
    return path


def _tokenizer(cfg: RunConfig, logger: RunLogger, run_dir: Path) -> Tokenizer:
    rel = logger.read_manifest().get("tokenizer")
    if rel:
        return Tokenizer.load(logger.resolve(rel))
    tok, cached = build_tokenizer(cfg.data)
    target = run_dir / "tokenizer.json"
    if logger.enabled:
        shutil.copy(cached, target)
        logger.set_field("tokenizer", "tokenizer.json")
    return tok


def open_context(cfg: RunConfig, run_dir: Path | None = None) -> StageContext:
    dist = distributed.init(cfg.hardware)
    runtime = hardware.resolve_runtime(cfg.hardware, dist.local_rank)
    run_dir = run_dir or run_dir_for(cfg)
    logger = RunLogger(run_dir, enabled=dist.is_main)
    setup_logging(log_file=run_dir / "log.txt" if dist.is_main else None)
    logger.init_manifest(run_dir.name, cfg.run.name, runtime.caps.to_dict(), runtime.summary())
    distributed.barrier()
    tok = _tokenizer(cfg, logger, run_dir)
    distributed.barrier()
    init = _resolve_init(cfg, logger)
    hardware.seed_everything(cfg.run.seed)
    stage = cfg.stage.name
    logger.stage_started(stage, to_dict(cfg), str(init) if init else None)
    return StageContext(cfg, stage, run_dir, logger, runtime, dist, tok, init)


def run_stage(cfg: RunConfig, run_dir: Path | None = None) -> dict[str, Any]:
    """Run one stage end-to-end and record it in the manifest."""
    ctx = open_context(cfg, run_dir)
    module = importlib.import_module(f"tinymoe.stages.{ctx.stage}")
    try:
        summary: dict[str, Any] = module.run(ctx)
    except Exception as exc:
        ctx.logger.stage_failed(ctx.stage, repr(exc))
        raise
    log.info("[%s] done -> %s", ctx.stage, ctx.run_dir)
    return summary


def stage_main(name: str, argv: Sequence[str] | None = None) -> None:
    """``python -m tinymoe.stages.<name> [group=..] [a.b=c ..]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    cfg = load_stage_config([f"stage={name}", *args])
    run_stage(cfg)
    distributed.shutdown()


def sub_train(base: TrainConfig, **changes: Any) -> TrainConfig:
    return replace(base, **changes)
