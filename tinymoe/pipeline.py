"""Pipelines: chain stages from one YAML. Each stage reads its init checkpoint from the run
manifest (``stage.init_from: auto``), so stages can be re-ordered, skipped or re-run.

Schema::

    name: smoke
    groups: {hardware: cpu_smoke, model: smoke, student: student_smoke, data: smoke}
    overrides: {...}            # root-level partial applied to every stage
    stages:
      - name: pretrain
        overrides: {...}        # this stage only

CLI: ``tinymoe pipeline <yaml> [--until STAGE] [group=name ...] [a.b=c ...]
[stages.<name>.a.b=c ...]``. Finished stages (status ``done`` in the manifest) are
skipped, so re-running with the same ``run.run_id`` resumes the pipeline. ``--until``
stops after the named stage; a pause request (``tinymoe.control``) stops it early.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tinymoe import control
from tinymoe.config import (
    ConfigError,
    RunConfig,
    compose,
    deep_merge,
    load_yaml,
    split_overrides,
)
from tinymoe.eval import report
from tinymoe.logging import RunLogger
from tinymoe.stages.common import run_dir_for, run_stage
from tinymoe.train import distributed

log = logging.getLogger(__name__)


def stage_configs(path: Path, args: Sequence[str] = ()) -> list[RunConfig]:
    spec = load_yaml(path)
    unknown = set(spec) - {"name", "groups", "overrides", "stages"}
    if unknown:
        raise ConfigError(f"{path}: unknown pipeline keys {sorted(unknown)}")
    groups, dotted = split_overrides(args)
    per_stage: dict[str, Any] = dotted.pop("stages", {})
    base_groups = {**spec.get("groups", {}), **groups}
    shared = deep_merge(spec.get("overrides", {}), {"run": {"name": spec.get("name", path.stem)}})
    configs = []
    for entry in spec.get("stages", []):
        name = entry["name"]
        cfg = compose(
            {**base_groups, "stage": name},
            shared,
            entry.get("overrides", {}),
            dotted,
            per_stage.get(name, {}),
        )
        configs.append(cfg)
    if not configs:
        raise ConfigError(f"{path}: no stages")
    return configs


def run_pipeline(path: Path, args: Sequence[str] = (), until: str | None = None) -> Path:
    configs = stage_configs(path, args)
    names = [c.stage.name for c in configs]
    if until is not None and until not in names:
        raise ConfigError(f"--until {until!r}: the pipeline's stages are {names}")
    run_dir = run_dir_for(configs[0])
    fixed = [_with_run_id(c, run_dir.name) for c in configs]
    logger = RunLogger(run_dir, enabled=distributed.env_info().is_main)
    if logger.enabled:
        control.clear(run_dir)  # a request left from an earlier process is stale
        logger.update_manifest(
            lambda m: m.update(
                run_id=run_dir.name,
                name=m.get("name", fixed[0].run.name),
                stages=m.get("stages", []),
                pipeline=str(path),
                pipeline_stages=names,
                pipeline_args=list(args),
            )
        )
    argv = [str(path), *args, *(["--until", until] if until else [])]
    with control.running(run_dir, str(path), argv, enabled=logger.enabled):
        finished = _run_stages(fixed, run_dir, logger, until)
    if finished and logger.enabled:
        rep = report.write(run_dir)
        log.info("[pipeline] report:\n%s", report.markdown(rep))
    distributed.shutdown()
    return run_dir


def _run_stages(
    configs: list[RunConfig], run_dir: Path, logger: RunLogger, until: str | None
) -> bool:
    """Run the unfinished stages in order; False if it stopped early (pause or --until)."""
    for cfg in configs:
        name = cfg.stage.name
        done = {
            e["name"] for e in logger.read_manifest().get("stages", []) if e.get("status") == "done"
        }
        if name in done:
            log.info("[pipeline] skipping finished stage %s", name)
        else:
            if control.pending(run_dir) is not None:
                log.info("[pipeline] pause requested: stopping before %s", name)
                control.clear(run_dir)
                return False
            try:
                run_stage(cfg, run_dir)
            except control.StopRequested as stop:
                log.info(
                    "[pipeline] %s; resume by re-running with run.run_id=%s", stop, run_dir.name
                )
                control.clear(run_dir)
                return False
        if name == until:
            log.info("[pipeline] stopping after %s (--until)", name)
            return False
    return True


def _with_run_id(cfg: RunConfig, run_id: str) -> RunConfig:
    cfg.run.run_id = run_id
    return cfg
