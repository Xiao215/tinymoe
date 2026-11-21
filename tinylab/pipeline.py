"""Pipelines: chain stages from one YAML. Each stage reads its init checkpoint from the run
manifest (``stage.init_from: auto``), so stages can be re-ordered, skipped or re-run.

Schema::

    name: smoke
    groups: {hardware: cpu_smoke, model: smoke, student: student_smoke, data: smoke}
    overrides: {...}            # root-level partial applied to every stage
    stages:
      - name: pretrain
        overrides: {...}        # this stage only

CLI: ``tinylab pipeline <yaml> [group=name ...] [a.b=c ...] [stages.<name>.a.b=c ...]``.
Finished stages (status ``done`` in the manifest) are skipped, so re-running with the same
``run.run_id`` resumes the pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tinylab.config import (
    ConfigError,
    RunConfig,
    compose,
    deep_merge,
    load_yaml,
    split_overrides,
)
from tinylab.eval import report
from tinylab.logging import RunLogger
from tinylab.stages.common import run_dir_for, run_stage
from tinylab.train import distributed

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


def run_pipeline(path: Path, args: Sequence[str] = ()) -> Path:
    configs = stage_configs(path, args)
    run_dir = run_dir_for(configs[0])
    fixed = [_with_run_id(c, run_dir.name) for c in configs]
    logger = RunLogger(run_dir, enabled=distributed.env_info().is_main)
    for cfg in fixed:
        done = {
            e["name"] for e in logger.read_manifest().get("stages", []) if e.get("status") == "done"
        }
        if cfg.stage.name in done:
            log.info("[pipeline] skipping finished stage %s", cfg.stage.name)
            continue
        run_stage(cfg, run_dir)
    if logger.enabled:
        rep = report.write(run_dir)
        log.info("[pipeline] report:\n%s", report.markdown(rep))
    distributed.shutdown()
    return run_dir


def _with_run_id(cfg: RunConfig, run_id: str) -> RunConfig:
    cfg.run.run_id = run_id
    return cfg
