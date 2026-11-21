"""Run logging: manifest, metrics JSONL, snapshots and artifacts under ``runs/<run_id>/``.

The viewer reads exactly these files, so the layout is a contract (``SCHEMA_VERSION``):

* ``manifest.json``   config, git hash, hardware, stage lineage, checkpoint paths
* ``metrics.jsonl``   one JSON object per (stage, step) with scalar metrics
* ``snapshots/<stage>/<step:08d>/<name>.json`` (+ ``<name>.npz``)  data, generations
* ``artifacts/``      reports, tables, calibration results

All writes are atomic (tmp file + rename) or append-only, so a live viewer never sees a
half-written file. Only the main rank writes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
log = logging.getLogger(__name__)


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Console handler that writes to the *current* ``sys.stderr``.

    ``logging.StreamHandler()`` binds the stream object at construction. When the stream
    is replaced and closed later (pytest capture, notebooks, a CLI embedded in another
    tool), every record then fails with "I/O operation on closed file"."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass  # always the current sys.stderr


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure std logging once: console + optional file handler."""
    root = logging.getLogger()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    if not any(getattr(h, "_tinylab", False) for h in root.handlers):
        handler = _StderrHandler()
        handler.setFormatter(fmt)
        handler._tinylab = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    if log_file is not None:
        target = str(log_file.resolve())
        if not any(getattr(h, "baseFilename", None) == target for h in root.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)
    root.setLevel(level)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _clean(value: Any) -> Any:
    """Replace NaN/inf (invalid JSON) with None, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def dumps(data: Any) -> str:
    return json.dumps(_clean(data), default=_json_default, allow_nan=False)


def write_json_atomic(path: Path, data: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(_clean(data), default=_json_default, indent=indent, allow_nan=False))
    tmp.replace(path)


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Commit hash and dirty flag of the source tree (None outside a git checkout)."""
    where = cwd or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=where, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=where, capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def new_run_id(name: str) -> str:
    return f"{name}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


class RunLogger:
    """Writes everything the viewer needs. A disabled logger (non-main rank) is a no-op."""

    def __init__(self, run_dir: Path, enabled: bool = True) -> None:
        self.run_dir = run_dir
        self.enabled = enabled
        self._t0 = time.time()
        if enabled:
            for sub in ("snapshots", "artifacts", "checkpoints"):
                (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- manifest
    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        data: dict[str, Any] = json.loads(self.manifest_path.read_text())
        return data

    def update_manifest(self, fn: Callable[[dict[str, Any]], None]) -> None:
        if not self.enabled:
            return
        manifest = self.read_manifest()
        fn(manifest)
        write_json_atomic(self.manifest_path, manifest)

    def init_manifest(
        self, run_id: str, name: str, hardware: Mapping[str, Any], runtime: Mapping[str, Any]
    ) -> None:
        def fill(m: dict[str, Any]) -> None:
            m.setdefault("schema_version", SCHEMA_VERSION)
            m.setdefault("run_id", run_id)
            m.setdefault("name", name)
            m.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
            m.setdefault("stages", [])
            m["git"] = git_info()
            m["hardware"] = dict(hardware)
            m["runtime"] = dict(runtime)

        self.update_manifest(fill)

    def _stage_entry(self, manifest: dict[str, Any], stage: str) -> dict[str, Any]:
        for entry in manifest.setdefault("stages", []):
            if entry["name"] == stage:
                assert isinstance(entry, dict)
                return entry
        entry = {"name": stage}
        manifest["stages"].append(entry)
        return entry

    def stage_started(self, stage: str, config: Mapping[str, Any], init_from: str | None) -> None:
        def fill(m: dict[str, Any]) -> None:
            entry = self._stage_entry(m, stage)
            entry.update(
                status="running",
                started=dt.datetime.now().isoformat(timespec="seconds"),
                config=dict(config),
                init_from=init_from,
            )

        self.update_manifest(fill)

    def stage_finished(
        self, stage: str, checkpoint: Path | None, summary: Mapping[str, Any] | None = None
    ) -> None:
        def fill(m: dict[str, Any]) -> None:
            entry = self._stage_entry(m, stage)
            entry.update(
                status="done",
                finished=dt.datetime.now().isoformat(timespec="seconds"),
                checkpoint=self.relative(checkpoint) if checkpoint else None,
                summary=dict(summary or {}),
            )

        self.update_manifest(fill)

    def stage_failed(self, stage: str, error: str) -> None:
        def fill(m: dict[str, Any]) -> None:
            self._stage_entry(m, stage).update(status="failed", error=error)

        self.update_manifest(fill)

    def set_field(self, key: str, value: Any) -> None:
        def fill(m: dict[str, Any]) -> None:
            m[key] = value

        self.update_manifest(fill)

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.run_dir.resolve()))
        except ValueError:
            return str(path)

    def resolve(self, rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else self.run_dir / path

    def checkpoint_of(self, stage: str) -> Path | None:
        for entry in self.read_manifest().get("stages", []):
            if entry["name"] == stage and entry.get("status") == "done" and entry.get("checkpoint"):
                return self.resolve(entry["checkpoint"])
        return None

    def last_finished_stage(self, before: str, order: tuple[str, ...]) -> str | None:
        done = {
            e["name"]
            for e in self.read_manifest().get("stages", [])
            if e.get("status") == "done" and e.get("checkpoint")
        }
        idx = order.index(before)
        for name in reversed(order[:idx]):
            if name in done:
                return name
        return None

    # ------------------------------------------------------------------ metrics
    def log_metrics(self, stage: str, step: int, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"stage": stage, "step": step, "time": round(time.time() - self._t0, 3)}
        record.update(metrics)
        with (self.run_dir / "metrics.jsonl").open("a") as fh:
            fh.write(dumps(record) + "\n")

    # ---------------------------------------------------------------- snapshots
    def save_snapshot(
        self,
        stage: str,
        step: int,
        data: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None = None,
        name: str = "data",
    ) -> Path | None:
        """Write ``<name>.json`` (+ ``<name>.npz``) under ``snapshots/<stage>/<step>/``.

        Different producers (internals, generations, rollouts) use different names, so
        an eval and a snapshot at the same step never overwrite each other."""
        if not self.enabled:
            return None
        target = self.run_dir / "snapshots" / stage / f"{step:08d}"
        target.mkdir(parents=True, exist_ok=True)
        if arrays:
            tmp = target / f".{name}.tmp{os.getpid()}.npz"
            payload_arrays: dict[str, Any] = {k: np.asarray(v) for k, v in arrays.items()}
            np.savez_compressed(tmp, **payload_arrays)
            tmp.replace(target / f"{name}.npz")
        payload = {"schema_version": SCHEMA_VERSION, "stage": stage, "step": step, **data}
        write_json_atomic(target / f"{name}.json", payload, indent=None)
        return target

    # ---------------------------------------------------------------- artifacts
    def write_artifact(self, name: str, data: Any) -> Path | None:
        if not self.enabled:
            return None
        path = self.run_dir / "artifacts" / name
        if isinstance(data, str):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        else:
            write_json_atomic(path, data)
        return path
