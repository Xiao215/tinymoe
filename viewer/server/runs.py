"""Read-only access to run directories (the RunLogger file contract, tinymoe/logging.py).

Everything is derived from files, so the viewer works the same on a live run (files are
appended / atomically replaced while we read) and on a finished one. Partial data is
normal: a stage that has not logged yet, a half-written last JSONL line, a snapshot whose
``generations.json`` is missing. Every reader tolerates those.

Global steps: stages restart their step counter, so the timeline places stage *i* after
the last step of stages ``< i`` (in manifest order) and every record gets ``gstep``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_STEP_DIR = re.compile(r"^\d{8}$")
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class NotFoundError(LookupError):
    """A run, stage, snapshot or artifact that does not exist (HTTP 404)."""


def safe_name(name: str) -> str:
    """Reject path components that could escape the runs directory."""
    if not _SAFE.match(name) or name in (".", ".."):
        raise NotFoundError(name)
    return name


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise NotFoundError(str(path)) from exc


def _clean(value: Any) -> Any:
    """NaN/inf -> None so responses are strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


@dataclass(frozen=True)
class StageSpan:
    name: str
    status: str
    start: int  # global step at which the stage begins
    end: int  # global step of its last record (== start until something is logged)


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    # ------------------------------------------------------------------- runs
    def run_dir(self, run_id: str) -> Path:
        path = self.root / safe_name(run_id)
        if not (path / "manifest.json").is_file():
            raise NotFoundError(run_id)
        return path

    def list_runs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for path in sorted(self.root.iterdir()):
            if not (path / "manifest.json").is_file():
                continue
            try:
                m = _read_json(path / "manifest.json")
            except NotFoundError:
                continue  # being rewritten right now
            out.append(
                {
                    "id": path.name,
                    "name": m.get("name", path.name),
                    "created": m.get("created"),
                    "stages": [
                        {"name": s["name"], "status": s.get("status", "unknown")}
                        for s in m.get("stages", [])
                    ],
                    "planned": m.get("pipeline_stages"),
                    "mtime": (path / "manifest.json").stat().st_mtime,
                }
            )
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    def manifest(self, run_id: str) -> dict[str, Any]:
        data: dict[str, Any] = _read_json(self.run_dir(run_id) / "manifest.json")
        return _clean(data)  # type: ignore[no-any-return]

    # ---------------------------------------------------------------- metrics
    def read_metrics(self, run_id: str, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Records after byte ``offset`` and the offset to continue from. A trailing line
        without a newline is still being written and is left for the next read."""
        path = self.run_dir(run_id) / "metrics.jsonl"
        if not path.is_file():
            return [], 0
        with path.open("rb") as fh:
            size = path.stat().st_size
            if offset > size:  # file was truncated / replaced
                offset = 0
            fh.seek(offset)
            chunk = fh.read(size - offset)
        end = chunk.rfind(b"\n") + 1
        records = []
        for line in chunk[:end].splitlines():
            if line.strip():
                try:
                    records.append(_clean(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return records, offset + end

    def timeline(self, run_id: str, records: list[dict[str, Any]] | None = None) -> list[StageSpan]:
        manifest = self.manifest(run_id)
        if records is None:
            records, _ = self.read_metrics(run_id)
        last: dict[str, int] = {}
        for r in records:
            last[r["stage"]] = max(last.get(r["stage"], 0), int(r["step"]))
        spans = []
        start = 0
        order = [s["name"] for s in manifest.get("stages", [])]
        order += [s for s in last if s not in order]
        status = {s["name"]: s.get("status", "unknown") for s in manifest.get("stages", [])}
        for name in order:
            end = start + last.get(name, 0)
            spans.append(StageSpan(name, status.get(name, "unknown"), start, end))
            start = end
        return spans

    @staticmethod
    def with_global_steps(
        records: list[dict[str, Any]], spans: list[StageSpan]
    ) -> list[dict[str, Any]]:
        offsets = {s.name: s.start for s in spans}
        return [{**r, "gstep": offsets.get(r["stage"], 0) + int(r["step"])} for r in records]

    # -------------------------------------------------------------- snapshots
    def snapshots(self, run_id: str) -> list[dict[str, Any]]:
        run = self.run_dir(run_id)
        spans = {s.name: s for s in self.timeline(run_id)}
        out: list[dict[str, Any]] = []
        base = run / "snapshots"
        if not base.is_dir():
            return out
        for stage_dir in base.iterdir():
            if not stage_dir.is_dir():
                continue
            for step_dir in stage_dir.iterdir():
                if not _STEP_DIR.match(step_dir.name):
                    continue
                files = sorted(p.stem for p in step_dir.glob("*.json"))
                if not files:
                    continue
                step = int(step_dir.name)
                span = spans.get(stage_dir.name)
                out.append(
                    {
                        "stage": stage_dir.name,
                        "step": step,
                        "gstep": (span.start if span else 0) + step,
                        "files": files,
                        "arrays": sorted(p.stem for p in step_dir.glob("*.npz")),
                    }
                )
        out.sort(key=lambda s: (s["gstep"], s["stage"]))
        return out

    def snapshot(self, run_id: str, stage: str, step: int, name: str) -> dict[str, Any]:
        path = self._step_dir(run_id, stage, step) / f"{safe_name(name)}.json"
        return _clean(_read_json(path))  # type: ignore[no-any-return]

    def snapshot_array(
        self, run_id: str, stage: str, step: int, name: str, key: str
    ) -> dict[str, Any]:
        path = self._step_dir(run_id, stage, step) / f"{safe_name(name)}.npz"
        if not path.is_file():
            raise NotFoundError(str(path))
        with np.load(path) as data:
            if key not in data.files:
                raise NotFoundError(key)
            arr = data[key].astype(np.float32)
        return {"key": key, "shape": list(arr.shape), "data": _clean(arr.round(4).tolist())}

    def _step_dir(self, run_id: str, stage: str, step: int) -> Path:
        path = self.run_dir(run_id) / "snapshots" / safe_name(stage) / f"{step:08d}"
        if not path.is_dir():
            raise NotFoundError(str(path))
        return path

    # -------------------------------------------------------------- artifacts
    def artifacts(self, run_id: str) -> list[str]:
        base = self.run_dir(run_id) / "artifacts"
        return sorted(p.name for p in base.iterdir() if p.is_file()) if base.is_dir() else []

    def artifact(self, run_id: str, name: str) -> Any:
        path = self.run_dir(run_id) / "artifacts" / safe_name(name)
        if path.suffix == ".json":
            return _clean(_read_json(path))
        if not path.is_file():
            raise NotFoundError(name)
        return {"text": path.read_text()}

    # ------------------------------------------------------------------- live
    def signature(self, run_id: str) -> dict[str, Any]:
        """Cheap change detector: sizes/mtimes of the files the UI reacts to."""
        run = self.run_dir(run_id)
        metrics = run / "metrics.jsonl"
        snaps = run / "snapshots"
        n_snaps = sum(1 for _ in snaps.glob("*/*/*.json")) if snaps.is_dir() else 0
        return {
            "manifest": (run / "manifest.json").stat().st_mtime,
            "metrics": metrics.stat().st_size if metrics.is_file() else 0,
            "snapshots": n_snaps,
            "artifacts": len(self.artifacts(run_id)),
        }
