"""Run control: a small file contract between a running pipeline and whoever controls it.

* ``<run_dir>/control.json`` (written by the viewer or ``tinymoe pause``) asks the
  pipeline to pause: ``{"request": "now"}`` makes the trainer save a checkpoint after
  its current step and stop; ``{"request": "after_stage"}`` stops before the next stage
  starts. The pipeline clears the file when it starts and when it acts on it.
* ``<run_dir>/process.json`` is written by the pipeline itself while it runs (pid,
  pipeline file, start time), so a viewer can tell whether the run is alive no matter
  who started it.

A paused run resumes by re-running its pipeline with the same ``run.run_id``: finished
stages are skipped and the paused stage resumes from the checkpoint it just saved.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

CONTROL_FILE = "control.json"
PROCESS_FILE = "process.json"
Request = Literal["now", "after_stage"]
REQUESTS: tuple[Request, ...] = ("now", "after_stage")


class StopRequested(Exception):  # noqa: N818 (a request, not an error)
    """Raised inside a stage when a pause was requested; the stage is marked paused."""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic: a reader never sees half a file


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def request_pause(run_dir: Path, when: Request) -> None:
    if when not in REQUESTS:
        raise ValueError(f"pause request must be one of {REQUESTS}, not {when!r}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / CONTROL_FILE, {"request": when, "at": _now()})


def pending(run_dir: Path) -> Request | None:
    data = _read_json(run_dir / CONTROL_FILE)
    req = data.get("request") if data else None
    return req if req in REQUESTS else None


def clear(run_dir: Path) -> None:
    (run_dir / CONTROL_FILE).unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def process_info(run_dir: Path) -> dict[str, Any] | None:
    """The pipeline process of this run if it is alive, else None."""
    data = _read_json(run_dir / PROCESS_FILE)
    if not data or not pid_alive(int(data.get("pid", 0))):
        return None
    return data


@contextmanager
def running(run_dir: Path, pipeline: str, argv: list[str], enabled: bool = True) -> Iterator[None]:
    """Mark the run as alive for the duration of the block (rank 0 only)."""
    if not enabled:
        yield
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PROCESS_FILE
    _write_json(path, {"pid": os.getpid(), "pipeline": pipeline, "argv": argv, "started": _now()})
    try:
        yield
    finally:
        data = _read_json(path)
        if data and data.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
