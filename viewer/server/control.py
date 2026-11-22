"""Run control from the viewer: start a pipeline, pause it, resume it.

Everything goes through the file contract of ``tinymoe.control`` (``control.json`` to
ask for a pause, ``process.json`` to see whether a pipeline is alive), so a run started
from a terminal is controlled exactly like one started here.

The viewer starts only what an allowlist names: a pipeline file under
``configs/pipeline/``, an optional ``--until`` stage of that pipeline and a run id of
safe characters. A resumed run re-uses the overrides recorded in its manifest. Nothing
else from the request reaches the command line.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tinymoe import control
from tinymoe.config import load_yaml
from viewer.server.runs import _read_json

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
LOG_FILE = "pipeline.log"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
Spawn = Callable[..., Any]  # subprocess.Popen (tests pass a fake)


class ControlError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _owned_args(args: list[str]) -> list[str]:
    """Overrides recorded for a run, minus the ones the controller sets itself."""
    return [a for a in args if not a.startswith(("run.run_id=", "run.out_dir="))]


class Controller:
    def __init__(
        self,
        runs_dir: Path,
        enabled: bool,
        root: Path = ROOT,
        spawn: Spawn = subprocess.Popen,
    ) -> None:
        self.runs_dir = runs_dir.resolve()
        self.enabled = enabled
        self.root = root
        self.spawn = spawn
        self.procs: dict[str, Any] = {}  # run id -> Popen we started (reaped by poll())

    # --------------------------------------------------------------- pipelines
    def pipelines(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((self.root / "configs" / "pipeline").glob("*.yaml")):
            try:
                spec = load_yaml(path)
            except Exception:
                continue
            out.append(
                {
                    "name": str(spec.get("name", path.stem)),
                    "file": path.name,
                    "stages": [s["name"] for s in spec.get("stages", [])],
                }
            )
        return out

    def _pipeline(self, file: str) -> dict[str, Any]:
        for p in self.pipelines():
            if p["file"] == file:
                return p
        raise ControlError(400, f"unknown pipeline {file!r}")

    def _pipeline_of(self, manifest: dict[str, Any]) -> str | None:
        """The pipeline file a run came from (recorded, else guessed from its name)."""
        names = {p["file"] for p in self.pipelines()}
        recorded = manifest.get("pipeline")
        if isinstance(recorded, str) and Path(recorded).name in names:
            return Path(recorded).name
        guess = f"{manifest.get('name', '')}.yaml"
        return guess if guess in names else None

    # ------------------------------------------------------------------- state
    def _alive(self, run_id: str) -> dict[str, Any] | None:
        info = control.process_info(self.runs_dir / run_id)
        proc = self.procs.get(run_id)
        if proc is not None:
            code = proc.poll()
            if code is None:
                return info or {"pid": proc.pid, "starting": True}
            self.procs.pop(run_id)
        return info

    def busy(self) -> str | None:
        """The run whose pipeline is alive, if any (one at a time: they share the device)."""
        for run_id in list(self.procs):
            if self._alive(run_id):
                return run_id
        if self.runs_dir.is_dir():
            for d in self.runs_dir.iterdir():
                if (d / control.PROCESS_FILE).exists() and control.process_info(d):
                    return d.name
        return None

    def state(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        alive = self._alive(run_id)
        pipeline = self._pipeline_of(manifest)
        stages = manifest.get("pipeline_stages")
        if not isinstance(stages, list) and pipeline:
            stages = self._pipeline(pipeline)["stages"]
        log = run_dir / LOG_FILE
        tail = log.read_text(errors="replace").splitlines()[-30:] if log.exists() else []
        return {
            "enabled": self.enabled,
            "running": alive is not None,
            "starting": bool(alive and alive.get("starting")),
            "pid": alive.get("pid") if alive else None,
            "started": alive.get("started") if alive else None,
            "pending": control.pending(run_dir),
            "pipeline": pipeline,
            "pipeline_stages": stages or [],
            "busy": self.busy(),
            "log": tail,
        }

    # ----------------------------------------------------------------- actions
    def _check(self) -> None:
        if not self.enabled:
            raise ControlError(403, "run control is off (start the viewer with --control)")

    def start(
        self, pipeline: str, run_id: str | None = None, until: str | None = None
    ) -> dict[str, Any]:
        """Start (run_id None or new) or resume (existing run) a pipeline in the background."""
        self._check()
        spec = self._pipeline(pipeline)
        if until is not None and until not in spec["stages"]:
            raise ControlError(400, f"{pipeline} has no stage {until!r}")
        args: list[str] = []
        if run_id is None:
            run_id = f"{spec['name']}-{dt.datetime.now():%Y%m%d-%H%M%S}"
        if not RUN_ID.match(run_id):
            raise ControlError(400, f"bad run id {run_id!r}")
        run_dir = self.runs_dir / run_id
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if self._pipeline_of(manifest) != pipeline:
                raise ControlError(400, f"{run_id} was not made by {pipeline}")
            recorded = manifest.get("pipeline_args")
            if isinstance(recorded, list):
                args = _owned_args([str(a) for a in recorded])
        busy = self.busy()
        if busy:
            raise ControlError(409, f"{busy} is already running; pause it first")
        cmd = [sys.executable, "-m", "tinymoe.cli", "pipeline"]
        if until:
            cmd += ["--until", until]
        cmd += [
            str(self.root / "configs" / "pipeline" / pipeline),
            *args,
            f"run.run_id={run_id}",
            f"run.out_dir={self.runs_dir}",
        ]
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            cmd = ["caffeinate", "-i", *cmd]  # keep the Mac awake while it trains
        run_dir.mkdir(parents=True, exist_ok=True)
        control.clear(run_dir)
        with (run_dir / LOG_FILE).open("ab") as log:
            log.write(f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n".encode())
            log.flush()
            self.procs[run_id] = self.spawn(
                cmd,
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # survives a viewer restart
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        return {"run_id": run_id, "cmd": cmd}

    def pause(self, run_id: str, when: str) -> None:
        self._check()
        if when not in control.REQUESTS:
            raise ControlError(400, f"pause must be one of {control.REQUESTS}")
        if self._alive(run_id) is None:
            raise ControlError(409, f"{run_id} is not running")
        control.request_pause(self.runs_dir / run_id, when)

    def cancel(self, run_id: str) -> None:
        """Withdraw a pause request that has not been acted on yet."""
        self._check()
        control.clear(self.runs_dir / run_id)


def loopback_host(host: str) -> bool:
    return host.strip("[]") in LOOPBACK


def request_host(header: str) -> str:
    """Hostname of a Host/Origin value: ``localhost:8765``, ``[::1]:8765``, ``http://x:1``."""
    value = header.split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):
        return value[1 : value.find("]")]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value
