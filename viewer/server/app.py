"""FastAPI backend of the viewer: typed JSON API over run directories + SSE live updates,
and the built React frontend (``viewer/web/dist``) as static files.

    GET /api/runs                                   runs, newest first
    GET /api/runs/{run}/manifest                    manifest.json
    GET /api/runs/{run}/metrics?offset=N            records after byte N (+ next offset,
                                                    stage timeline, global steps)
    GET /api/runs/{run}/snapshots                   snapshot index (stage, step, gstep)
    GET /api/runs/{run}/snapshots/{stage}/{step}/{name}          snapshot JSON
    GET /api/runs/{run}/snapshots/{stage}/{step}/{name}/{key}    one npz array
    GET /api/runs/{run}/artifacts[/{name}]          reports, comparison tables
    GET /api/runs/{run}/events                      SSE: metrics / snapshots / manifest
    GET /api/runs/{run}/architecture?stage=&which=  parameter shapes of a stage's model

Run control (``viewer/server/control.py``; on by default only on a loopback host):

    GET  /api/control                               enabled? + startable pipelines
    GET  /api/runs/{run}/control                    alive? pid, pending pause, log tail
    POST /api/control/start        {pipeline, until?}        start a new run
    POST /api/runs/{run}/control/resume   {until?}           resume (or rerun) a run
    POST /api/runs/{run}/control/pause    {when}             "now" | "after_stage"
    POST /api/runs/{run}/control/cancel                      withdraw a pause request

Control requests must name a loopback Host (a DNS-rebinding page cannot reach them), a
matching Origin when one is sent, and a JSON body (so a cross-site form cannot either).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response
from starlette.types import Scope

from viewer.server import architecture
from viewer.server.control import ControlError, Controller, loopback_host, request_host
from viewer.server.runs import NotFoundError, RunStore, StageSpan

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class WebFiles(StaticFiles):
    """The built frontend. Vite content-hashes everything under ``assets/``, so those are
    cached for good; ``index.html`` (which names the current hashes) is revalidated on
    every load, so a rebuilt viewer shows up on a plain reload."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        immutable = path.startswith("assets/") and response.status_code == 200
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if immutable else "no-cache"
        )
        return response


class StageStatus(BaseModel):
    name: str
    status: str


class RunSummary(BaseModel):
    id: str
    name: str
    created: str | None
    stages: list[StageStatus]
    planned: list[str] | None = None
    mtime: float


class Span(BaseModel):
    name: str
    status: str
    start: int
    end: int


class MetricsResponse(BaseModel):
    records: list[dict[str, Any]]
    offset: int
    timeline: list[Span]


class SnapshotEntry(BaseModel):
    stage: str
    step: int
    gstep: int
    files: list[str]
    arrays: list[str]


class ArrayResponse(BaseModel):
    key: str
    shape: list[int]
    data: Any


class StartBody(BaseModel):
    pipeline: str
    until: str | None = None


class ResumeBody(BaseModel):
    until: str | None = None


class PauseBody(BaseModel):
    when: str = "now"


def control_guard(request: Request) -> None:
    """Only same-machine pages may drive training."""
    host = request_host(request.headers.get("host", ""))
    if not loopback_host(host):
        raise HTTPException(403, "run control only answers requests to localhost")
    origin = request.headers.get("origin")
    if origin is not None and request_host(origin) != host:
        raise HTTPException(403, "cross-origin control request refused")
    if request.method == "POST" and not request.headers.get("content-type", "").startswith(
        "application/json"
    ):
        raise HTTPException(415, "control requests take a JSON body")


def _spans(spans: list[StageSpan]) -> list[Span]:
    return [Span(name=s.name, status=s.status, start=s.start, end=s.end) for s in spans]


def create_app(
    runs_dir: Path,
    poll_seconds: float = 1.0,
    web_dist: Path = WEB_DIST,
    control: bool = False,
    controller: Controller | None = None,
) -> FastAPI:
    store = RunStore(runs_dir)
    ctl = controller or Controller(runs_dir, enabled=control)
    architecture.prewarm()
    app = FastAPI(title="TinyMoE viewer", version="1")
    ctl_dep = [Depends(control_guard)]

    def guard(fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"not found: {exc}") from exc
        except ControlError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs() -> list[dict[str, Any]]:
        return store.list_runs()

    @app.get("/api/runs/{run_id}/manifest")
    def manifest(run_id: str) -> dict[str, Any]:
        data: dict[str, Any] = guard(store.manifest, run_id)
        return data

    @app.get("/api/runs/{run_id}/metrics", response_model=MetricsResponse)
    def metrics(run_id: str, offset: int = 0) -> MetricsResponse:
        all_records, end = guard(store.read_metrics, run_id, 0)
        spans = store.timeline(run_id, all_records)
        records, end = store.read_metrics(run_id, offset) if offset else (all_records, end)
        return MetricsResponse(
            records=store.with_global_steps(records, spans), offset=end, timeline=_spans(spans)
        )

    @app.get("/api/runs/{run_id}/snapshots", response_model=list[SnapshotEntry])
    def snapshots(run_id: str) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = guard(store.snapshots, run_id)
        return data

    @app.get("/api/runs/{run_id}/snapshots/{stage}/{step}/{name}")
    def snapshot(run_id: str, stage: str, step: int, name: str) -> dict[str, Any]:
        data: dict[str, Any] = guard(store.snapshot, run_id, stage, step, name)
        return data

    @app.get(
        "/api/runs/{run_id}/snapshots/{stage}/{step}/{name}/{key}", response_model=ArrayResponse
    )
    def snapshot_array(run_id: str, stage: str, step: int, name: str, key: str) -> dict[str, Any]:
        data: dict[str, Any] = guard(store.snapshot_array, run_id, stage, step, name, key)
        return data

    @app.get("/api/runs/{run_id}/artifacts", response_model=list[str])
    def artifacts(run_id: str) -> list[str]:
        names: list[str] = guard(store.artifacts, run_id)
        return names

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str) -> Any:
        return guard(store.artifact, run_id, name)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, offset: int = 0) -> StreamingResponse:
        guard(store.run_dir, run_id)

        stream = event_stream(store, run_id, offset, poll_seconds, request.is_disconnected)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream, media_type="text/event-stream", headers=headers)

    @app.get("/api/runs/{run_id}/architecture")
    def model_architecture(
        run_id: str, stage: str | None = None, which: str = "model"
    ) -> dict[str, Any]:
        run_dir = guard(store.run_dir, run_id)
        data: dict[str, Any] = guard(architecture.describe, run_dir, stage, which)
        return data

    # ------------------------------------------------------------ run control
    @app.get("/api/control")
    def control_info() -> dict[str, Any]:
        return {"enabled": ctl.enabled, "pipelines": ctl.pipelines(), "busy": ctl.busy()}

    @app.get("/api/runs/{run_id}/control", dependencies=ctl_dep)
    def control_state(run_id: str) -> dict[str, Any]:
        data: dict[str, Any] = guard(ctl.state, run_id, guard(store.manifest, run_id))
        return data

    @app.post("/api/control/start", dependencies=ctl_dep)
    def control_start(body: StartBody) -> dict[str, Any]:
        data: dict[str, Any] = guard(ctl.start, body.pipeline, None, body.until)
        return data

    @app.post("/api/runs/{run_id}/control/resume", dependencies=ctl_dep)
    def control_resume(run_id: str, body: ResumeBody) -> dict[str, Any]:
        manifest = guard(store.manifest, run_id)
        pipeline = ctl._pipeline_of(manifest)
        if pipeline is None:
            raise HTTPException(400, f"cannot tell which pipeline made {run_id}")
        data: dict[str, Any] = guard(ctl.start, pipeline, run_id, body.until)
        return data

    @app.post("/api/runs/{run_id}/control/pause", dependencies=ctl_dep)
    def control_pause(run_id: str, body: PauseBody) -> dict[str, Any]:
        guard(store.run_dir, run_id)
        guard(ctl.pause, run_id, body.when)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/control/cancel", dependencies=ctl_dep)
    def control_cancel(run_id: str) -> dict[str, Any]:
        guard(store.run_dir, run_id)
        guard(ctl.cancel, run_id)
        return {"ok": True}

    if web_dist.is_dir():
        app.mount("/", WebFiles(directory=web_dist, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def missing_frontend() -> str:
            return (
                "<h1>TinyMoE viewer</h1><p>The frontend is not built. Run "
                "<code>make web</code> (needs Node), then restart. The JSON API is live at "
                "<a href='/api/runs'>/api/runs</a>.</p>"
            )

    return app


async def event_stream(
    store: RunStore,
    run_id: str,
    offset: int,
    poll_seconds: float,
    disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    """SSE events for one run: ``hello``, then ``metrics`` (new records after ``offset``,
    with global steps and the current timeline) and ``snapshots`` / ``manifest`` /
    ``artifacts`` notifications whenever those files change. Polls file signatures, so
    it works for any writer (other processes, DDP rank 0, a run on a network disk)."""
    last: dict[str, Any] = {}
    pos = offset
    yield _sse("hello", {"run": run_id})
    while not await disconnected():
        sig = store.signature(run_id)
        if sig["metrics"] != last.get("metrics"):
            all_records, _ = store.read_metrics(run_id, 0)
            spans = store.timeline(run_id, all_records)
            new, pos = store.read_metrics(run_id, pos)
            if new:
                payload = {
                    "records": store.with_global_steps(new, spans),
                    "offset": pos,
                    "timeline": [s.model_dump() for s in _spans(spans)],
                }
                yield _sse("metrics", payload)
        for key in ("snapshots", "manifest", "artifacts"):
            if last and sig[key] != last.get(key):
                yield _sse(key, {key: sig[key]})
        last = sig
        await asyncio.sleep(poll_seconds)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def serve(
    runs_dir: Path, host: str = "127.0.0.1", port: int = 8765, control: bool | None = None
) -> None:
    import uvicorn

    poll = float(os.environ.get("TINYMOE_VIEWER_POLL", "1.0"))
    enabled = loopback_host(host) if control is None else control
    app = create_app(runs_dir, poll, control=enabled)
    uvicorn.run(app, host=host, port=port, log_level="info")
