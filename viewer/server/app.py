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
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response
from starlette.types import Scope

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


def _spans(spans: list[StageSpan]) -> list[Span]:
    return [Span(name=s.name, status=s.status, start=s.start, end=s.end) for s in spans]


def create_app(runs_dir: Path, poll_seconds: float = 1.0, web_dist: Path = WEB_DIST) -> FastAPI:
    store = RunStore(runs_dir)
    app = FastAPI(title="TinyMoE viewer", version="1")

    def guard(fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"not found: {exc}") from exc

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


def serve(runs_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    poll = float(os.environ.get("TINYMOE_VIEWER_POLL", "1.0"))
    uvicorn.run(create_app(runs_dir, poll), host=host, port=port, log_level="info")
