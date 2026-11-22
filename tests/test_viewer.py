"""M6: viewer API over the RunLogger file contract (live + partial data)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tinymoe.logging import RunLogger
from viewer.server.app import create_app, event_stream
from viewer.server.runs import RunStore


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    logger = RunLogger(tmp_path / "r1")
    logger.init_manifest("r1", "demo", {"device": "cpu"}, {"precision": "fp32"})
    logger.stage_started("pretrain", {"train": {}}, None)
    for step in (10, 20, 30):
        logger.log_metrics("pretrain", step, {"loss": 5.0 - step / 10, "loss/stories": 4.0})
    logger.save_snapshot(
        "pretrain", 30, {"model": {"n_layers": 2}}, {"attn_story": np.ones((2, 4, 3, 3))}
    )
    logger.save_snapshot(
        "pretrain",
        30,
        {
            "kind": "base",
            "generations": [{"id": "s0", "kind": "story", "prompt": "p", "completion": "c"}],
        },
        name="generations",
    )
    logger.stage_finished("pretrain", None, {"val/loss/stories": float("nan")})
    logger.stage_started("midtrain", {"train": {}}, "pretrain")
    logger.log_metrics("midtrain", 5, {"loss": 1.0})
    logger.write_artifact("report.json", {"rows": []})
    return tmp_path


@pytest.fixture
def client(run_root: Path) -> TestClient:
    return TestClient(create_app(run_root, poll_seconds=0.05))


def test_runs_and_manifest(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert [r["id"] for r in runs] == ["r1"]
    assert [s["status"] for s in runs[0]["stages"]] == ["done", "running"]
    manifest = client.get("/api/runs/r1/manifest").json()
    assert manifest["stages"][0]["summary"]["val/loss/stories"] is None  # NaN -> null


def test_metrics_timeline_and_global_steps(client: TestClient) -> None:
    res = client.get("/api/runs/r1/metrics").json()
    assert [s["name"] for s in res["timeline"]] == ["pretrain", "midtrain"]
    assert res["timeline"][1]["start"] == 30 and res["timeline"][1]["end"] == 35
    assert [r["gstep"] for r in res["records"]] == [10, 20, 30, 35]
    again = client.get(f"/api/runs/r1/metrics?offset={res['offset']}").json()
    assert again["records"] == []


def test_partial_line_is_not_returned(run_root: Path) -> None:
    store = RunStore(run_root)
    _, offset = store.read_metrics("r1")
    with (run_root / "r1" / "metrics.jsonl").open("a") as fh:
        fh.write('{"stage": "midtrain", "step": 6, "lo')  # writer mid-line
    records, offset2 = store.read_metrics("r1", offset)
    assert records == [] and offset2 == offset
    with (run_root / "r1" / "metrics.jsonl").open("a") as fh:
        fh.write('ss": 0.5}\n')
    records, _ = store.read_metrics("r1", offset)
    assert records[0]["loss"] == 0.5


def test_snapshots_and_arrays(client: TestClient) -> None:
    snaps = client.get("/api/runs/r1/snapshots").json()
    assert snaps == [
        {
            "stage": "pretrain",
            "step": 30,
            "gstep": 30,
            "files": ["data", "generations"],
            "arrays": ["data"],
        }
    ]
    gens = client.get("/api/runs/r1/snapshots/pretrain/30/generations").json()
    assert gens["generations"][0]["id"] == "s0" and gens["schema_version"] == 1
    arr = client.get("/api/runs/r1/snapshots/pretrain/30/data/attn_story").json()
    assert arr["shape"] == [2, 4, 3, 3]


def test_artifacts(client: TestClient) -> None:
    assert client.get("/api/runs/r1/artifacts").json() == ["report.json"]
    assert client.get("/api/runs/r1/artifacts/report.json").json() == {"rows": []}


def test_not_found_and_traversal(client: TestClient) -> None:
    assert client.get("/api/runs/nope/manifest").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fetc/manifest").status_code == 404
    assert client.get("/api/runs/r1/snapshots/pretrain/99/data").status_code == 404
    assert client.get("/api/runs/r1/artifacts/..%2Fmanifest.json").status_code == 404


def test_sse_streams_new_metrics(run_root: Path) -> None:
    """The event stream delivers records appended after the client's offset, with global
    steps, and announces new snapshots."""
    store = RunStore(run_root)
    _, offset = store.read_metrics("r1")
    logger = RunLogger(run_root / "r1")

    async def never() -> bool:
        return False

    async def collect() -> list[tuple[str, dict[str, object]]]:
        stream = event_stream(store, "r1", offset, 0.01, never)
        events = []
        async for chunk in stream:
            name, data = chunk.split("\n")[0][7:], json.loads(chunk.split("\n")[1][6:])
            events.append((name, data))
            if name == "hello":
                logger.log_metrics("midtrain", 9, {"loss": 0.25})
            elif name == "metrics":
                logger.save_snapshot("midtrain", 9, {"model": {}})
            elif name == "snapshots":
                break
        await stream.aclose()
        return events

    events = asyncio.run(asyncio.wait_for(collect(), timeout=10))
    assert [e[0] for e in events] == ["hello", "metrics", "snapshots"]
    record = events[1][1]["records"][0]  # type: ignore[index]
    assert record["step"] == 9 and record["gstep"] == 39


def test_frontend_fallback_page(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200


def test_frontend_cache_headers(run_root: Path, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>viewer</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)")
    client = TestClient(create_app(run_root, web_dist=dist))
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
    asset = client.get("/assets/index-abc123.js")
    assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
    assert client.get("/api/runs").status_code == 200  # the API is not shadowed
