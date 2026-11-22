"""Run control: pause requests, process liveness, the viewer's control API and the
architecture endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from helpers import RandomRows, make_trainer, read_losses, tiny_model

from tinymoe import control, pipeline
from tinymoe.config import BatchPlan, load_stage_config
from tinymoe.logging import RunLogger
from tinymoe.model.config import param_count
from tinymoe.train.loop import Hooks
from tinymoe.train.providers import RowProvider
from viewer.server.app import create_app
from viewer.server.control import Controller, request_host

PLAN = BatchPlan(micro_batch_size=2, grad_accum=2, world_size=1, unit=16)
KW: dict[str, Any] = {"max_steps": 10, "optimizer": "adamw", "schedule": "constant"}


def _provider() -> RowProvider:
    return RowProvider(RandomRows(16), PLAN, 0, torch.device("cpu"))


def test_pause_now_checkpoints_and_resume_is_bit_identical(tmp_path: Path) -> None:
    full = make_trainer(tmp_path / "a", tiny_model(True), _provider(), plan=PLAN, **KW)
    full.fit()

    run = tmp_path / "b"

    def ask(step: int) -> None:  # a pause request arrives during step 4
        control.request_pause(run, "now")

    hooks = Hooks(snapshot=ask, snapshot_interval=4)
    part = make_trainer(run, tiny_model(True), _provider(), plan=PLAN, hooks=hooks, **KW)
    with pytest.raises(control.StopRequested):
        part.fit()
    assert (run / "ckpt" / "step_00000004.pt").exists()
    assert max(read_losses(run)) == 4  # stopped right after the step it was asked in

    control.clear(run)
    resumed = make_trainer(
        run, tiny_model(True, seed=7), _provider(), plan=PLAN, resume="must", **KW
    )
    resumed.fit()
    a, b = read_losses(tmp_path / "a"), read_losses(run)
    assert all(a[s] == b[s] for s in range(1, 11))
    pa, pb = full.raw_model.state_dict(), resumed.raw_model.state_dict()
    assert all(torch.equal(pa[k], pb[k]) for k in pa)


def test_after_stage_request_does_not_stop_the_trainer(tmp_path: Path) -> None:
    control.request_pause(tmp_path, "after_stage")
    make_trainer(tmp_path, tiny_model(False), _provider(), plan=PLAN, **KW).fit()
    assert max(read_losses(tmp_path)) == 10
    assert control.pending(tmp_path) == "after_stage"


def test_control_files(tmp_path: Path) -> None:
    assert control.pending(tmp_path) is None
    with pytest.raises(ValueError):
        control.request_pause(tmp_path, "later")  # type: ignore[arg-type]
    (tmp_path / control.CONTROL_FILE).write_text("{not json")
    assert control.pending(tmp_path) is None  # a torn file is no request
    with control.running(tmp_path, "p.yaml", ["p.yaml"]):
        info = control.process_info(tmp_path)
        assert info is not None and info["pid"] == os.getpid()
    assert control.process_info(tmp_path) is None
    assert not (tmp_path / control.PROCESS_FILE).exists()
    (tmp_path / control.PROCESS_FILE).write_text('{"pid": 999999999}')
    assert control.process_info(tmp_path) is None  # stale pid: not alive


def test_stage_paused_status(tmp_path: Path) -> None:
    log = RunLogger(tmp_path)
    log.stage_started("pretrain", {}, None)
    log.stage_paused("pretrain")
    entry = log.read_manifest()["stages"][0]
    assert entry["status"] == "paused" and "paused" in entry


def test_request_host() -> None:
    assert request_host("localhost:8765") == "localhost"
    assert request_host("[::1]:8765") == "::1"
    assert request_host("http://127.0.0.1:8765") == "127.0.0.1"
    assert request_host("evil.example") == "evil.example"


def _fake_stages(monkeypatch: pytest.MonkeyPatch, pause_in: str | None = None) -> list[str]:
    ran: list[str] = []

    def run_stage(cfg: Any, run_dir: Path) -> dict[str, Any]:
        log = RunLogger(run_dir)
        log.stage_started(cfg.stage.name, {}, None)
        ran.append(cfg.stage.name)
        if cfg.stage.name == pause_in:
            log.stage_paused(cfg.stage.name)
            raise control.StopRequested("paused")
        if cfg.stage.name == "midtrain":  # asked while midtrain runs: stop before sft
            control.request_pause(run_dir, "after_stage")
        log.stage_finished(cfg.stage.name, None, {})
        return {}

    monkeypatch.setattr(pipeline, "run_stage", run_stage)
    monkeypatch.setattr(pipeline.report, "write", lambda run_dir: {})
    monkeypatch.setattr(pipeline.report, "markdown", lambda rep: "")
    return ran


def test_pipeline_until_pause_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    smoke = Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "smoke.yaml"
    args = [f"run.out_dir={tmp_path}", "run.run_id=r"]
    ran = _fake_stages(monkeypatch)
    with pytest.raises(Exception, match="--until"):
        pipeline.run_pipeline(smoke, args, until="nope")
    run = pipeline.run_pipeline(smoke, args, until="pretrain")
    assert ran == ["pretrain"]
    m = RunLogger(run).read_manifest()
    assert m["pipeline_stages"][0] == "pretrain" and m["pipeline_args"] == args
    assert not (run / control.PROCESS_FILE).exists()
    pipeline.run_pipeline(smoke, args)  # midtrain asks to stop after itself
    assert ran == ["pretrain", "midtrain"]
    assert control.pending(run) is None  # acted on, then cleared

    ran = _fake_stages(monkeypatch, pause_in="dpo")
    pipeline.run_pipeline(smoke, args)
    assert ran == ["sft", "dpo"]
    status = {s["name"]: s["status"] for s in RunLogger(run).read_manifest()["stages"]}
    assert status["dpo"] == "paused"
    ran = _fake_stages(monkeypatch)
    pipeline.run_pipeline(smoke, args)
    assert ran == ["dpo", "grpo", "distill"]  # resumes at the paused stage


# --------------------------------------------------------------------- API
class FakeProc:
    def __init__(self, cmd: list[str], **kw: Any) -> None:
        self.cmd, self.kw, self.pid, self.code = cmd, kw, 4242, None

    def poll(self) -> int | None:
        return self.code


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, Controller, list[FakeProc]]:
    spawned: list[FakeProc] = []

    def spawn(cmd: list[str], **kw: Any) -> FakeProc:
        spawned.append(FakeProc(cmd, **kw))
        return spawned[-1]

    runs = tmp_path / "runs"
    logger = RunLogger(runs / "smoke-1")
    logger.init_manifest("smoke-1", "smoke", {}, {})
    logger.set_field("pipeline", "configs/pipeline/smoke.yaml")
    logger.set_field("pipeline_args", ["data=smoke", "run.run_id=smoke-1", "train.lr=0.01"])
    cfg = load_stage_config(["model=smoke", "stage=pretrain"])
    cfg.model.vocab_size = 300
    from dataclasses import asdict

    logger.stage_started("pretrain", asdict(cfg), None)
    logger.stage_paused("pretrain")
    ctl = Controller(runs, enabled=True, spawn=spawn)
    app = create_app(runs, controller=ctl, web_dist=tmp_path / "none")
    return TestClient(app, base_url="http://localhost:8765"), ctl, spawned


def test_control_api_resume_and_pause(api: tuple[TestClient, Controller, list[FakeProc]]) -> None:
    client, ctl, spawned = api
    info = client.get("/api/control").json()
    assert info["enabled"] and "smoke.yaml" in {p["file"] for p in info["pipelines"]}
    state = client.get("/api/runs/smoke-1/control").json()
    assert not state["running"] and state["pipeline"] == "smoke.yaml"
    assert client.post("/api/runs/smoke-1/control/pause", json={"when": "now"}).status_code == 409

    r = client.post("/api/runs/smoke-1/control/resume", json={"until": "sft"})
    assert r.status_code == 200, r.text
    cmd = spawned[0].cmd
    assert cmd[cmd.index("--until") + 1] == "sft"
    assert "train.lr=0.01" in cmd and "run.run_id=smoke-1" in cmd
    assert sum(a.startswith("run.run_id=") for a in cmd) == 1  # recorded id not duplicated
    assert client.get("/api/runs/smoke-1/control").json()["running"]
    # one pipeline at a time
    assert client.post("/api/control/start", json={"pipeline": "smoke.yaml"}).status_code == 409

    assert client.post("/api/runs/smoke-1/control/pause", json={"when": "now"}).status_code == 200
    assert control.pending(ctl.runs_dir / "smoke-1") == "now"
    assert client.post("/api/runs/smoke-1/control/cancel", json={}).status_code == 200
    assert control.pending(ctl.runs_dir / "smoke-1") is None

    spawned[0].code = 0  # the process exits: no longer running
    assert not client.get("/api/runs/smoke-1/control").json()["running"]
    r = client.post("/api/control/start", json={"pipeline": "smoke.yaml"})
    assert r.status_code == 200 and r.json()["run_id"].startswith("smoke-")


def test_control_api_refuses(
    api: tuple[TestClient, Controller, list[FakeProc]], tmp_path: Path
) -> None:
    client, ctl, spawned = api
    bad = [
        client.post("/api/control/start", json={"pipeline": "../../etc/x.yaml"}),
        client.post("/api/control/start", json={"pipeline": "smoke.yaml", "until": "nope"}),
        client.post("/api/runs/smoke-1/control/pause", json={"when": "soon"}),
    ]
    assert [r.status_code for r in bad] == [400, 400, 400]
    # DNS rebinding: a page on another name that resolves to us
    r = client.post(
        "/api/control/start", json={"pipeline": "smoke.yaml"}, headers={"host": "evil.example"}
    )
    assert r.status_code == 403
    r = client.post(
        "/api/control/start",
        json={"pipeline": "smoke.yaml"},
        headers={"origin": "http://evil.example"},
    )
    assert r.status_code == 403
    r = client.post(  # a cross-site form post cannot send JSON
        "/api/control/start",
        content='{"pipeline": "smoke.yaml"}',
        headers={"content-type": "text/plain"},
    )
    assert r.status_code == 415
    assert not spawned
    off = TestClient(
        create_app(ctl.runs_dir, web_dist=tmp_path / "none", control=False),
        base_url="http://localhost",
    )
    assert off.post("/api/control/start", json={"pipeline": "smoke.yaml"}).status_code == 403


def test_architecture_endpoint(api: tuple[TestClient, Controller, list[FakeProc]]) -> None:
    client, _, _ = api
    r = client.get("/api/runs/smoke-1/architecture")
    assert r.status_code == 200, r.text
    d = r.json()
    cfg = load_stage_config(["model=smoke"]).model
    cfg.vocab_size = 300
    assert d["counts"] == param_count(cfg).as_dict()
    assert sum(p["numel"] for p in d["params"]) == d["counts"]["total"]
    assert any(p["name"].endswith("ffn.w_gate") for p in d["params"]) == cfg.moe.enabled
    assert client.get("/api/runs/smoke-1/architecture?stage=sft").status_code == 404
    assert client.get("/api/runs/nope/architecture").status_code == 404
