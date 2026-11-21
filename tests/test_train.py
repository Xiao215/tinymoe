from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from helpers import (
    FixedBatch,
    RandomRows,
    cpu_runtime,
    make_trainer,
    read_losses,
    tiny_model,
)

from tinymoe import hardware
from tinymoe.config import BatchPlan, HardwareConfig, TrainConfig
from tinymoe.data.batch import Batch
from tinymoe.train.autobatch import find_micro_batch
from tinymoe.train.loop import Hooks
from tinymoe.train.losses import LossOutput, lm_loss
from tinymoe.train.optim import build_optimizer, classify, newton_schulz
from tinymoe.train.providers import RowProvider
from tinymoe.train.schedule import lr_multiplier


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_overfit_tiny_batch(tmp_path: Path, moe: bool, optimizer: str) -> None:
    """Memorise one random batch (identical contexts never get conflicting labels)."""
    model = tiny_model(moe)
    seq = torch.randint(0, 61, (4, 17), generator=torch.Generator().manual_seed(1))
    batch = Batch(input_ids=seq[:, :-1], labels=seq[:, 1:])
    trainer = make_trainer(
        tmp_path,
        model,
        FixedBatch(batch),
        max_steps=150,
        optimizer=optimizer,
        lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
    )
    trainer.fit()
    losses = read_losses(tmp_path)
    assert losses[1] > 3.5  # ~ln(61)
    assert losses[150] < 0.05, losses[150]


def test_resume_is_bit_identical(tmp_path: Path) -> None:
    kw = {
        "max_steps": 12,
        "optimizer": "muon",
        "schedule": "wsd",
        "warmup_steps": 3,
        "decay_frac": 0.5,
    }
    full = make_trainer(
        tmp_path / "a",
        tiny_model(True),
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        plan=_plan(),
        **kw,
    )
    full.fit()

    class CrashError(RuntimeError):
        pass

    def crash(step: int) -> None:
        raise CrashError

    hooks = Hooks(snapshot=crash, snapshot_interval=7)
    part = make_trainer(
        tmp_path / "b",
        tiny_model(True),
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        hooks=hooks,
        ckpt_interval=6,
        plan=_plan(),
        **kw,
    )
    with pytest.raises(CrashError):
        part.fit()
    resumed = make_trainer(
        tmp_path / "b",
        tiny_model(True, seed=99),
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        ckpt_interval=6,
        resume="must",
        plan=_plan(),
        **kw,
    )
    resumed.fit()
    a, b = read_losses(tmp_path / "a"), read_losses(tmp_path / "b")
    assert all(a[s] == b[s] for s in range(1, 13)), (a, b)
    pa = full.raw_model.state_dict()
    pb = resumed.raw_model.state_dict()
    assert all(torch.equal(pa[k], pb[k]) for k in pa)


def _plan() -> BatchPlan:
    return BatchPlan(micro_batch_size=2, grad_accum=2, world_size=1, unit=16)


def _cpu() -> torch.device:
    return torch.device("cpu")


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_precision_paths_cpu(tmp_path: Path, precision: str) -> None:
    rt = cpu_runtime(precision)
    t = make_trainer(
        tmp_path,
        tiny_model(True),
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        runtime=rt,
        max_steps=30,
    )
    t.fit()
    losses = read_losses(tmp_path)
    assert losses[30] < losses[1] - 0.5
    if rt.precision == "fp16":
        assert "loss_scale" in json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])


@pytest.mark.skipif(not hardware.accelerator_types(), reason="needs an accelerator")
@pytest.mark.parametrize("precision", ["fp32", "fp16", "auto"])
def test_precision_paths_accelerator(tmp_path: Path, precision: str) -> None:
    kind = hardware.accelerator_types()[0]
    rt = hardware.resolve_runtime(HardwareConfig(device=kind, precision=precision))  # type: ignore[arg-type]
    model = tiny_model(True).to(rt.device)
    t = make_trainer(
        tmp_path,
        model,
        RowProvider(RandomRows(16), _plan(), 0, rt.device),
        runtime=rt,
        max_steps=30,
    )
    t.fit()
    losses = read_losses(tmp_path)
    assert losses[30] < losses[1] - 0.5


def test_loss_scaler_skips_non_finite(tmp_path: Path) -> None:
    rt = cpu_runtime("fp16")
    if rt.precision != "fp16":
        pytest.skip("CPU autocast lacks fp16 on this build")
    calls = {"n": 0}

    def loss_fn(m: torch.nn.Module, b: object) -> LossOutput:
        out = lm_loss(m, b)  # type: ignore[arg-type]
        calls["n"] += 1
        if calls["n"] == 3:
            out.loss = out.loss * float("inf")
        return out

    model = tiny_model(False)
    t = make_trainer(
        tmp_path,
        model,
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        runtime=rt,
        plan=_plan(),
        loss_fn=loss_fn,
        max_steps=4,
    )
    t.fit()
    recs = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    skipped = [r["step"] for r in recs if r["skipped"]]
    assert skipped == [2]  # micro call 3 belongs to step 2 (accum=2)
    assert recs[1]["loss_scale"] == recs[0]["loss_scale"] / 2


def test_compile_path(tmp_path: Path) -> None:
    rt = cpu_runtime("fp32", "on", "eager")
    assert rt.compile
    t = make_trainer(
        tmp_path,
        tiny_model(False),
        RowProvider(RandomRows(16), _plan(), 0, _cpu()),
        runtime=rt,
        max_steps=10,
    )
    t.fit()
    losses = read_losses(tmp_path)
    assert losses[10] < losses[1]


def test_newton_schulz_orthogonalises() -> None:
    torch.manual_seed(0)
    g = torch.randn(3, 24, 40)
    x = newton_schulz(g, 5, torch.float32)
    sv = torch.linalg.svdvals(x)
    assert sv.min() > 0.6 and sv.max() < 1.3
    torch.testing.assert_close(x[1], newton_schulz(g[1], 5, torch.float32))


def test_param_classification() -> None:
    model = tiny_model(True)
    roles = {n: classify(n, p) for n, p in model.named_parameters()}
    assert roles["embed.weight"] == "embed"
    assert roles["layers.1.ffn.router.weight"] == "no_decay"
    assert roles["layers.0.attn_norm.weight"] == "no_decay"
    assert roles["layers.1.ffn.w_gate"] == "muon" and roles["layers.0.attn.q_proj.weight"] == "muon"
    opt = build_optimizer(model, TrainConfig(optimizer="muon"), cpu_runtime())
    assert len(opt.optimizers) == 2
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in model.parameters())


def test_schedules() -> None:
    wsd = TrainConfig(
        schedule="wsd", warmup_steps=10, max_steps=100, decay_frac=0.2, min_lr_ratio=0.1
    )
    mults = [lr_multiplier(s, wsd) for s in range(100)]
    assert mults[0] == pytest.approx(0.1) and mults[9] == 1.0
    assert all(m == 1.0 for m in mults[10:80])
    assert mults[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in itertools.pairwise(mults[80:]))
    cos = TrainConfig(schedule="cosine", warmup_steps=0, max_steps=50, min_lr_ratio=0.0)
    assert lr_multiplier(0, cos) == pytest.approx(1.0) and lr_multiplier(49, cos) < 0.01
    decay_only = TrainConfig(
        schedule="wsd", warmup_steps=0, max_steps=10, decay_frac=1.0, decay_shape="linear"
    )
    assert lr_multiplier(0, decay_only) < 1.0


def test_autobatch_cpu() -> None:
    model = tiny_model(False)

    def step(size: int) -> None:
        if size > 6:
            raise RuntimeError("CPU out of memory (simulated)")
        x = torch.randint(0, 61, (size, 16))
        model(x).logits.sum().backward()

    assert find_micro_batch(step, 64, torch.device("cpu"), None) == 6
    with pytest.raises(RuntimeError, match="boom"):
        find_micro_batch(lambda s: (_ for _ in ()).throw(RuntimeError("boom")), 8, _cpu(), None)


@pytest.mark.distributed
def test_ddp_gloo_two_processes_match_single(tmp_path: Path) -> None:
    """2-rank DDP (gloo, CPU) == 1 process with 2x gradient accumulation, bit-for-bit close."""
    worker = Path(__file__).parent / "ddp_worker.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent), "OMP_NUM_THREADS": "1"}
    single = tmp_path / "single.json"
    subprocess.run([sys.executable, str(worker), str(single)], check=True, env=env, timeout=300)
    multi = tmp_path / "multi.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29531",
            str(worker),
            str(multi),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    a, b = json.loads(single.read_text()), json.loads(multi.read_text())
    assert b["world_size"] == 2 and a["world_size"] == 1
    assert b["ranks_agree"]
    for x, y in zip(a["losses"], b["losses"], strict=True):
        assert abs(x - y) < 1e-5, (a["losses"], b["losses"])
    assert abs(a["param_sum"] - b["param_sum"]) < 1e-4


def test_autobatch_never_tries_projected_overruns(monkeypatch: pytest.MonkeyPatch) -> None:
    """With measurable memory, sizes whose linear projection exceeds the budget are skipped
    (on unified memory an oversized trial would swap instead of raising OOM)."""
    from tinymoe import hardware

    peak = {"bytes": 0.0}
    tried: list[int] = []
    monkeypatch.setattr(hardware, "has_memory_stats", lambda device: True)
    monkeypatch.setattr(
        hardware, "memory_bytes", lambda device: {"allocated": 0, "peak": int(peak["bytes"])}
    )

    def step(size: int) -> None:
        tried.append(size)
        peak["bytes"] = 1e9 + size * 1e9  # 1 GB fixed + 1 GB per row

    assert find_micro_batch(step, 64, _cpu(), budget_gb=5.0) == 4
    assert max(tried) == 4 and 8 not in tried and 5 not in tried
