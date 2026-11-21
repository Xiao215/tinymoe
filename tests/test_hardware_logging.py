from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tinylab import hardware
from tinylab.cli import main
from tinylab.config import HardwareConfig
from tinylab.logging import RunLogger


def test_cpu_capabilities() -> None:
    caps = hardware.detect(HardwareConfig(device="cpu"))
    assert caps.device_type == "cpu"
    assert "math" in caps.sdpa_backends


def test_forced_fp32_and_no_compile() -> None:
    rt = hardware.resolve_runtime(HardwareConfig(device="cpu", precision="fp32", compile="off"))
    assert rt.precision == "fp32" and rt.amp_dtype is None and not rt.compile
    assert not rt.use_loss_scaling


def test_auto_precision_on_cpu_is_fp32() -> None:
    rt = hardware.resolve_runtime(HardwareConfig(device="cpu", precision="auto"))
    assert rt.precision == "fp32"


def test_requested_bf16_on_cpu() -> None:
    rt = hardware.resolve_runtime(HardwareConfig(device="cpu", precision="bf16"))
    # CPU autocast supports bf16; if a build lacks it we must fall back, never crash
    assert rt.precision in ("bf16", "fp32")
    assert any("precision=" in d for d in rt.decisions)


def test_sdpa_forced_math() -> None:
    rt = hardware.resolve_runtime(HardwareConfig(device="cpu", sdpa_backend="math"))
    assert rt.sdpa_backend == "math"
    q = torch.randn(1, 2, 4, 8)
    with rt.sdpa_context():
        torch.nn.functional.scaled_dot_product_attention(q, q, q)


def test_unavailable_sdpa_falls_back() -> None:
    rt = hardware.resolve_runtime(HardwareConfig(device="cpu", sdpa_backend="cudnn"))
    assert rt.sdpa_backend in ("cudnn", "math")


def test_compile_on_with_eager_backend() -> None:
    rt = hardware.resolve_runtime(
        HardwareConfig(device="cpu", compile="on", compile_backend="eager")
    )
    assert rt.compile


def test_compile_bad_backend_falls_back() -> None:
    rt = hardware.resolve_runtime(
        HardwareConfig(device="cpu", compile="on", compile_backend="no_such_backend")
    )
    assert not rt.compile
    assert any("fallback" in d for d in rt.decisions)


def test_missing_device_raises() -> None:
    if hardware.accelerator_types():
        pytest.skip("machine has an accelerator")
    with pytest.raises(RuntimeError):
        hardware.resolve_device("mps")


@pytest.mark.skipif(not hardware.accelerator_types(), reason="needs an accelerator")
def test_accelerator_precision_paths() -> None:
    kind = hardware.accelerator_types()[0]
    for precision in ("fp32", "fp16", "bf16", "auto"):
        rt = hardware.resolve_runtime(HardwareConfig(device=kind, precision=precision))
        assert rt.device.type == kind
        assert rt.precision in ("fp32", "fp16", "bf16")
        assert rt.use_loss_scaling == (rt.precision == "fp16")


def test_rng_preserved() -> None:
    hardware.seed_everything(3)
    with hardware.preserved_rng():
        torch.rand(5)
        np.random.rand(3)
        random.random()
    a = (torch.rand(1).item(), np.random.rand(), random.random())
    hardware.seed_everything(3)
    b = (torch.rand(1).item(), np.random.rand(), random.random())
    assert a == b


def test_run_logger(tmp_path: Path) -> None:
    lg = RunLogger(tmp_path)
    lg.init_manifest("r1", "r", {"device": "cpu"}, {"precision": "fp32"})
    lg.stage_started("pretrain", {"a": 1}, init_from=None)
    lg.log_metrics("pretrain", 1, {"loss": 2.0, "bad": float("nan")})
    ckpt = tmp_path / "checkpoints" / "pretrain" / "final.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    lg.stage_finished("pretrain", ckpt, {"val_loss": 1.0})
    lg.save_snapshot("pretrain", 10, {"x": [1, 2]}, {"arr": np.zeros((2, 2))})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"][0]["checkpoint"] == "checkpoints/pretrain/final.pt"
    assert lg.checkpoint_of("pretrain") == tmp_path / "checkpoints/pretrain/final.pt"
    assert lg.last_finished_stage("sft", ("pretrain", "midtrain", "sft")) == "pretrain"
    line = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert line["loss"] == 2.0 and line["bad"] is None
    snap = tmp_path / "snapshots" / "pretrain" / "00000010"
    assert json.loads((snap / "data.json").read_text())["x"] == [1, 2]
    assert np.load(snap / "arrays.npz")["arr"].shape == (2, 2)


def test_disabled_logger_writes_nothing(tmp_path: Path) -> None:
    lg = RunLogger(tmp_path / "r", enabled=False)
    lg.log_metrics("s", 1, {"a": 1})
    assert not (tmp_path / "r").exists()


def test_info_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "hardware=cpu_smoke"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "cpu" in out["backends"]
    assert out["runtime"]["precision"] == "fp32"
