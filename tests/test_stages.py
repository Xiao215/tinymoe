"""The smoke pipeline end to end (M4-M10): pre/mid-training, SFT, DPO, GRPO and distillation
through ``run_pipeline``, plus eval, snapshots, report and bench."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from helpers import RandomRows, cpu_runtime, make_trainer, read_losses, tiny_model

from tinylab.config import STAGE_ORDER, load_stage_config
from tinylab.data.tokenizer import Tokenizer
from tinylab.eval.evaluator import Evaluator, pass_at_k
from tinylab.eval.snapshots import Snapshotter
from tinylab.logging import RunLogger
from tinylab.pipeline import run_pipeline, stage_configs
from tinylab.train.loop import Hooks

PIPELINES = Path(__file__).resolve().parent.parent / "configs" / "pipeline"


def smoke_args(out: Path, data_cache: Path, *extra: str) -> list[str]:
    return [
        f"run.out_dir={out}",
        "run.run_id=t",
        f"data.cache_dir={data_cache}",
        "stages.pretrain.train.max_steps=80",
        "stages.midtrain.train.max_steps=20",
        "midtrain.context_extension.steps=4",
        "stages.sft.train.max_steps=40",
        "stages.dpo.train.max_steps=12",
        *extra,
    ]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory, data_cache: Path) -> Path:
    out = tmp_path_factory.mktemp("runs")
    return run_pipeline(PIPELINES / "smoke.yaml", smoke_args(out, data_cache))


def _records(run_dir: Path, stage: str) -> list[dict[str, Any]]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["stage"] == stage]


def test_pipeline_configs_compose() -> None:
    cfgs = stage_configs(PIPELINES / "smoke.yaml", ["stages.midtrain.train.lr=1e-4"])
    assert [c.stage.name for c in cfgs][:2] == ["pretrain", "midtrain"]
    assert cfgs[0].train.decay_frac == 0.0  # stable phase only; annealing is mid-training
    assert cfgs[1].train.decay_frac == 1.0 and cfgs[1].train.warmup_steps == 0
    assert cfgs[1].train.lr == 1e-4 and cfgs[0].train.lr != 1e-4
    assert cfgs[0].hardware.device == "cpu"


@pytest.mark.parametrize("pipeline", sorted(p.name for p in PIPELINES.glob("*.yaml")))
def test_every_pipeline_composes_with_exact_batches(pipeline: str) -> None:
    """All shipped pipelines load and validate, and every token budget splits into whole
    rows (derive_batch would refuse it at stage start, hours into a run)."""
    cfgs = stage_configs(PIPELINES / pipeline)
    assert [c.stage.name for c in cfgs] == list(STAGE_ORDER)
    for c in cfgs:
        t = c.train
        if t.tokens_per_step:
            assert t.tokens_per_step % t.seq_len == 0, (pipeline, c.stage.name)
        ce = c.midtrain.context_extension
        if c.stage.name == "midtrain" and ce.enabled and ce.tokens_per_step:
            assert ce.tokens_per_step % ce.seq_len == 0, pipeline


def test_manifest_lineage(smoke_run: Path) -> None:
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    stages = {s["name"]: s for s in manifest["stages"]}
    assert stages["pretrain"]["status"] == "done" and stages["midtrain"]["status"] == "done"
    assert stages["pretrain"]["init_from"] is None
    assert stages["midtrain"]["init_from"].endswith("pretrain/final.pt")
    assert (smoke_run / stages["midtrain"]["checkpoint"]).is_file()
    assert (smoke_run / "tokenizer.json").is_file()


def test_loss_decreases_on_every_source(smoke_run: Path) -> None:
    recs = [r for r in _records(smoke_run, "pretrain") if "loss/stories" in r]
    first, last = recs[0], recs[-1]
    for source in ("stories", "math"):
        assert last[f"loss/{source}"] < first[f"loss/{source}"] - 1.0, source
    mid = [r for r in _records(smoke_run, "midtrain") if "loss/math_hq" in r]
    assert mid, "mid-training logs the high-quality source separately"


def test_midtrain_anneals_and_extends(smoke_run: Path) -> None:
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    entry = next(s for s in manifest["stages"] if s["name"] == "midtrain")
    anneal_steps = entry["config"]["train"]["max_steps"]  # then context extension follows
    recs = [r for r in _records(smoke_run, "midtrain") if "lr_mult" in r]
    mults = [r["lr_mult"] for r in recs if r["step"] <= anneal_steps]
    assert len(mults) >= 2 and mults[0] > mults[-1]  # WSD decay phase
    ext_mults = [r["lr_mult"] for r in recs if r["step"] > anneal_steps]
    assert ext_mults and set(ext_mults) == {1.0}  # extension: constant schedule at its own lr
    ext = json.loads((smoke_run / "artifacts" / "context_extension.json").read_text())
    assert set(ext) == {"before", "rope_scaled", "after"}
    assert all("val_long/loss/stories" in v for v in ext.values())


def test_post_training_stages(smoke_run: Path) -> None:
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    stages = {s["name"]: s for s in manifest["stages"]}
    assert [s["name"] for s in manifest["stages"]] == [
        "pretrain",
        "midtrain",
        "sft",
        "dpo",
        "grpo",
        "distill",
    ]
    assert all(s["status"] == "done" for s in stages.values())
    for prev, stage in (("midtrain", "sft"), ("sft", "dpo"), ("dpo", "grpo"), ("grpo", "distill")):
        expected = (smoke_run / stages[prev]["checkpoint"]).resolve()  # that stage's final
        assert Path(stages[stage]["init_from"]).resolve() == expected, stage
    assert any("val/loss/sft" in r for r in _records(smoke_run, "sft"))
    assert any("dpo/accuracy" in r for r in _records(smoke_run, "dpo"))
    rl = [r for r in _records(smoke_run, "grpo") if "rl/reward" in r]
    assert rl and any("rl/clip_frac" in r for r in _records(smoke_run, "grpo"))
    calib = json.loads((smoke_run / "artifacts" / "calibration.json").read_text())
    assert calib["selected"]
    assert (smoke_run / "snapshots" / "grpo").is_dir()


def test_distill_comparison(smoke_run: Path) -> None:
    table = json.loads((smoke_run / "artifacts" / "distill_comparison.json").read_text())
    rows = {r["mode"]: r for r in table["rows"]}
    assert list(rows) == ["baseline", "offpolicy", "logit", "onpolicy"]
    assert rows["onpolicy"]["init"] == "offpolicy" and rows["logit"]["init"] == "student base"
    spans = [r["steps"] for r in table["rows"]]
    assert all(a[1] == b[0] for a, b in itertools.pairwise(spans))  # contiguous steps
    assert all("eval/math/pass1" in r["metrics"] for r in table["rows"])
    assert table["markdown"].count("\n") == len(rows) + 2  # header, rule, teacher, modes
    recs = _records(smoke_run, "distill")
    assert any("kd/kl" in r for r in recs) and any("kd/teacher_nll" in r for r in recs)
    assert any("distill/teacher/eval/math/pass1" in r for r in recs)
    gens = {
        json.loads(p.read_text()).get("model")
        for p in (smoke_run / "snapshots" / "distill").glob("*/generations.json")
    }
    assert {"teacher", "baseline", "onpolicy"} <= gens
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    summary = next(s for s in manifest["stages"] if s["name"] == "distill")["summary"]
    top = table["rows"][int(summary["distill/best_index"])]  # modes run in MODES order here
    assert summary["eval/math/pass1"] == top["metrics"]["eval/math/pass1"]


def test_eval_metrics_and_generations(smoke_run: Path) -> None:
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    summary = manifest["stages"][0]["summary"]
    for key in ("val/loss/stories", "eval/math/pass1", "eval/math/format", "eval/math/pass4"):
        assert key in summary
    assert any(k.startswith("eval/math/pass1/d") for k in summary)
    snaps = sorted((smoke_run / "snapshots" / "pretrain").iterdir())
    gens = json.loads((snaps[-1] / "generations.json").read_text())
    assert gens["kind"] == "base"
    assert {g["kind"] for g in gens["generations"]} >= {"story", "math"}


def test_snapshot_contents(smoke_run: Path) -> None:
    snap_dir = sorted((smoke_run / "snapshots" / "pretrain").iterdir())[-1]
    data = json.loads((snap_dir / "data.json").read_text())
    for key in ("model", "probes", "layers", "logit_lens", "weights", "embeddings", "moe"):
        assert key in data, key
    moe = data["moe"]
    n_exp = data["model"]["n_experts"]
    assert len(moe["load"][0]) == n_exp and len(moe["bias"][0]) == n_exp
    assert len(moe["specialization"]["counts"][0]) == len(moe["specialization"]["categories"])
    assert (snap_dir / "data.npz").is_file()


def test_report_written(smoke_run: Path) -> None:
    md = (smoke_run / "artifacts" / "report.md").read_text()
    assert "| pretrain |" in md and "| midtrain |" in md


def test_pipeline_rerun_skips_finished(smoke_run: Path, data_cache: Path) -> None:
    before = (smoke_run / "metrics.jsonl").read_text()
    run_pipeline(PIPELINES / "smoke.yaml", smoke_args(smoke_run.parent, data_cache))
    assert (smoke_run / "metrics.jsonl").read_text() == before


def test_pass_at_k() -> None:
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(4, 4, 2) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(5, 2, 2) == pytest.approx(1 - 3 / 10)


def test_hooks_leave_training_bit_identical(tmp_path: Path, tok: Tokenizer) -> None:
    """Snapshots and evals run inside the loop but never change the training numbers."""
    cfg = load_stage_config(
        [
            "data=smoke",
            "train.seq_len=16",
            "snapshot.max_probe_tokens=12",
            "eval.max_new_tokens=4",
            "eval.math_per_bucket=1",
            "eval.story_prompts=0",
            "eval.probe_max_new_tokens=4",
        ]
    )
    rt = cpu_runtime()

    def train(with_hooks: bool, out: Path) -> dict[int, float]:
        model = tiny_model(True, vocab_size=tok.vocab_size, max_seq_len=128)
        logger = RunLogger(out)
        hooks = None
        if with_hooks:
            snap = Snapshotter(cfg, tok, rt, logger, "test")
            ev = Evaluator(cfg, tok, rt, logger, "test")
            hooks = Hooks(
                evaluate=lambda step: ev.run(model, step, with_lm=False),
                snapshot=lambda step: snap(model, step, trainer.recent_load),
                snapshot_interval=3,
            )
        trainer = make_trainer(
            out,
            model,
            _Rows(RandomRows(16, tok.vocab_size)),
            hooks=hooks,
            max_steps=9,
            eval_interval=4,
        )
        trainer.fit()
        return read_losses(out)

    plain = train(False, tmp_path / "plain")
    hooked = train(True, tmp_path / "hooked")
    assert plain == hooked
    assert any((tmp_path / "hooked" / "snapshots" / "test").iterdir())


class _Rows:
    """Adapts a row-indexed source to the (step, micro) provider protocol."""

    def __init__(self, source: RandomRows) -> None:
        self.source = source

    def batch(self, step: int, micro: int) -> Any:
        return self.source.batch(range(step * 4, step * 4 + 4))

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


def test_bench_smoke(tmp_path: Path) -> None:
    from tinylab.tools import bench

    res = bench.run(PIPELINES / "smoke.yaml", ["bench.steps=2", "bench.warmup=1"])
    names = [s["stage"] for s in res["stages"]]
    assert names[:3] == ["pretrain", "midtrain/anneal", "midtrain/context"]
    assert all(s["step_s"] > 0 and s["est_min"] > 0 for s in res["stages"][:3])
    assert res["fits"]
    assert "total estimate" in bench.markdown(res)
