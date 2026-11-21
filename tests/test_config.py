from __future__ import annotations

from pathlib import Path

import pytest

from tinymoe.config import (
    ConfigError,
    RunConfig,
    compose,
    derive_batch,
    from_dict,
    load_stage_config,
    parse_override,
    split_overrides,
    to_dict,
    yaml_load,
)


def test_defaults_validate() -> None:
    cfg = from_dict({})
    assert isinstance(cfg, RunConfig)
    assert cfg.train.lr > 0


def test_unknown_key_suggests() -> None:
    with pytest.raises(ConfigError, match="did you mean 'lr'"):
        from_dict({"train": {"lrr": 1.0}})


def test_type_errors() -> None:
    with pytest.raises(ConfigError, match="expected int"):
        from_dict({"train": {"max_steps": "ten"}})
    with pytest.raises(ConfigError, match="expected bool"):
        from_dict({"model": {"qk_norm": 1}})
    with pytest.raises(ConfigError, match="not one of"):
        from_dict({"train": {"optimizer": "sgd"}})
    # ints are accepted where floats are expected, but bools are not
    assert from_dict({"train": {"lr": 1}}).train.lr == 1.0
    with pytest.raises(ConfigError):
        from_dict({"train": {"lr": True}})


def test_yaml_12_scalars_for_overrides() -> None:
    """``on``/``off``/``yes``/``no`` stay strings (YAML 1.1 made them booleans, which broke
    the documented ``hardware.compile=off``); ``3e-4`` is a float."""
    assert [yaml_load(s) for s in ("off", "on", "yes", "no")] == ["off", "on", "yes", "no"]
    assert [yaml_load(s) for s in ("true", "False", "3e-4")] == [True, False, 3e-4]
    cfg = load_stage_config(["hardware.compile=off", "hardware.enforce_memory_budget=false"])
    assert cfg.hardware.compile == "off" and cfg.hardware.enforce_memory_budget is False


def test_union_auto_batch() -> None:
    assert from_dict({"train": {"micro_batch_size": "auto"}}).train.micro_batch_size == "auto"
    assert from_dict({"train": {"micro_batch_size": 4}}).train.micro_batch_size == 4
    with pytest.raises(ConfigError):
        from_dict({"train": {"micro_batch_size": "big"}})


def test_cross_field_validation_collects_all() -> None:
    with pytest.raises(ConfigError) as info:
        from_dict({"model": {"d_model": 100, "n_heads": 3}, "train": {"lr": -1.0}})
    msg = str(info.value)
    assert "divisible by n_heads" in msg and "train.lr" in msg


def test_moe_validation() -> None:
    with pytest.raises(ConfigError, match="top_k"):
        from_dict({"model": {"moe": {"enabled": True, "n_experts": 2, "top_k": 3}}})


def test_seq_len_vs_model() -> None:
    with pytest.raises(ConfigError, match="max_seq_len"):
        from_dict({"model": {"max_seq_len": 64}, "train": {"seq_len": 128}})


def test_parse_override_yaml_values() -> None:
    assert parse_override("train.lr=3e-3") == (["train", "lr"], 3e-3)
    assert parse_override("model.moe.enabled=true") == (["model", "moe", "enabled"], True)
    assert parse_override("train.betas=[0.9,0.99]") == (["train", "betas"], [0.9, 0.99])
    with pytest.raises(ConfigError):
        parse_override("novalue")


def test_split_groups_and_dotted() -> None:
    groups, dotted = split_overrides(["model=smoke", "train.lr=0.01", "train.max_steps=3"])
    assert groups == {"model": "smoke"}
    assert dotted == {"train": {"lr": 0.01, "max_steps": 3}}


def test_compose_order(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "stage").mkdir()
    (tmp_path / "model" / "m.yaml").write_text("d_model: 64\nn_heads: 4\n")
    (tmp_path / "stage" / "s.yaml").write_text("stage: {name: sft}\ntrain: {lr: 0.5}\n")
    cfg = compose({"model": "m", "stage": "s"}, {"train": {"lr": 0.25}}, root=tmp_path)
    assert cfg.model.d_model == 64
    assert cfg.stage.name == "sft"
    assert cfg.train.lr == 0.25  # CLI-level partial wins over the stage recipe


def test_student_group_loads_model_file(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "tiny.yaml").write_text("d_model: 32\nn_heads: 2\nn_kv_heads: 1\n")
    cfg = compose({"student": "tiny"}, root=tmp_path)
    assert cfg.student.d_model == 32 and cfg.model.d_model == 128


def test_repo_configs_load() -> None:
    for hw in ("auto", "cpu_smoke", "mac_m1_16gb", "cuda_24gb", "cuda_80gb"):
        cfg = load_stage_config([f"hardware={hw}"])
        assert cfg.hardware.device


def test_roundtrip_to_dict() -> None:
    cfg = from_dict({"model": {"moe": {"enabled": True}}})
    again = from_dict(to_dict(cfg))
    assert again == cfg


@pytest.mark.parametrize(
    ("units", "unit", "mbs", "world", "expect_mbs", "expect_accum"),
    [
        (4096, 128, 8, 1, 8, 4),
        (4096, 128, 8, 2, 8, 2),
        (4096, 128, 64, 1, 32, 1),  # micro batch larger than the global batch
        (32 * 64, 64, 12, 1, 8, 4),  # 12 does not divide 32 -> lowered to 8
        (16, 1, 4, 4, 4, 1),  # example-based stages: unit=1
    ],
)
def test_batch_math(
    units: int, unit: int, mbs: int, world: int, expect_mbs: int, expect_accum: int
) -> None:
    plan = derive_batch(units, unit, mbs, world)
    assert (plan.micro_batch_size, plan.grad_accum) == (expect_mbs, expect_accum)
    assert plan.units_per_step == units  # the global batch is always exact


def test_batch_math_errors() -> None:
    with pytest.raises(ConfigError):
        derive_batch(1000, 128, 8, 1)  # not a multiple of seq len
    with pytest.raises(ConfigError):
        derive_batch(3 * 128, 128, 1, 2)  # rows not divisible by world size
