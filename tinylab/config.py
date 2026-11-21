"""Typed configuration: dataclass schema, YAML group files, dot-overrides, validation.

Why dataclasses + YAML rather than OmegaConf/Hydra: the schema is the single source of
truth, ``mypy --strict`` can check every access, and composition rules stay small and
explicit (see ``docs/decisions.md``).

Composition order (later wins)::

    dataclass defaults
      <- group files   hardware=<name>  model=<name>  student=<name>  data=<name>
      <- stage recipe  stage=<name>     (root-level partial: may touch train/eval/...)
      <- pipeline-level overrides, then per-stage overrides
      <- CLI dot-overrides   train.lr=3e-3   moe.top_k=2 ...

Group files for ``hardware``/``model``/``data`` are *section scoped*: the YAML content
is merged into that section. ``student=<name>`` loads a file from ``configs/model`` into
the ``student`` section. Stage recipes (``configs/stage``) are root-level partials.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
import os
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types and cross-field inconsistencies."""


# ----------------------------------------------------------------------- yaml loader


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``3e-4`` as a float (YAML 1.1 requires ``3.0e-4``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


# --------------------------------------------------------------------------- schema


@dataclass
class RunMeta:
    """Identity and output location of a run."""

    name: str = "run"
    out_dir: str = "runs"
    run_id: str | None = None  # None -> "<name>-<timestamp>"
    seed: int = 0


@dataclass
class HardwareConfig:
    """Hardware policy. Everything is *requested* here and *resolved* in hardware.py."""

    device: str = "auto"  # auto | cpu | <accelerator>[:index]
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    precision_bench: bool = True  # auto: benchmark low-precision matmul vs fp32
    low_precision_min_speedup: float = 1.15
    compile: Literal["auto", "on", "off"] = "auto"
    compile_backend: str = "inductor"
    fused_optimizer: Literal["auto", "on", "off"] = "auto"
    sdpa_backend: Literal["auto", "math", "flash", "efficient", "cudnn"] = "auto"
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    num_threads: int | None = None
    memory_budget_gb: float | None = None  # used by bench + micro_batch_size=auto
    peak_tflops: float | None = None  # for MFU; None -> measured matmul peak
    dist_backend: Literal["auto", "nccl", "gloo"] = "auto"
    deterministic: bool = False


@dataclass
class RopeScalingConfig:
    """Context-extension scaling of RoPE (none | ntk base scaling | linear PI | YaRN)."""

    type: Literal["none", "ntk", "linear", "yarn"] = "none"
    factor: float = 1.0
    original_max_seq_len: int | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0


@dataclass
class MoEConfig:
    """DeepSeek-V3/GLM-style fine-grained MoE with shared experts."""

    enabled: bool = False
    n_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 1
    expert_hidden: int = 128
    n_dense_layers: int = 1
    score_func: Literal["sigmoid", "softmax"] = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling: float = 1.0
    balance: Literal["aux_free", "switch", "none"] = "aux_free"
    bias_update_rate: float = 1e-3
    seq_aux_weight: float = 1e-4
    switch_aux_weight: float = 1e-2
    z_loss_weight: float = 0.0
    capacity_factor: float | None = None  # None -> no token dropping
    dispatch: Literal["auto", "loop", "padded"] = "auto"  # see model/moe.py
    router_init_std: float | None = None  # None -> model.init_std


@dataclass
class MTPConfig:
    """Multi-token prediction (DeepSeek-V3). depth=0 disables it."""

    depth: int = 0
    loss_weight: float = 0.3


@dataclass
class ModelConfig:
    """Decoder-only transformer hyper-parameters."""

    vocab_size: int = 0  # 0 -> taken from the tokenizer at build time
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int | None = None  # None -> d_model // n_heads
    ffn_hidden: int = 384
    max_seq_len: int = 256
    rope_base: float = 10000.0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02
    sliding_window: int | None = None
    global_every: int = 1  # with sliding_window: every Nth layer is global
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)


@dataclass
class TokenizerConfig:
    vocab_size: int = 8192
    train_docs_per_source: int = 20000
    min_frequency: int = 2


@dataclass
class StoriesConfig:
    """TinyStories (Eldan & Li, 2023) or an offline synthetic stand-in."""

    source: Literal["hf", "synthetic"] = "hf"
    hf_path: str = "roneneldan/TinyStories"
    hf_train_split: str = "train"
    hf_val_split: str = "validation"
    num_train_docs: int = 100000
    num_val_docs: int = 2000
    max_chars: int = 2000
    fallback_to_synthetic: bool = True


@dataclass
class MathWorldConfig:
    """Synthetic arithmetic world. Difficulty = digits x steps (x ops)."""

    num_train_docs: int = 100000
    num_val_docs: int = 2000
    digits: list[int] = field(default_factory=lambda: [1, 2, 3])
    steps: list[int] = field(default_factory=lambda: [1, 2, 3])
    ops: list[str] = field(default_factory=lambda: ["+", "-", "*"])
    kind_weights: dict[str, float] = field(
        default_factory=lambda: {"arithmetic": 0.4, "word": 0.3, "facts": 0.15, "algebra": 0.15}
    )
    hq_min_steps: int = 2  # "high quality" (mid-training) docs use at least this many steps
    max_mul_operand: int = 12  # keep products learnable for tiny models


@dataclass
class DataConfig:
    cache_dir: str = "data_cache"
    seed: int = 1234
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    stories: StoriesConfig = field(default_factory=StoriesConfig)
    mathworld: MathWorldConfig = field(default_factory=MathWorldConfig)
    pretrain_mix: dict[str, float] = field(default_factory=lambda: {"stories": 0.6, "math": 0.4})
    midtrain_mix: dict[str, float] = field(
        default_factory=lambda: {"stories": 0.3, "math": 0.2, "math_hq": 0.5}
    )
    doc_mask: bool = False  # document-boundary-aware attention + per-doc positions
    sft_val_fraction: float = 0.05


@dataclass
class MuonConfig:
    """Muon (Jordan et al., 2024; Moonlight/Kimi scaling, Liu et al., 2025)."""

    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    scale: Literal["moonlight", "keller"] = "moonlight"
    ns_dtype: Literal["auto", "fp32", "bf16"] = "auto"
    lr: float | None = None  # None -> train.lr (moonlight scaling makes them comparable)


@dataclass
class TrainConfig:
    """Optimisation + loop settings shared by every stage."""

    optimizer: Literal["muon", "adamw"] = "adamw"
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.1
    decay_embeddings: bool = False
    grad_clip: float = 1.0
    z_loss_weight: float = 0.0  # PaLM output-logit z-loss
    schedule: Literal["wsd", "cosine", "constant"] = "wsd"
    warmup_steps: int = 100
    decay_frac: float = 0.2
    decay_shape: Literal["linear", "sqrt", "cosine"] = "sqrt"
    max_steps: int = 1000
    seq_len: int = 256
    tokens_per_step: int | None = None  # token-based stages
    examples_per_step: int | None = None  # example-based stages (DPO, GRPO)
    micro_batch_size: int | Literal["auto"] = 8
    auto_batch_max: int = 256
    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 8
    ckpt_interval: int = 500
    keep_ckpts: int = 2
    resume: Literal["auto", "never", "must"] = "auto"
    muon: MuonConfig = field(default_factory=MuonConfig)


@dataclass
class EvalConfig:
    math_per_bucket: int = 20
    passk_k: int = 4
    passk_samples: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 128
    story_prompts: int = 8
    story_max_new_tokens: int = 160
    story_min_words: int = 20
    story_max_words: int = 250
    lm_batches: int = 8
    probe_math: int = 3
    probe_stories: list[str] = field(
        default_factory=lambda: ["Once upon a time, there was a little cat", "Tom and Lily went"]
    )
    probe_instructions: list[str] = field(
        default_factory=lambda: ["Write a short story about a dog."]
    )
    probe_max_new_tokens: int = 96
    seed: int = 4242


@dataclass
class SnapshotConfig:
    enabled: bool = True
    interval: int = 200
    max_probe_tokens: int = 48
    attention_max_tokens: int = 32
    logit_lens_topk: int = 5
    logit_lens_positions: int = 6
    embedding_tokens: list[str] = field(
        default_factory=lambda: [
            *[str(d) for d in range(10)],
            "+", "-", "*", "=", " the", " cat", " dog", " happy", " sad", " big", " small",
            " girl", " boy", " Answer",
        ]
    )  # fmt: skip
    generations: bool = True


@dataclass
class StageConfig:
    name: Literal["pretrain", "midtrain", "sft", "dpo", "grpo", "distill"] = "pretrain"
    init_from: str = "auto"  # auto = previous stage in manifest | <stage name> | scratch


@dataclass
class ContextExtensionConfig:
    enabled: bool = True
    seq_len: int = 1024
    rope_scaling: RopeScalingConfig = field(
        default_factory=lambda: RopeScalingConfig(type="ntk", factor=4.0)
    )
    steps: int = 200
    lr: float = 1e-4
    tokens_per_step: int | None = None
    micro_batch_size: int = 2
    eval_batches: int = 4


@dataclass
class MidtrainConfig:
    context_extension: ContextExtensionConfig = field(default_factory=ContextExtensionConfig)


@dataclass
class LoRAConfig:
    """LoRA (Hu et al., 2021) on dense attention/MLP projections."""

    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class SFTConfig:
    n_math: int = 4000
    n_story: int = 1000
    think_traces: bool = True
    packing: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class DPOConfig:
    """DPO (Rafailov et al., 2023), IPO (Azar et al., 2023), SimPO (Meng et al., 2024)."""

    loss: Literal["dpo", "ipo", "simpo"] = "dpo"
    beta: float = 0.1
    simpo_gamma: float = 0.5
    label_smoothing: float = 0.0
    n_pairs: int = 2000
    rejected_kinds: dict[str, float] = field(
        default_factory=lambda: {"wrong_answer": 0.5, "no_format": 0.25, "rambling": 0.25}
    )
    model_sampled_prompts: int = 0  # >0: also mine pairs from SFT samples via the verifier
    samples_per_prompt: int = 4
    story_pairs_fraction: float = 0.0


@dataclass
class OverlongConfig:
    """DAPO soft overlong punishment."""

    enabled: bool = False
    cache_len: int = 16  # penalty ramps linearly over the last cache_len tokens
    penalty: float = 1.0


@dataclass
class IcePopConfig:
    """Mask tokens whose train/rollout probability ratio leaves [low, high] (GLM-5 IcePop)."""

    enabled: bool = False
    low: float = 0.5
    high: float = 2.0


@dataclass
class CalibrationConfig:
    enabled: bool = True
    samples_per_bucket: int = 16
    low: float = 0.1
    high: float = 0.6
    max_buckets: int = 4


@dataclass
class GRPOConfig:
    """GRPO (Shao et al., 2024), Dr. GRPO (Liu et al., 2025), DAPO (Yu et al., 2025)."""

    variant: Literal["grpo", "dr_grpo", "dapo", "custom"] = "grpo"
    group_size: int = 8
    prompts_per_rollout: int = 16
    ppo_epochs: int = 2
    minibatches_per_epoch: int = 2
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 96
    # loss knobs (variant presets fill these unless variant == "custom")
    std_normalize: bool = True
    loss_agg: Literal["seq_mean", "token_mean", "constant"] = "seq_mean"
    clip_low: float = 0.2
    clip_high: float = 0.2
    dynamic_sampling: bool = False
    kl_coef: float = 0.02
    reward_correct: float = 1.0
    reward_format: float = 0.2
    overlong: OverlongConfig = field(default_factory=OverlongConfig)
    routing_replay: bool = True
    icepop: IcePopConfig = field(default_factory=IcePopConfig)
    moe_balance_scale: float = 0.1  # MoE balancing stays on but at reduced weight
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    max_resample_rounds: int = 3


@dataclass
class DistillConfig:
    """Teacher -> student: off-policy (seq-level), logit KD, on-policy reverse KL."""

    modes: list[str] = field(default_factory=lambda: ["baseline", "offpolicy", "logit", "onpolicy"])
    student_pretrain_steps: int = 300
    student_lr: float = 1e-3
    teacher_prompts: int = 1000
    teacher_samples_per_prompt: int = 4
    kd_temperature: float = 1.0
    kd_ce_weight: float = 0.0  # mix hard-label CE into logit KD
    onpolicy_steps: int = 100
    onpolicy_prompts_per_step: int = 16
    onpolicy_lr: float = 2e-4
    onpolicy_max_new_tokens: int = 96
    onpolicy_temperature: float = 1.0


@dataclass
class BenchConfig:
    steps: int = 6
    warmup: int = 2
    snapshot_overhead: bool = True
    generation_tokens: int = 32


@dataclass
class RunConfig:
    run: RunMeta = field(default_factory=RunMeta)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    student: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    midtrain: MidtrainConfig = field(default_factory=MidtrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


STAGE_ORDER: tuple[str, ...] = ("pretrain", "midtrain", "sft", "dpo", "grpo", "distill")
SECTION_GROUPS: dict[str, tuple[str, str]] = {
    # group name -> (config sub-directory, target section)
    "hardware": ("hardware", "hardware"),
    "model": ("model", "model"),
    "student": ("model", "student"),
    "data": ("data", "data"),
}
GROUP_ORDER: tuple[str, ...] = ("hardware", "model", "student", "data", "stage")


# ------------------------------------------------------------------- dict <-> schema


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Validate ``value`` against annotation ``tp`` (recursively) and return it."""
    origin = get_origin(tp)
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if isinstance(value, tp):
            return value
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping for {tp.__name__}, got {value!r}")
        return _from_dict(tp, value, path)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: may not be null")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: {value!r} matches none of {args}: {errors}")
    if origin is Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{path}: {value!r} not one of {list(get_args(tp))}")
        return value
    if origin is list:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        (item_tp,) = get_args(tp)
        return [_coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        key_tp, val_tp = get_args(tp)
        return {
            _coerce(key_tp, k, f"{path}.<key>"): _coerce(val_tp, v, f"{path}.{k}")
            for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported annotation {_type_name(tp)}")


def _from_dict(cls: type[Any], data: Mapping[str, Any], path: str = "") -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{path}.{key}" if path else str(key)
        if key not in names:
            close = difflib.get_close_matches(str(key), names, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise ConfigError(f"unknown config key '{full}'{hint}")
        kwargs[key] = _coerce(hints[key], value, full)
    return cls(**kwargs)


def from_dict(data: Mapping[str, Any]) -> RunConfig:
    """Build and validate a :class:`RunConfig` from a (possibly partial) nested mapping."""
    cfg: RunConfig = _from_dict(RunConfig, data)
    validate(cfg)
    return cfg


def to_dict(cfg: Any) -> dict[str, Any]:
    """Plain nested dict (JSON/YAML-serialisable) of any config dataclass."""
    out: dict[str, Any] = dataclasses.asdict(cfg)
    return out


def section_from_dict(cls: type[Any], data: Mapping[str, Any]) -> Any:
    """Typed construction of a single section (e.g. a ModelConfig stored in a checkpoint)."""
    return _from_dict(cls, data)


# ------------------------------------------------------------------- merge/overrides


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; non-mapping values (incl. lists) in ``update`` replace."""
    out: dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(text: str) -> tuple[list[str], Any]:
    """``a.b.c=value`` -> (["a","b","c"], yaml-parsed value)."""
    if "=" not in text:
        raise ConfigError(f"override '{text}' must look like key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise ConfigError(f"override '{text}' has an empty key")
    value = yaml_load(raw) if raw.strip() else ""
    return key.split("."), value


def nest(keys: Sequence[str], value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        out = {key: out}
    return out


def split_overrides(args: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Separate ``group=name`` selections from ``dotted.key=value`` overrides."""
    groups: dict[str, str] = {}
    dotted: dict[str, Any] = {}
    for arg in args:
        keys, value = parse_override(arg)
        if len(keys) == 1 and keys[0] in GROUP_ORDER:
            groups[keys[0]] = str(value)
        else:
            dotted = deep_merge(dotted, nest(keys, value))
    return groups, dotted


# ------------------------------------------------------------------------ loading


def config_root() -> Path:
    """Directory holding the YAML groups: $TINYLAB_CONFIG_DIR, ./configs, or the repo's."""
    env = os.environ.get("TINYLAB_CONFIG_DIR")
    if env:
        return Path(env)
    local = Path.cwd() / "configs"
    if local.is_dir():
        return local
    return Path(__file__).resolve().parent.parent / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def group_dict(group: str, name: str, root: Path | None = None) -> dict[str, Any]:
    """Load one group file and return it as a root-level partial config."""
    root = root or config_root()
    if group == "stage":
        return load_yaml(root / "stage" / f"{name}.yaml")
    if group not in SECTION_GROUPS:
        raise ConfigError(f"unknown config group '{group}' (known: {list(GROUP_ORDER)})")
    subdir, section = SECTION_GROUPS[group]
    return {section: load_yaml(root / subdir / f"{name}.yaml")}


def compose(
    groups: Mapping[str, str],
    *partials: Mapping[str, Any],
    root: Path | None = None,
) -> RunConfig:
    """Compose a RunConfig from group selections followed by root-level partial dicts."""
    merged: dict[str, Any] = {}
    for group in GROUP_ORDER:
        if group in groups:
            merged = deep_merge(merged, group_dict(group, groups[group], root))
    for partial in partials:
        merged = deep_merge(merged, partial)
    return from_dict(merged)


def load_stage_config(args: Sequence[str], root: Path | None = None) -> RunConfig:
    """CLI entry: ``stage=pretrain model=smoke train.lr=1e-3`` -> RunConfig."""
    groups, dotted = split_overrides(args)
    return compose(groups, dotted, root=root)


# --------------------------------------------------------------------- validation


def validate(cfg: RunConfig) -> None:
    """Cross-field checks. Collects every problem before raising."""
    errors: list[str] = []
    for label, model in (("model", cfg.model), ("student", cfg.student)):
        errors += [f"{label}: {e}" for e in model_errors(model)]
    t = cfg.train
    if t.lr <= 0:
        errors.append("train.lr must be > 0")
    if not 0 <= t.min_lr_ratio <= 1:
        errors.append("train.min_lr_ratio must be in [0, 1]")
    if len(t.betas) != 2 or not all(0 <= b < 1 for b in t.betas):
        errors.append("train.betas must be two values in [0, 1)")
    if not 0 <= t.decay_frac <= 1:
        errors.append("train.decay_frac must be in [0, 1]")
    if t.max_steps < 0 or t.warmup_steps < 0:
        errors.append("train.max_steps and train.warmup_steps must be >= 0")
    if isinstance(t.micro_batch_size, int) and t.micro_batch_size < 1:
        errors.append("train.micro_batch_size must be >= 1 or 'auto'")
    for name in ("log_interval", "eval_interval", "ckpt_interval", "seq_len"):
        if getattr(t, name) < 1:
            errors.append(f"train.{name} must be >= 1")
    if t.seq_len > cfg.model.max_seq_len:
        errors.append(
            f"train.seq_len ({t.seq_len}) exceeds model.max_seq_len ({cfg.model.max_seq_len})"
        )
    for mix_name in ("pretrain_mix", "midtrain_mix"):
        mix: dict[str, float] = getattr(cfg.data, mix_name)
        unknown = set(mix) - {"stories", "math", "math_hq"}
        if unknown:
            errors.append(f"data.{mix_name}: unknown sources {sorted(unknown)}")
        if any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
            errors.append(f"data.{mix_name}: weights must be >= 0 and sum > 0")
    mw = cfg.data.mathworld
    if not mw.digits or not mw.steps or min(mw.digits) < 1 or min(mw.steps) < 1:
        errors.append("data.mathworld.digits/steps must be non-empty lists of ints >= 1")
    if set(mw.ops) - {"+", "-", "*"}:
        errors.append("data.mathworld.ops must be a subset of ['+', '-', '*']")
    g = cfg.grpo
    if g.group_size < 2:
        errors.append("grpo.group_size must be >= 2 (advantages are group-relative)")
    if g.clip_low < 0 or g.clip_high < 0:
        errors.append("grpo.clip_low/high must be >= 0")
    known_modes = {"baseline", "offpolicy", "logit", "onpolicy"}
    if set(cfg.distill.modes) - known_modes:
        errors.append(f"distill.modes must be a subset of {sorted(known_modes)}")
    ce = cfg.midtrain.context_extension
    if ce.enabled and ce.seq_len < t.seq_len:
        errors.append("midtrain.context_extension.seq_len must be >= train.seq_len")
    if cfg.stage.init_from not in ("auto", "scratch", *STAGE_ORDER):
        errors.append(f"stage.init_from must be auto|scratch|{'|'.join(STAGE_ORDER)}")
    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))


def model_errors(m: ModelConfig) -> list[str]:
    errors: list[str] = []
    if m.d_model < 1 or m.n_layers < 1 or m.n_heads < 1 or m.n_kv_heads < 1:
        errors.append("d_model, n_layers, n_heads, n_kv_heads must be >= 1")
        return errors
    if m.head_dim is None and m.d_model % m.n_heads:
        errors.append(f"d_model ({m.d_model}) must be divisible by n_heads ({m.n_heads})")
    if m.n_heads % m.n_kv_heads:
        errors.append(f"n_heads ({m.n_heads}) must be divisible by n_kv_heads ({m.n_kv_heads})")
    hd = m.head_dim or (m.d_model // m.n_heads)
    if hd % 2:
        errors.append(f"head_dim ({hd}) must be even for RoPE")
    if m.global_every < 1:
        errors.append("global_every must be >= 1")
    if m.moe.enabled:
        if not 1 <= m.moe.top_k <= m.moe.n_experts:
            errors.append(f"moe.top_k ({m.moe.top_k}) must be in [1, n_experts]")
        if not 0 <= m.moe.n_dense_layers <= m.n_layers:
            errors.append("moe.n_dense_layers must be in [0, n_layers]")
        if m.moe.capacity_factor is not None and m.moe.capacity_factor <= 0:
            errors.append("moe.capacity_factor must be > 0 or null")
        if m.moe.n_shared_experts < 0:
            errors.append("moe.n_shared_experts must be >= 0")
    if m.mtp.depth < 0:
        errors.append("mtp.depth must be >= 0")
    return errors


# ---------------------------------------------------------------------- batch math


@dataclass(frozen=True)
class BatchPlan:
    """How a global batch is split into micro-batches across ranks and accumulation steps.

    ``unit`` is the size of one row in the micro batch (seq_len for token-based stages, 1
    for example-based ones). The global batch is kept *exact*: the micro batch is lowered
    to the largest divisor of the per-rank row count when necessary.
    """

    micro_batch_size: int
    grad_accum: int
    world_size: int
    unit: int

    @property
    def rows_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.world_size

    @property
    def units_per_step(self) -> int:
        return self.rows_per_step * self.unit


def derive_batch(global_units: int, unit: int, micro_batch_size: int, world_size: int) -> BatchPlan:
    """Derive grad-accumulation steps from a global batch size.

    >>> derive_batch(global_units=4096, unit=128, micro_batch_size=8, world_size=1)
    BatchPlan(micro_batch_size=8, grad_accum=4, world_size=1, unit=128)
    """
    if min(global_units, unit, micro_batch_size, world_size) < 1:
        raise ConfigError("batch sizes must be positive")
    if global_units % unit:
        raise ConfigError(f"global batch {global_units} is not a multiple of row size {unit}")
    rows = global_units // unit
    if rows % world_size:
        raise ConfigError(f"global rows {rows} not divisible by world size {world_size}")
    per_rank = rows // world_size
    mbs = min(micro_batch_size, per_rank)
    while per_rank % mbs:
        mbs -= 1
    return BatchPlan(
        micro_batch_size=mbs, grad_accum=per_rank // mbs, world_size=world_size, unit=unit
    )


def head_dim(m: ModelConfig) -> int:
    return m.head_dim or m.d_model // m.n_heads


def ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)
