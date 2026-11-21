"""Device, dtype and capability detection with explicit, logged fallbacks.

This is the *only* module that knows backend names. Everything else consults a
:class:`Capabilities` (what the machine can do) and a :class:`Runtime` (what this run will
do, after applying the config policy). Every fallback is recorded in ``Runtime.decisions``
and logged at startup so a run's numerics are never a surprise.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import random
import resource
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinymoe.config import HardwareConfig

log = logging.getLogger(__name__)

_CUDA = "cuda"
_MPS = "mps"
_CPU = "cpu"
_SDPA_BACKENDS: dict[str, SDPBackend] = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}
_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class Capabilities:
    """What the current machine/backend supports (detected, never assumed)."""

    device_type: str
    device_index: int | None
    device_name: str
    device_count: int
    total_memory_bytes: int | None
    autocast_dtypes: tuple[str, ...]
    native_bf16: bool  # hardware bf16 (fast); emulated bf16 may still be in autocast_dtypes
    fused_adamw: bool
    sdpa_backends: tuple[str, ...]
    compile_available: bool
    compile_recommended: bool
    torch_version: str
    python_version: str
    platform: str
    cpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Runtime:
    """Resolved execution policy for one process."""

    device: torch.device
    caps: Capabilities
    precision: str
    use_loss_scaling: bool
    compile: bool
    compile_backend: str
    fused_optimizer: bool
    sdpa_backend: str
    measured_tflops: float | None
    peak_tflops: float | None
    decisions: list[str] = field(default_factory=list)

    @property
    def amp_dtype(self) -> torch.dtype | None:
        return None if self.precision == "fp32" else _DTYPES[self.precision]

    @property
    def device_type(self) -> str:
        return self.device.type

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Mixed-precision context; master weights stay fp32 (autocast only casts ops)."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def sdpa_context(self) -> contextlib.AbstractContextManager[Any]:
        if self.sdpa_backend == "auto":
            return contextlib.nullcontext()
        return sdpa_kernel([_SDPA_BACKENDS[self.sdpa_backend]])

    def summary(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "precision": self.precision,
            "loss_scaling": self.use_loss_scaling,
            "compile": self.compile,
            "fused_optimizer": self.fused_optimizer,
            "sdpa_backend": self.sdpa_backend,
            "measured_tflops": self.measured_tflops,
            "peak_tflops": self.peak_tflops,
            "decisions": list(self.decisions),
        }


# ------------------------------------------------------------------ device choice


def accelerator_types() -> list[str]:
    """Available accelerator backends in preference order (cuda > mps)."""
    found = []
    if torch.cuda.is_available():
        found.append(_CUDA)
    if torch.backends.mps.is_available():
        found.append(_MPS)
    return found


def resolve_device(requested: str, local_rank: int = 0) -> torch.device:
    """``auto`` -> best accelerator, else CPU. Explicit requests are validated."""
    if requested == "auto":
        accels = accelerator_types()
        kind = accels[0] if accels else _CPU
    else:
        kind = requested.split(":")[0]
        if kind != _CPU and kind not in accelerator_types():
            raise RuntimeError(f"requested device '{requested}' is not available on this machine")
    if kind == _CUDA:
        index = int(requested.split(":")[1]) if ":" in requested else local_rank
        return torch.device(kind, index)
    return torch.device(kind)


def _device_memory(device: torch.device) -> int | None:
    if device.type == _CUDA:
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == _MPS:
        return int(torch.mps.recommended_max_memory())
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _device_name(device: torch.device) -> str:
    if device.type == _CUDA:
        return str(torch.cuda.get_device_name(device))
    return f"{device.type} ({platform.processor() or platform.machine()})"


# ---------------------------------------------------------------------- probes


def _probe(fn: Callable[[], object]) -> bool:
    try:
        fn()
        return True
    except Exception as exc:  # probes must never crash startup
        log.debug("probe failed: %s", exc)
        return False


def _probe_autocast(device: torch.device, dtype: torch.dtype) -> bool:
    def run() -> None:
        lin = torch.nn.Linear(16, 16).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = lin(torch.randn(4, 16, device=device))
        if out.dtype != dtype:
            raise RuntimeError("autocast did not change dtype")
        torch.autograd.backward(out.float().sum())

    return _probe(run)


def _probe_fused_adamw(device: torch.device) -> bool:
    def run() -> None:
        p = torch.nn.Parameter(torch.randn(8, device=device))
        opt = torch.optim.AdamW([p], lr=1e-3, fused=True)
        torch.autograd.backward(p.sum())
        opt.step()

    return _probe(run)


def _probe_sdpa(device: torch.device, backend: str) -> bool:
    dtype = torch.float32 if device.type == _CPU else torch.float16

    def run() -> None:
        q = torch.randn(1, 2, 8, 16, device=device, dtype=dtype)
        with sdpa_kernel([_SDPA_BACKENDS[backend]]):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)

    return _probe(run)


def probe_compile(device: torch.device, backend: str) -> bool:
    """Try to compile and run a tiny function; any failure means 'fall back to eager'."""

    def run() -> None:
        fn = torch.compile(lambda x: torch.nn.functional.silu(x) * 2.0, backend=backend)
        fn(torch.randn(8, device=device))

    return _probe(run)


def detect(cfg: HardwareConfig, local_rank: int = 0) -> Capabilities:
    """Detect capabilities of the device selected by ``cfg.device``."""
    device = resolve_device(cfg.device, local_rank)
    if device.type == _CUDA:
        torch.cuda.set_device(device)
    autocast = tuple(name for name in ("bf16", "fp16") if _probe_autocast(device, _DTYPES[name]))
    native_bf16 = device.type == _CUDA and bool(torch.cuda.is_bf16_supported())
    if device.type == _CUDA:
        count = torch.cuda.device_count()
    elif device.type == _MPS:
        count = 1
    else:
        count = 1
    return Capabilities(
        device_type=device.type,
        device_index=device.index,
        device_name=_device_name(device),
        device_count=count,
        total_memory_bytes=_device_memory(device),
        autocast_dtypes=autocast,
        native_bf16=native_bf16,
        fused_adamw=_probe_fused_adamw(device),
        sdpa_backends=tuple(b for b in _SDPA_BACKENDS if _probe_sdpa(device, b)),
        compile_available=hasattr(torch, "compile"),
        compile_recommended=device.type == _CUDA,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        cpu_count=os.cpu_count() or 1,
    )


def matmul_tflops(
    device: torch.device, dtype: torch.dtype, n: int = 1024, iters: int = 10
) -> float:
    """Measured dense-matmul throughput (TFLOP/s); used for auto precision and MFU."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(2):
        a @ b
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return 2 * n**3 * iters / elapsed / 1e12


def best_tflops(
    device: torch.device, dtypes: dict[str, torch.dtype], rounds: int = 3, n: int = 2048
) -> dict[str, float]:
    """Best-of-``rounds`` matmul throughput per dtype, measured in interleaved rounds.

    A single short timing is dominated by clock ramp-up and scheduling noise: on an M1 Pro
    one 1024-squared run reported fp16 at 1.26x fp32 and the next at 1.00x. Interleaving the
    dtypes exposes each to the same thermal state, and the best round is the least noisy
    estimate of what the hardware can do.
    """
    best = dict.fromkeys(dtypes, 0.0)
    for _ in range(rounds):
        for name, dtype in dtypes.items():
            best[name] = max(best[name], matmul_tflops(device, dtype, n=n))
    return best


# ------------------------------------------------------------------- resolution


def _resolve_precision(
    cfg: HardwareConfig, caps: Capabilities, device: torch.device
) -> tuple[str, str]:
    req = cfg.precision
    if req == "fp32":
        return "fp32", "precision=fp32 (requested)"
    if req in ("bf16", "fp16"):
        if req in caps.autocast_dtypes:
            return req, f"precision={req} (requested)"
        return "fp32", f"precision=fp32 (fallback: {req} autocast unsupported on {device.type})"
    # auto
    if caps.native_bf16 and "bf16" in caps.autocast_dtypes:
        return "bf16", "precision=bf16 (auto: native bf16)"
    if device.type == _CPU or not caps.autocast_dtypes:
        return "fp32", f"precision=fp32 (auto: no fast low-precision path on {device.type})"
    if not cfg.precision_bench:
        return "fp32", "precision=fp32 (auto, benchmarking disabled)"
    dtypes = {"fp32": torch.float32, **{n: _DTYPES[n] for n in caps.autocast_dtypes}}
    tflops = best_tflops(device, dtypes)
    speeds = {name: tflops[name] / tflops["fp32"] for name in caps.autocast_dtypes}
    ok = {k: v for k, v in speeds.items() if v >= cfg.low_precision_min_speedup}
    if not ok:
        return "fp32", f"precision=fp32 (auto: low precision not faster, {speeds})"
    best = max(ok, key=lambda k: ok[k])
    if best == "fp16" and ok.get("bf16", 0.0) >= 0.95 * ok["fp16"]:
        best = "bf16"  # within 5%: prefer bf16, which needs no loss scaling
    best_speed = ok[best]
    return best, f"precision={best} (auto: measured {best_speed:.2f}x vs fp32)"


def resolve_runtime(cfg: HardwareConfig, local_rank: int = 0) -> Runtime:
    """Detect capabilities, apply the config policy, and log every decision."""
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    caps = detect(cfg, local_rank)
    device = resolve_device(cfg.device, local_rank)
    decisions: list[str] = [f"device={device} ({caps.device_name})"]

    precision, note = _resolve_precision(cfg, caps, device)
    decisions.append(note)
    use_scaling = precision == "fp16"
    if use_scaling:
        decisions.append("fp16 -> dynamic loss scaling enabled")

    compile_on = False
    if cfg.compile == "on" or (cfg.compile == "auto" and caps.compile_recommended):
        if caps.compile_available and probe_compile(device, cfg.compile_backend):
            compile_on = True
            decisions.append(f"compile=on (backend={cfg.compile_backend})")
        else:
            decisions.append("compile=off (fallback: torch.compile probe failed)")
    else:
        decisions.append(f"compile=off ({cfg.compile})")

    fused = cfg.fused_optimizer != "off" and caps.fused_adamw
    if cfg.fused_optimizer == "on" and not caps.fused_adamw:
        decisions.append("fused AdamW unavailable -> foreach fallback")
    else:
        decisions.append(f"adamw={'fused' if fused else 'foreach'}")

    sdpa = cfg.sdpa_backend
    if sdpa != "auto" and sdpa not in caps.sdpa_backends:
        decisions.append(f"sdpa={sdpa} unavailable -> math fallback")
        sdpa = "math"
    else:
        decisions.append(f"sdpa={sdpa} (available: {','.join(caps.sdpa_backends)})")

    if cfg.memory_budget_gb and cfg.enforce_memory_budget:
        decisions.append(cap_memory(device, cfg.memory_budget_gb))

    measured = None
    if cfg.peak_tflops is None:
        dtype = _DTYPES[precision]
        measured = matmul_tflops(device, dtype)
    runtime = Runtime(
        device=device,
        caps=caps,
        precision=precision,
        use_loss_scaling=use_scaling,
        compile=compile_on,
        compile_backend=cfg.compile_backend,
        fused_optimizer=fused,
        sdpa_backend=sdpa,
        measured_tflops=measured,
        peak_tflops=cfg.peak_tflops,
        decisions=decisions,
    )
    for line in decisions:
        log.info("[hardware] %s", line)
    return runtime


# --------------------------------------------------------------- device helpers


def synchronize(device: torch.device) -> None:
    if device.type == _CUDA:
        torch.cuda.synchronize(device)
    elif device.type == _MPS:
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == _CUDA:
        torch.cuda.empty_cache()
    elif device.type == _MPS:
        torch.mps.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == _CUDA:
        torch.cuda.reset_peak_memory_stats(device)


def cap_memory(device: torch.device, budget_gb: float) -> str:
    """Make allocations beyond the budget raise OOM instead of swapping.

    Unified-memory Macs happily page GPU buffers to disk, which turns an oversized batch
    into a frozen machine rather than an error. Capping the process (MPS: fraction of the
    recommended working set; CUDA: fraction of device memory) restores the OOM signal
    that ``micro_batch_size: auto`` and the trainer rely on.
    """
    if device.type == _MPS:
        total = float(torch.mps.recommended_max_memory())
        frac = min(max(budget_gb * 1e9 / total, 0.05), 2.0)
        torch.mps.set_per_process_memory_fraction(frac)
        return f"memory cap {budget_gb:.1f} GB (mps fraction {frac:.2f} of {total / 1e9:.1f} GB)"
    if device.type == _CUDA:
        total = float(torch.cuda.get_device_properties(device).total_memory)
        frac = min(budget_gb * 1e9 / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(frac, device)
        return f"memory cap {budget_gb:.1f} GB (cuda fraction {frac:.2f})"
    return f"memory budget {budget_gb:.1f} GB (advisory on {device.type}: checked via RSS)"


def process_rss_bytes() -> int:
    """Peak resident set size of this process (bytes; macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def memory_bytes(device: torch.device) -> dict[str, int]:
    """Current/peak memory for the device (unified-memory devices also report driver use)."""
    if device.type == _CUDA:
        return {
            "allocated": int(torch.cuda.memory_allocated(device)),
            "peak": int(torch.cuda.max_memory_allocated(device)),
        }
    if device.type == _MPS:
        driver = int(torch.mps.driver_allocated_memory())
        return {"allocated": int(torch.mps.current_allocated_memory()), "peak": driver}
    rss = process_rss_bytes()
    return {"allocated": rss, "peak": rss}


def host_device() -> torch.device:
    """Host memory: where checkpoints are staged before moving to the run device."""
    return torch.device(_CPU)


def has_memory_stats(device: torch.device) -> bool:
    """Whether :func:`memory_bytes` reflects this device (CPU only reports peak RSS)."""
    return device.type in (_CUDA, _MPS)


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    """Every RNG that can affect a run (python, numpy, torch CPU + accelerator)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state[_CUDA] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state[_MPS] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if _CUDA in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[_CUDA])
    if _MPS in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state[_MPS])


@contextlib.contextmanager
def preserved_rng() -> Iterator[None]:
    """Run a block without perturbing any RNG stream (used by snapshots/eval)."""
    state = rng_state()
    try:
        yield
    finally:
        set_rng_state(state)


def dist_backend(requested: str, device: torch.device) -> str:
    """NCCL for CUDA devices, gloo otherwise (CPU, MPS)."""
    if requested != "auto":
        return requested
    return "nccl" if device.type == _CUDA else "gloo"


def describe() -> dict[str, Any]:
    """Capabilities of every detectable backend, for ``tinymoe info``."""
    out: dict[str, Any] = {}
    for kind in [_CPU, *accelerator_types()]:
        caps = detect(HardwareConfig(device=kind))
        out[kind] = caps.to_dict()
    return out
