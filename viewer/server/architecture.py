"""Model architecture of a run's stage, read off the real module tree.

The model is built from the stage's resolved config on the shape-only ("meta") device:
every parameter has its true name, shape and count, but nothing is allocated, so this
is cheap and never touches a checkpoint.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from viewer.server.runs import NotFoundError


def _stage_entry(manifest: dict[str, Any], stage: str | None) -> dict[str, Any]:
    stages = [s for s in manifest.get("stages", []) if isinstance(s.get("config"), dict)]
    if not stages:
        raise NotFoundError("no stage with a recorded config yet")
    if stage is None:
        last: dict[str, Any] = stages[-1]
        return last
    for s in stages:
        if s["name"] == stage:
            found: dict[str, Any] = s
            return found
    raise NotFoundError(f"stage {stage!r} has no recorded config")


@lru_cache(maxsize=32)
def _describe(run_dir: str, stage: str | None, which: str, mtime: float) -> dict[str, Any]:
    del mtime  # cache key only: a rewritten manifest describes afresh
    from tinymoe import hardware
    from tinymoe.config import from_dict
    from tinymoe.data.tokenizer import Tokenizer
    from tinymoe.model.config import param_count
    from tinymoe.model.transformer import Transformer

    root = Path(run_dir)
    entry = _stage_entry(json.loads((root / "manifest.json").read_text()), stage)
    rc = from_dict(entry["config"])
    if which not in ("model", "student"):
        raise NotFoundError(f"unknown model {which!r}")
    mc = getattr(rc, which)
    if mc.vocab_size < 1:
        tok = root / "tokenizer.json"
        if not tok.exists():
            raise NotFoundError("the run has no tokenizer yet")
        mc.vocab_size = Tokenizer.load(tok).vocab_size
    with hardware.shape_only_device():
        model = Transformer(mc)
    params = [
        {"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in model.named_parameters()
    ]
    buffers = [
        {"name": n, "shape": list(b.shape), "numel": b.numel()}
        for n, b in model.named_buffers()
        if not n.endswith((".cos", ".sin", "load_counts"))
    ]
    train = entry["config"].get("train", {})
    return {
        "stage": entry["name"],
        "which": which,
        "config": asdict(mc),
        "counts": param_count(mc).as_dict(),
        "params": params,
        "buffers": buffers,
        "seq_len": train.get("seq_len"),
        "tokens_per_step": train.get("tokens_per_step"),
    }


def describe(run_dir: Path, stage: str | None = None, which: str = "model") -> dict[str, Any]:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        raise NotFoundError(f"no manifest in {run_dir.name}")
    return _describe(str(run_dir), stage, which, manifest.stat().st_mtime)


def prewarm() -> None:
    """Import torch and the model code in the background (several seconds cold), so the
    first architecture request does not wait for it."""

    def load() -> None:
        import tinymoe.model.transformer  # noqa: F401

    threading.Thread(target=load, name="architecture-prewarm", daemon=True).start()
