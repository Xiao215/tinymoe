"""``tinylab chat <checkpoint>``: multi-turn chat REPL over the single chat template.

Shows the ``<think>`` block verbatim (so the effect of SFT/RL on reasoning is visible),
the verifier's parsed answer for math-looking replies, and tokens/s. ``/reset`` clears
the history, ``/quit`` exits.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import torch

from tinylab import hardware
from tinylab.config import load_stage_config
from tinylab.data import mathworld
from tinylab.data.tokenizer import Message, Tokenizer, render_chat, stop_ids
from tinylab.model.transformer import generate
from tinylab.train import checkpoint


def default_tokenizer(ckpt: Path) -> Path:
    """``runs/<id>/checkpoints/<stage>/final.pt`` -> ``runs/<id>/tokenizer.json``."""
    for parent in ckpt.resolve().parents:
        candidate = parent / "tokenizer.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no tokenizer.json above {ckpt}; pass --tokenizer")


def main(ckpt: Path, tokenizer: Path | None, overrides: Sequence[str]) -> int:
    cfg = load_stage_config(list(overrides))
    runtime = hardware.resolve_runtime(cfg.hardware)
    tok = Tokenizer.load(tokenizer or default_tokenizer(ckpt))
    model = checkpoint.load_model(ckpt, runtime.device).eval()
    gen = torch.Generator().manual_seed(cfg.run.seed)
    history: list[Message] = []
    print(f"loaded {ckpt} on {runtime.device}; /reset clears history, /quit exits")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            return 0
        if line == "/quit":
            return 0
        if line == "/reset":
            history.clear()
            continue
        if not line:
            continue
        history.append(Message("user", line))
        ids, _ = render_chat(tok, history, add_generation_prompt=True)
        t0 = time.perf_counter()
        with runtime.autocast():
            out = generate(
                model,
                [ids],
                cfg.eval.max_new_tokens,
                cfg.eval.temperature,
                1.0,
                stop_ids(tok),
                tok.pad_id,
                gen,
            )
        seq = out.sequences[0]
        dt = time.perf_counter() - t0
        body = seq[:-1] if seq and seq[-1] in stop_ids(tok) else seq
        reply = tok.decode(body)
        history.append(Message("assistant", reply))
        print(f"model> {reply}")
        answer = mathworld.extract_answer(reply)
        extra = f", parsed answer {answer}" if answer is not None else ""
        print(f"  [{len(seq)} tokens, {len(seq) / max(dt, 1e-9):.0f} tok/s{extra}]")
