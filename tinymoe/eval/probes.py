"""Fixed probe prompts, identical at every eval of every stage.

They let the viewer show "the same question answered over time": a base model rambles,
mid-training adds arithmetic, SFT adds the ``<think>`` format, RL sharpens correctness.
Base stages (pretrain/midtrain) are probed in the raw pre-training format; chat stages
through the single chat template.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinymoe.config import RunConfig
from tinymoe.data import mathworld
from tinymoe.data.mathworld import Bucket
from tinymoe.data.tokenizer import Message, Tokenizer, render_chat

BASE_STAGES = ("pretrain", "midtrain")


def eval_kind(stage: str) -> str:
    """'base' for raw-text stages, 'chat' once the chat template has been learned."""
    return "base" if stage in BASE_STAGES else "chat"


@dataclass(frozen=True)
class Probe:
    id: str
    kind: str  # story | instruction | math
    prompt: str
    answer: int | None = None


def math_prompt_ids(tok: Tokenizer, question: str, kind: str) -> list[int]:
    if kind == "base":
        return [tok.bos_id, *tok.encode(f"Question: {question}\n")]
    return chat_prompt_ids(tok, question)


def chat_prompt_ids(tok: Tokenizer, text: str) -> list[int]:
    return render_chat(tok, [Message("user", text)], add_generation_prompt=True)[0]


def probes(cfg: RunConfig) -> list[Probe]:
    mw = cfg.data.mathworld
    bucket = Bucket(mw.digits[len(mw.digits) // 2], mw.steps[len(mw.steps) // 2])
    out = [Probe(f"story{i}", "story", text) for i, text in enumerate(cfg.eval.probe_stories)]
    out += [Probe(f"instr{i}", "instruction", t) for i, t in enumerate(cfg.eval.probe_instructions)]
    for i in range(cfg.eval.probe_math):
        p = mathworld.make_problem(mathworld.rng_for(cfg.eval.seed, "probe", i), bucket, mw)
        out.append(Probe(f"math{i}", "math", p.question, p.answer))
    return out


def probe_ids(tok: Tokenizer, probe: Probe, kind: str) -> list[int]:
    if probe.kind == "story":
        return [tok.bos_id, *tok.encode(probe.prompt)]
    if probe.kind == "math":
        return math_prompt_ids(tok, probe.prompt, kind)
    return chat_prompt_ids(tok, probe.prompt)
