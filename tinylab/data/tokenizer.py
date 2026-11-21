"""Byte-level BPE tokenizer with individually split digits, and the chat template.

* Byte-level BPE (GPT-2 / Llama-3 style) never produces UNK, so any text round-trips.
* Digits are pre-split into single characters (Llama-3, Qwen, DeepSeek do the same): the
  model then sees a consistent place-value representation, which is what makes
  multi-digit arithmetic learnable for small models.
* :func:`render_chat` is the *single* chat-template implementation used by SFT, DPO, RL,
  evaluation and the chat REPL, so the train and inference formats can never drift.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS = "<|pad|>", "<|bos|>", "<|eos|>"
USER, ASSISTANT, SYSTEM, END = "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"
THINK, THINK_END = "<think>", "</think>"
SPECIAL_TOKENS = (PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, END, THINK, THINK_END)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class Tokenizer:
    """Thin typed wrapper around a HF ``tokenizers.Tokenizer``."""

    def __init__(self, tk: HFTokenizer) -> None:
        self._tk = tk
        found = {tok: tk.token_to_id(tok) for tok in SPECIAL_TOKENS}
        missing = [t for t, i in found.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens {missing}")
        self.special_ids: dict[str, int] = {t: int(i) for t, i in found.items() if i is not None}
        self.pad_id = self.special_ids[PAD]
        self.bos_id = self.special_ids[BOS]
        self.eos_id = self.special_ids[EOS]
        self.end_id = self.special_ids[END]
        self.user_id = self.special_ids[USER]
        self.assistant_id = self.special_ids[ASSISTANT]
        self.system_id = self.special_ids[SYSTEM]
        self.think_id = self.special_ids[THINK]
        self.think_end_id = self.special_ids[THINK_END]
        self.role_ids: dict[str, int] = {
            "system": self.system_id,
            "user": self.user_id,
            "assistant": self.assistant_id,
        }

    # ----------------------------------------------------------- construction
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2) -> Tokenizer:
        tk = HFTokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return cls(tk)

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        return cls(HFTokenizer.from_file(str(path)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))

    # ------------------------------------------------------------------- api
    @property
    def vocab_size(self) -> int:
        return int(self._tk.get_vocab_size())

    def fingerprint(self) -> str:
        return hashlib.sha1(self._tk.to_str().encode()).hexdigest()[:12]

    def encode(self, text: str) -> list[int]:
        """Encode raw text; special-token strings inside it map to their ids."""
        return list(self._tk.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [list(e.ids) for e in self._tk.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return str(self._tk.decode(list(ids), skip_special_tokens=skip_special))

    def token_str(self, token_id: int) -> str:
        """Human-readable surface form of one token (for viewer labels)."""
        if token_id in self.special_ids.values():
            return str(self._tk.id_to_token(token_id))
        return self.decode([token_id])

    def token_id(self, text: str) -> int | None:
        """Id of ``text`` if it encodes to exactly one token."""
        ids = self.encode(text)
        return ids[0] if len(ids) == 1 else None


def render_chat(
    tok: Tokenizer, messages: Sequence[Message], add_generation_prompt: bool = False
) -> tuple[list[int], list[int]]:
    """Chat template -> (token ids, assistant mask).

    Layout: ``<|bos|>`` then per message ``<|role|> content <|end|>``. The mask is 1 on
    assistant content *and* its ``<|end|>`` (so the model learns to stop) and 0 on role
    tokens, system and user text - the standard assistant-only SFT loss.
    With ``add_generation_prompt`` a trailing ``<|assistant|>`` opens the reply.
    """
    ids = [tok.bos_id]
    mask = [0]
    for msg in messages:
        content = tok.encode(msg.content)
        learn = 1 if msg.role == "assistant" else 0
        ids += [tok.role_ids[msg.role], *content, tok.end_id]
        mask += [0] + [learn] * (len(content) + 1)
    if add_generation_prompt:
        ids.append(tok.assistant_id)
        mask.append(0)
    return ids, mask


def stop_ids(tok: Tokenizer) -> tuple[int, ...]:
    """Tokens that terminate an assistant turn during generation."""
    return (tok.end_id, tok.eos_id)
