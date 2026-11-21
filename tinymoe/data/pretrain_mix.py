"""Tokenised memmap shards and a position-addressable mixture sampler.

* Each (source, split) is one flat ``uint16``/``uint32`` file of ``<|bos|> doc <|eos|>``
  documents (nanoGPT/Megatron style). Reading is a zero-copy ``np.memmap``.
* :class:`MixtureSampler` maps a *global row index* to (source, window) with an RNG seeded
  by ``(seed, index)``. Resuming therefore needs only the step counter, and DDP ranks read
  disjoint rows of the same global batch without coordination.
* Optional document masking returns per-token doc ids and per-doc positions so attention
  never crosses a document boundary (Llama-3 style intra-document masking).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tinymoe.config import DataConfig
from tinymoe.data import mathworld, stories
from tinymoe.data.batch import Batch, doc_layout, stack_rows
from tinymoe.data.tokenizer import Tokenizer

log = logging.getLogger(__name__)
SOURCES = ("stories", "math", "math_hq")
SPLITS = ("train", "val")


def _key(*parts: object) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def doc_count(cfg: DataConfig, source: str, split: str) -> int:
    if source == "stories":
        return cfg.stories.num_train_docs if split == "train" else cfg.stories.num_val_docs
    return cfg.mathworld.num_train_docs if split == "train" else cfg.mathworld.num_val_docs


def source_texts(cfg: DataConfig, source: str, split: str, n: int | None = None) -> list[str]:
    """Raw documents of one source/split (deterministic for a given config)."""
    n = doc_count(cfg, source, split) if n is None else n
    if source == "stories":
        return stories.load_stories(cfg.stories, split, n, Path(cfg.cache_dir), cfg.seed)
    if source in ("math", "math_hq"):
        hq = source == "math_hq"
        return [
            mathworld.pretrain_document(cfg.mathworld, cfg.seed, i, split, hq=hq) for i in range(n)
        ]
    raise ValueError(f"unknown source '{source}'")


def _tokenizer_corpus(cfg: DataConfig) -> Iterator[str]:
    n = cfg.tokenizer.train_docs_per_source
    for source in ("stories", "math", "math_hq"):
        yield from source_texts(cfg, source, "train", min(n, doc_count(cfg, source, "train")))


def build_tokenizer(cfg: DataConfig) -> tuple[Tokenizer, Path]:
    """Train (or load the cached) tokenizer for this data config."""
    key = _key("tok", asdict(cfg.tokenizer), asdict(cfg.stories), asdict(cfg.mathworld), cfg.seed)
    path = Path(cfg.cache_dir) / "tokenizers" / key / "tokenizer.json"
    if path.is_file():
        return Tokenizer.load(path), path
    log.info("training BPE tokenizer (vocab=%d)", cfg.tokenizer.vocab_size)
    tok = Tokenizer.train(
        _tokenizer_corpus(cfg), cfg.tokenizer.vocab_size, cfg.tokenizer.min_frequency
    )
    tok.save(path)
    return tok, path


def shard_dtype(vocab_size: int) -> np.dtype[np.unsignedinteger]:
    """uint16 halves disk/memory for vocabularies up to 65536 tokens."""
    return np.dtype(np.uint16) if vocab_size <= 2**16 else np.dtype(np.uint32)


def write_shard(path: Path, docs: Sequence[str], tok: Tokenizer) -> dict[str, object]:
    """Tokenise ``docs`` as ``<|bos|> doc <|eos|>`` and write a flat binary shard + meta."""
    dtype = shard_dtype(tok.vocab_size)
    arrays = [
        np.asarray([tok.bos_id, *ids, tok.eos_id], dtype=dtype) for ids in tok.encode_batch(docs)
    ]
    flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    flat.tofile(tmp)
    tmp.replace(path)
    meta: dict[str, object] = {
        "n_tokens": int(flat.size),
        "n_docs": len(docs),
        "dtype": np.dtype(dtype).name,
        "tokenizer": tok.fingerprint(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    return meta


def open_shard(path: Path) -> np.ndarray:
    meta = json.loads(path.with_suffix(".json").read_text())
    return np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r", shape=(meta["n_tokens"],))


def build_shards(
    cfg: DataConfig, tok: Tokenizer, sources: Sequence[str] = SOURCES
) -> dict[str, dict[str, Path]]:
    """Ensure shards exist for every (source, split); returns their paths (cached)."""
    out: dict[str, dict[str, Path]] = {}
    for source in sources:
        out[source] = {}
        sub = asdict(cfg.stories) if source == "stories" else asdict(cfg.mathworld)
        for split in SPLITS:
            key = _key("shard", source, split, sub, cfg.seed, tok.fingerprint())
            path = Path(cfg.cache_dir) / "shards" / key / f"{source}.{split}.bin"
            if not (path.is_file() and path.with_suffix(".json").is_file()):
                docs = source_texts(cfg, source, split)
                meta = write_shard(path, docs, tok)
                log.info("shard %s/%s: %s tokens", source, split, f"{meta['n_tokens']:,}")
            out[source][split] = path
    return out


class MixtureSampler:
    """Seeded, stateless mapping ``row index -> (source, seq_len+1 tokens)``."""

    def __init__(
        self,
        shards: Mapping[str, np.ndarray],
        weights: Mapping[str, float],
        seq_len: int,
        seed: int,
        bos_id: int,
        doc_mask: bool = False,
    ) -> None:
        self.names = [name for name in sorted(weights) if weights[name] > 0]
        missing = [n for n in self.names if n not in shards]
        if missing:
            raise ValueError(f"no shards for sources {missing}")
        total = sum(weights[n] for n in self.names)
        self.probs = np.array([weights[n] / total for n in self.names])
        self.shards = [shards[n] for n in self.names]
        for name, shard in zip(self.names, self.shards, strict=True):
            if len(shard) <= seq_len + 1:
                raise ValueError(f"source '{name}' has {len(shard)} tokens < seq_len+1")
        self.seq_len = seq_len
        self.seed = seed
        self.bos_id = bos_id
        self.doc_mask = doc_mask

    def row(self, index: int) -> tuple[int, np.ndarray]:
        rng = np.random.default_rng([self.seed, index])
        src = int(rng.choice(len(self.names), p=self.probs))
        shard = self.shards[src]
        offset = int(rng.integers(0, len(shard) - self.seq_len - 1))
        return src, np.asarray(shard[offset : offset + self.seq_len + 1], dtype=np.int64)

    def batch(self, indices: Sequence[int]) -> Batch:
        srcs, rows = zip(*(self.row(i) for i in indices), strict=True)
        inputs = [r[:-1] for r in rows]
        labels = [r[1:] for r in rows]
        if not self.doc_mask:
            return stack_rows(inputs, labels, sources=srcs)
        layouts = [doc_layout(x, self.bos_id) for x in inputs]
        return stack_rows(
            inputs,
            labels,
            doc_ids=[d for d, _ in layouts],
            positions=[p for _, p in layouts],
            sources=srcs,
        )


def open_sources(paths: Mapping[str, Mapping[str, Path]], split: str) -> dict[str, np.ndarray]:
    return {name: open_shard(splits[split]) for name, splits in paths.items()}
