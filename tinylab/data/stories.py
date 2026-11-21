"""TinyStories (Eldan & Li, 2023) download/cache/clean, plus an offline synthetic stand-in.

TinyStories is the standard "small models can be fluent" corpus: simple vocabulary, short
narratives. We stream only the number of documents the config asks for and cache them as
JSONL, so a run never downloads the whole dataset. The synthetic generator keeps CI and
smoke runs offline and deterministic.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import unicodedata
from pathlib import Path

from tinylab.config import StoriesConfig
from tinylab.data.mathworld import rng_for

log = logging.getLogger(__name__)

# Subjects used for "Write a short story about a <subject>." instructions and checks.
SUBJECTS = (
    "cat", "dog", "bird", "fish", "girl", "boy", "bear", "rabbit", "frog", "duck",
    "tree", "ball", "car", "boat", "cake", "flower", "dragon", "princess", "king", "mouse",
)  # fmt: skip

_NAMES = ("Tom", "Lily", "Sam", "Mia", "Ben", "Anna", "Max", "Sue", "Tim", "Lucy")
_FEELINGS = ("happy", "sad", "scared", "excited", "curious", "tired", "proud")
_PLACES = ("park", "garden", "forest", "beach", "house", "school", "river")
_THINGS = ("a red ball", "a big box", "a shiny stone", "a little kite", "a blue hat", "a toy car")


def clean(text: str, max_chars: int) -> str | None:
    """Normalise unicode/whitespace; drop documents with non-ASCII leftovers or no content."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text or not text.isascii():
        return None
    if len(text) > max_chars:
        cut = text.rfind(".", 0, max_chars)
        text = text[: cut + 1] if cut > max_chars // 2 else text[:max_chars]
    return text


def synthetic_story(seed: int, index: int, split: str) -> str:
    """A short template story with a subject, a small problem and a resolution."""
    rng = rng_for(seed, f"story:{split}", index)
    name = rng.choice(_NAMES)
    subject = rng.choice(SUBJECTS)
    feeling, feeling2 = rng.sample(_FEELINGS, 2)
    place = rng.choice(_PLACES)
    thing = rng.choice(_THINGS)
    sentences = [
        f"Once upon a time, there was a little {subject} who lived near the {place}.",
        f"One day, {name} came to play with the {subject}.",
        f"They found {thing} and felt very {feeling}.",
        rng.choice(
            [
                f"But then the {subject} lost {thing}, and {name} was {feeling2}.",
                f"Then it started to rain, and the {subject} was {feeling2}.",
                f"Suddenly, a big wind came and the {subject} felt {feeling2}.",
            ]
        ),
        rng.choice(
            [
                f"{name} helped the {subject}, and they looked everywhere together.",
                f"The {subject} was brave and tried again.",
                f'{name} hugged the {subject} and said, "Do not worry."',
            ]
        ),
        rng.choice(
            [
                "In the end, they were friends forever.",
                f"From that day on, the {subject} was always {feeling}.",
                "They learned that sharing makes everyone happy.",
            ]
        ),
    ]
    return " ".join(sentences)


def _cache_path(cfg: StoriesConfig, split: str, n: int, cache_dir: Path) -> Path:
    key = hashlib.sha1(f"{cfg.hf_path}|{split}|{n}|{cfg.max_chars}".encode()).hexdigest()[:12]
    return cache_dir / "stories" / f"{key}.jsonl"


def _download(cfg: StoriesConfig, split: str, n: int) -> list[str]:
    import datasets  # heavy import, only when actually downloading

    hf_split = cfg.hf_train_split if split == "train" else cfg.hf_val_split
    stream = datasets.load_dataset(cfg.hf_path, split=hf_split, streaming=True)
    out: list[str] = []
    for row in itertools.islice(stream, n * 2):  # some rows are dropped by clean()
        text = clean(str(row["text"]), cfg.max_chars)
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_stories(cfg: StoriesConfig, split: str, n: int, cache_dir: Path, seed: int) -> list[str]:
    """``n`` cleaned stories for ``split`` ("train" | "val"), cached on disk."""
    if cfg.source == "synthetic":
        return [synthetic_story(seed, i, split) for i in range(n)]
    path = _cache_path(cfg, split, n, cache_dir)
    if path.is_file():
        return [json.loads(line)["text"] for line in path.read_text().splitlines()]
    try:
        stories = _download(cfg, split, n)
    except Exception as exc:  # network/HF errors
        if not cfg.fallback_to_synthetic:
            raise
        log.warning("TinyStories download failed (%s); falling back to synthetic stories", exc)
        return [synthetic_story(seed, i, split) for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps({"text": s}) + "\n" for s in stories))
    tmp.replace(path)
    log.info("cached %d %s stories at %s", len(stories), split, path)
    return stories


def find_subject(story: str) -> str | None:
    """First known subject noun mentioned in the story (for instruction data)."""
    lowered = story.lower()
    hits = [(m.start(), s) for s in SUBJECTS if (m := re.search(rf"\b{s}s?\b", lowered))]
    return min(hits)[1] if hits else None


def mentions(text: str, subject: str) -> bool:
    return re.search(rf"\b{re.escape(subject)}s?\b", text.lower()) is not None
