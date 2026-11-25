"""Document generator: random sentences plus a unique marker sentence per (doc, version, nonce)."""
from __future__ import annotations

import random
import re

_SYL = ["ka", "lo", "mi", "ta", "ren", "vo", "su", "ni", "pa", "do", "el", "or", "bex", "qui", "zal", "tho"]
MARKER = re.compile(r"zq-([a-z0-9]+)-(\d+)-(\d+)")


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3)))


def marker(doc_id: str, version: int, nonce: int) -> str:
    return f"zq-{doc_id}-{version}-{nonce}"


def make_text(doc_id: str, version: int, nonce: int, n_sentences: int = 4) -> str:
    rng = random.Random(f"{doc_id}/{version}/{nonce}")
    sentences = [
        " ".join(_word(rng) for _ in range(rng.randint(6, 11))).capitalize() + "."
        for _ in range(n_sentences)
    ]
    sentences.insert(rng.randint(0, len(sentences)), f"Reference marker {marker(doc_id, version, nonce)}.")
    return " ".join(sentences)


def parse_marker(text: str) -> tuple[str, int] | None:
    m = MARKER.search(text)
    return (m.group(1), int(m.group(2))) if m else None
