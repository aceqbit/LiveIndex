"""Sentence-aware chunker: ~target tokens per chunk with fractional overlap, deterministic."""
from __future__ import annotations

import re
from collections.abc import Callable

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")

TokenCounter = Callable[[str], int]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def _split_long(sentence: str, count: TokenCounter, target: int) -> list[str]:
    """Break a sentence longer than target into word runs of at most target tokens."""
    out: list[str] = []
    cur: list[str] = []
    for w in sentence.split():
        cur.append(w)
        if count(" ".join(cur)) > target and len(cur) > 1:
            cur.pop()
            out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(
    text: str, count: TokenCounter, target: int = 300, overlap: float = 0.15
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if count(text) <= target:
        return [text]
    units: list[tuple[str, int]] = []
    for s in split_sentences(text):
        parts = [s] if count(s) <= target else _split_long(s, count, target)
        units.extend((p, count(p)) for p in parts)
    budget = int(target * overlap)
    chunks: list[str] = []
    start = 0
    n = len(units)
    while start < n:
        end, total = start, 0
        while end < n and (total + units[end][1] <= target or end == start):
            total += units[end][1]
            end += 1
        chunks.append(" ".join(u[0] for u in units[start:end]))
        if end >= n:
            break
        # step back to build overlap, but always advance at least one unit
        back, acc = end, 0
        while back > start + 1 and acc + units[back - 1][1] <= budget:
            back -= 1
            acc += units[back][1]
        start = max(back, start + 1)
    return chunks
