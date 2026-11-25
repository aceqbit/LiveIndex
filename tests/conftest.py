import re

import numpy as np
import psycopg
import pytest

from liveindex.wal.log import WriteLog, database_url


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests: hashed words -> normalised 384-d vector."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _v(self, text: str) -> np.ndarray:
        v = np.zeros(384, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            rng = np.random.default_rng(abs(hash_w(w)))
            v += rng.normal(size=384).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, **_):
        if not texts:
            return np.zeros((0, 384), np.float32)
        return np.stack([self._v(t) for t in texts])

    def encode_queries(self, texts):
        return self.encode(texts)


def hash_w(w: str) -> int:
    h = 1469598103934665603
    for ch in w.encode():
        h = ((h ^ ch) * 1099511628211) % (1 << 63)
    return h


@pytest.fixture
def emb():
    return HashEmbedder()


@pytest.fixture
def log():
    try:
        psycopg.connect(database_url(), connect_timeout=3).close()
    except psycopg.Error:
        pytest.skip("postgres not reachable")
    lg = WriteLog()
    lg.reset()
    yield lg
    lg.close()
