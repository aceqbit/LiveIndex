"""bge-small-en-v1.5 embeddings: batched, L2-normalised, with an on-disk cache keyed by text hash."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "cache" / "embeddings.sqlite"


def text_key(text: str, model: str = MODEL_NAME) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class Embedder:
    def __init__(self, cache_path: Path | None = DEFAULT_CACHE, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._tok = None
        self._db = None
        self._lock = threading.Lock()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def warm(self) -> None:
        """Load model and tokenizer now, on the calling thread (lazy imports are not thread-safe)."""
        _ = self.tokenizer
        self.model.encode(["warm up"], normalize_embeddings=True, show_progress_bar=False)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        return self._tok

    def count_tokens(self, text: str) -> int:
        with self._lock:
            return len(self.tokenizer.tokenize(text))

    def _get(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        if self._db is None:
            return found
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = ",".join("?" * len(part))
            for k, v in self._db.execute(f"SELECT k, v FROM emb WHERE k IN ({q})", part):
                found[k] = np.frombuffer(v, dtype=np.float32)
        return found

    def encode(self, texts: list[str], batch_size: int = 64, prefix: str = "") -> np.ndarray:
        """Return an (n, 384) float32 matrix of normalised embeddings."""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        full = [prefix + t for t in texts]
        keys = [text_key(t, self.model_name) for t in full]
        cached = self._get(list(set(keys)))
        todo = {}
        for k, t in zip(keys, full, strict=True):
            if k not in cached and k not in todo:
                todo[k] = t
        if todo:
            ks = list(todo)
            with self._lock:
                vecs = self.model.encode(
                    [todo[k] for k in ks], batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ).astype(np.float32)
            for k, v in zip(ks, vecs, strict=True):
                cached[k] = v
            if self._db is not None:
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?)", [(k, v.tobytes()) for k, v in zip(ks, vecs, strict=True)]
                )
                self._db.commit()
        return np.stack([cached[k] for k in keys])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, prefix=QUERY_PREFIX)
