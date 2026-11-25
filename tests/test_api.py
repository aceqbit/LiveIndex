import numpy as np
from fastapi.testclient import TestClient

from liveindex.api.app import create_app


class FakeEmb:
    def count_tokens(self, t):
        return len(t.split())

    def _v(self, t):
        v = np.zeros(384, dtype=np.float32)
        v[hash(t.split()[0]) % 384] = 1.0
        return v

    def encode(self, ts):
        return np.stack([self._v(t) for t in ts]) if ts else np.zeros((0, 384), np.float32)

    def encode_queries(self, ts):
        return self.encode(ts)


def test_upsert_and_search():
    c = TestClient(create_app(FakeEmb()))  # type: ignore[arg-type]
    c.post("/v1/docs", json={"doc_id": "a", "text": "apple pie recipe"})
    c.post("/v1/docs", json={"doc_id": "b", "text": "banana bread"})
    r = c.post("/v1/search", json={"query": "apple", "k": 2}).json()
    assert r["hits"][0]["doc_id"] == "a"
