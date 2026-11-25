import numpy as np

from liveindex.index.hnsw import HNSWIndex
from liveindex.search import rrf


def _vecs(n, d=16, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_filter_and_mark_deleted_supported():
    v = _vecs(200)
    h = HNSWIndex(dim=16, capacity=10)  # forces resize
    h.add(v)
    mask = np.arange(200) % 2 == 0
    hits, exact = h.knn(v[1], 10, mask)
    assert not exact and all(c % 2 == 0 for c, _ in hits)
    h.mark_deleted(hits[0][0])
    labels = h._idx.knn_query(v[1][None], k=10)[0][0].tolist()
    assert hits[0][0] not in labels


def test_exact_fallback_matches_brute_force():
    v = _vecs(300, seed=1)
    h = HNSWIndex(dim=16)
    h.add(v)
    mask = np.zeros(300, dtype=bool)
    mask[[5, 50, 120, 250]] = True  # 1.3% < 5%
    hits, exact = h.knn(v[7], 3, mask)
    assert exact
    ids = np.flatnonzero(mask)
    want = ids[np.argsort(-(v[ids] @ v[7]))][:3]
    assert [c for c, _ in hits] == want.tolist()


def test_rrf_scores():
    out = dict(rrf([[1, 2], [2, 3]]))
    assert out[2] > out[1] > out[3]
    assert abs(out[2] - (1 / 62 + 1 / 61)) < 1e-12
