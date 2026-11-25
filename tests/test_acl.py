import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from liveindex.api.app import create_app
from liveindex.api.auth import resolve_principal, sign
from liveindex.index.hnsw import HNSWIndex
from liveindex.indexer import Indexer
from liveindex.search import Searcher


def make(log, emb):
    ix = Indexer(log, emb)
    return ix, Searcher(ix, emb)


def ids(res):
    return [h["doc_id"] for h in res[1]]


def test_revoke_hides_at_e_not_before(log, emb):
    ix, s = make(log, emb)
    e1 = log.upsert("d", "secret quarterly forecast figures", allowed=["alice"])
    e2 = log.set_acl("d", [])
    ix.catch_up()
    assert ids(s.search("quarterly forecast", principal="alice", at_epoch=e1)) == ["d"]
    assert ids(s.search("quarterly forecast", principal="alice", at_epoch=e2 - 1)) == ["d"]
    assert ids(s.search("quarterly forecast", principal="alice", at_epoch=e2)) == []
    assert ids(s.search("quarterly forecast", principal="alice")) == []
    assert ids(s.search("quarterly forecast", principal="bob", at_epoch=e1)) == []


def test_group_removal_hides_at_e(log, emb):
    ix, s = make(log, emb)
    log.upsert("d", "team roadmap for the launch", allowed=["eng"])
    e_add = log.group_add("eng", "alice")
    e_rm = log.group_remove("eng", "alice")
    ix.catch_up()
    assert ids(s.search("roadmap launch", principal="alice", at_epoch=e_add)) == ["d"]
    assert ids(s.search("roadmap launch", principal="alice", at_epoch=e_rm - 1)) == ["d"]
    assert ids(s.search("roadmap launch", principal="alice", at_epoch=e_rm)) == []


def test_two_groups_union(log, emb):
    ix, s = make(log, emb)
    log.upsert("a", "alpha project notes", allowed=["g1"])
    log.upsert("b", "alpha budget notes", allowed=["g2"])
    log.upsert("c", "alpha private notes", allowed=["g3"])
    log.group_add("g1", "u")
    log.group_add("g2", "u")
    ix.catch_up()
    assert set(ids(s.search("alpha notes", principal="u"))) == {"a", "b"}


def test_deny_by_default(log, emb):
    ix, s = make(log, emb)
    log.upsert("d", "no acl entries at all")
    ix.catch_up()
    assert ids(s.search("acl entries", principal="anyone")) == []


def test_exact_fallback_with_tiny_allowed_set(log, emb):
    ix, s = make(log, emb)
    for i in range(60):
        log.upsert(f"d{i}", f"filler document number {i} about topic {i % 7}", allowed=["many"])
    log.upsert("rare", "filler document about topic rare", allowed=["solo"])
    ix.catch_up()
    _e, hits, meta = s.search("filler document topic", k=5, principal="solo")
    assert meta["fallback_exact"] is True
    assert [h["doc_id"] for h in hits] == ["rare"]


def test_filtered_knn_matches_brute_force_at_tiny_selectivity():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(500, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    h = HNSWIndex(dim=32)
    h.add(v)
    mask = np.zeros(500, dtype=bool)
    mask[rng.choice(500, 10, replace=False)] = True  # 2%
    got, exact = h.knn(v[0], 5, mask)
    ids_ = np.flatnonzero(mask)
    want = ids_[np.argsort(-(v[ids_] @ v[0]))][:5]
    assert exact and [c for c, _ in got] == want.tolist()


def test_token_roundtrip_and_forgery(monkeypatch):
    monkeypatch.setenv("LIVEINDEX_TOKEN_SECRET", "s3cret")
    monkeypatch.delenv("ALLOW_TEST_PRINCIPAL", raising=False)
    assert resolve_principal(f"Bearer {sign('alice')}", None) == "alice"
    with pytest.raises(HTTPException):
        resolve_principal("Bearer alice.deadbeef", None)
    with pytest.raises(HTTPException):
        resolve_principal(None, "alice")  # header ignored unless ALLOW_TEST_PRINCIPAL=1


def test_principal_in_body_is_ignored(log, emb, monkeypatch):
    monkeypatch.delenv("ALLOW_TEST_PRINCIPAL", raising=False)
    with TestClient(create_app(emb, log)) as c:
        r = c.post("/v1/search", json={"query": "x", "principal": "alice"})
        assert r.status_code == 401


def test_search_py_has_no_post_filter():
    import pathlib
    import re

    src = pathlib.Path("src/liveindex/search.py").read_text(encoding="utf-8")
    assert not re.search(r"post_?filter|filter_after|\.filter\(|postfilter", src, re.IGNORECASE)
    # the mask must be handed to the candidate generators, never applied to their output
    assert "bm25.score(query, mask" in src and "hnsw.knn(qvec, 100, mask)" in src
