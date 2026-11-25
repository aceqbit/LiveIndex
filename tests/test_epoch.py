import time

from liveindex.indexer import Indexer
from liveindex.search import Searcher


def make(log, emb):
    ix = Indexer(log, emb)
    log.on_commit = ix.wake
    return ix, Searcher(ix, emb)


def test_delete_visibility_and_time_travel(log, emb):
    ix, s = make(log, emb)
    e1 = log.upsert("d1", "quokka habitats on rottnest island", allowed=["u"])
    ix.catch_up()
    e2 = log.delete("d1")
    ix.catch_up()
    assert (e1, e2) == (1, 2)
    _, hits, _ = s.search("quokka habitats", principal="u")
    assert hits == []
    _, hits, _ = s.search("quokka habitats", principal="u", at_epoch=e2 - 1)
    assert [h["doc_id"] for h in hits] == ["d1"]
    _, hits, _ = s.search("quokka habitats", principal="u", at_epoch=e2)
    assert hits == []


def test_update_swaps_versions_in_same_epoch(log, emb):
    ix, s = make(log, emb)
    e1 = log.upsert("d", "alpha beta gamma original text", allowed=["u"])
    e2 = log.upsert("d", "delta epsilon zeta revised text")
    ix.catch_up()
    old = [c for c in range(ix._n) if ix.chunk_version[c] == 1]
    new = [c for c in range(ix._n) if ix.chunk_version[c] == 2]
    assert all(ix._dead[c] == e2 for c in old) and all(ix._born[c] == e2 for c in new)
    _, hits, _ = s.search("original", principal="u", at_epoch=e1)
    assert hits and hits[0]["version"] == 1
    _, hits, _ = s.search("revised", principal="u")
    assert hits and all(h["version"] == 2 for h in hits)
    _, hits, _ = s.search("original", principal="u")
    assert all(h["version"] == 2 for h in hits)


def test_restart_replays_to_same_state(log, emb):
    ix, s = make(log, emb)
    log.upsert("a", "red fox jumps", allowed=["u"])
    log.upsert("b", "blue whale sings", allowed=["u"])
    log.delete("a")
    log.upsert("b", "blue whale sings louder", allowed=["u"])
    ix.catch_up()
    before = s.search("whale", principal="u")
    t0 = time.perf_counter()
    ix2 = Indexer(log, emb)
    ix2.catch_up()
    replay = time.perf_counter() - t0
    after = Searcher(ix2, emb).search("whale", principal="u")
    assert ix2.index_epoch == ix.index_epoch == 4
    assert [(h["doc_id"], h["version"]) for h in before[1]] == [(h["doc_id"], h["version"]) for h in after[1]]
    assert (ix2._dead[: ix2._n] == ix._dead[: ix._n]).all()
    assert replay < 5


def test_min_epoch_waits_for_indexer(log, emb):
    ix, s = make(log, emb)
    ix.start()
    try:
        e = log.upsert("a", "waiting for the indexer", allowed=["u"])
        _, hits, _ = s.search("indexer", principal="u", min_epoch=e)
        assert hits and hits[0]["epoch"] >= e
    finally:
        ix.stop()


def test_epochs_are_dense_and_wal_ordered(log, emb):
    es = [log.upsert(f"d{i}", f"text {i}", allowed=["u"]) for i in range(5)]
    assert es == [1, 2, 3, 4, 5]
    assert [r.epoch for r in log.read_after(0)][:3] == [1, 1, 2]  # upsert + acl_grant per epoch


def test_noop_write_still_advances_index_epoch(log, emb):
    """Regression (found by the suite): a set_acl that changes nothing bumps the epoch but writes no
    WAL rows; the indexer must still reach that epoch or min_epoch waits time out."""
    ix, s = make(log, emb)
    log.upsert("d", "some text here", allowed=["u"])
    e = log.set_acl("d", ["u"])  # no-op
    assert e == 2
    ix.catch_up()
    assert ix.index_epoch == 2
    assert s.search("text", principal="u", min_epoch=e, timeout=0.5)[1]
