import itertools

from liveindex.chunking import chunk_text


def wc(s: str) -> int:
    return len(s.split())


def make(n_sent: int, words: int = 10) -> str:
    return " ".join(
        " ".join(f"w{i}x{j}" for j in range(words)).capitalize() + "." for i in range(n_sent)
    )


def test_short_text_single_chunk():
    t = make(3)
    assert chunk_text(t, wc, 300) == [t]


def test_empty():
    assert chunk_text("   ", wc) == []


def test_no_empty_chunks_and_size_bound():
    cs = chunk_text(make(80), wc, target=50, overlap=0.15)
    assert len(cs) > 1
    assert all(c.strip() for c in cs)
    assert all(wc(c) <= 50 for c in cs)


def test_overlap_respected_and_covers_text():
    t = make(80, words=4)
    cs = chunk_text(t, wc, target=50, overlap=0.15)
    assert len(cs) > 1
    for a, b in itertools.pairwise(cs):
        assert set(a.split()[-8:]) & set(b.split()[:8]), "adjacent chunks must overlap"
    assert set(t.split()) == {w for c in cs for w in c.split()}


def test_deterministic():
    t = make(60)
    assert chunk_text(t, wc, 40) == chunk_text(t, wc, 40)


def test_overlong_sentence_split():
    t = " ".join(f"t{i}" for i in range(200)) + "."
    cs = chunk_text(t, wc, target=50)
    assert len(cs) >= 4 and all(wc(c) <= 50 for c in cs)
