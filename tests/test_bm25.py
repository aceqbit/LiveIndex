import math
import random

import numpy as np

from liveindex.index.bm25 import K1, B, BM25Index, tokenize

VOCAB = [f"w{i}" for i in range(60)] + ["the", "of", "and"]


def reference(corpus: list[str], query: str, mask: list[bool], top_n: int):
    docs = [tokenize(t) for t in corpus]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    out = []
    for i, d in enumerate(docs):
        if not mask[i]:
            continue
        s = 0.0
        for t in set(tokenize(query)):
            df = sum(1 for x in docs if t in x)
            tf = d.count(t)
            if tf == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(d) / avgdl))
        if s > 0:
            out.append((i, s))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:top_n]


def test_tokenizer_lowercases_and_drops_stopwords():
    assert tokenize("The Quick, brown FOX of 42!") == ["quick", "brown", "fox", "42"]


def test_matches_reference_on_random_corpora():
    rng = random.Random(7)
    for _ in range(5):
        corpus = [" ".join(rng.choices(VOCAB, k=rng.randint(3, 40))) for _ in range(150)]
        idx = BM25Index()
        idx.add_chunks(corpus[:70])
        idx.add_chunks(corpus[70:])
        for _q in range(20):
            query = " ".join(rng.choices(VOCAB, k=rng.randint(1, 4)))
            mask = [rng.random() < 0.7 for _ in corpus]
            got = idx.score(query, np.array(mask), 10)
            want = reference(corpus, query, mask, 10)
            assert [g[0] for g in got] == [w[0] for w in want]
            assert np.allclose([g[1] for g in got], [w[1] for w in want])


def test_masked_out_chunks_never_returned():
    idx = BM25Index()
    idx.add_chunks(["apple pie", "apple tart", "banana"])
    assert [c for c, _ in idx.score("apple", np.array([False, True, True]))] == [1]


def test_mask_shorter_than_index_hides_newer_chunks():
    idx = BM25Index()
    idx.add_chunks(["apple pie", "apple tart"])
    assert [c for c, _ in idx.score("apple", np.array([True]))] == [0]
