import re

import numpy as np
import pytest

from liveindex_check import machine
from liveindex_check.adapters import naive_ann_baseline as nb
from liveindex_check.cli import run_suite


def _hash_encode(self, text):
    v = np.zeros(nb.DIM, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = 1469598103934665603
        for ch in w.encode():
            h = ((h ^ ch) * 1099511628211) % (1 << 63)
        v += np.random.default_rng(h).normal(size=nb.DIM).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(nb._Embedder, "encode", _hash_encode)
    monkeypatch.setattr(machine.TIMING, "tti_s", 0.6)
    monkeypatch.setattr(machine.TIMING, "poll_s", 0.01)


def run(fixed, tmp_path, lag=0.2):
    a = nb.NaiveAnnBaseline(lag_s=lag, fixed=fixed)
    try:
        return run_suite(a, system="naive_fixed" if fixed else "naive", ops=400, steps=25, seeds=[1], results_dir=tmp_path)
    finally:
        a.close()


def test_default_baseline_fails_i1_and_i3(tmp_path):
    out = run(False, tmp_path)
    assert "I1" in out["violated"] and "I3" in out["violated"], out["violated"]
    assert out["shrunk_sequence"] and out["failure_file"]


def test_fixed_baseline_passes(tmp_path):
    out = run(True, tmp_path)
    assert out["violated"] == [], out["sample_violations"][:3]
