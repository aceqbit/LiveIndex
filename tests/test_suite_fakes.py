import ast
import pathlib

import pytest

from fakes import FakeAdapter
from liveindex_check.cli import run_suite


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch):
    from liveindex_check import machine

    monkeypatch.setattr(machine.TIMING, "tti_s", 0.05)
    monkeypatch.setattr(machine.TIMING, "poll_s", 0.01)


def run(bug, tmp_path):
    return run_suite(
        FakeAdapter(bug), system=f"fake_{bug}", ops=600, steps=30, seeds=[1], results_dir=tmp_path
    )


def test_correct_fake_passes(tmp_path):
    out = run(None, tmp_path)
    assert out["violated"] == [] and out["ops"] > 100 and out["queries"] > 100
    assert out["leak_rate"] == 0 and out["stale_hit_rate"] == 0


@pytest.mark.parametrize("bug", ["I1", "I2", "I3", "I4"])
def test_broken_fake_fails_exactly_its_invariant(bug, tmp_path):
    out = run(bug, tmp_path)
    assert out["violated"] == [bug], out["violated"]
    assert out["failure_file"] and pathlib.Path(out["failure_file"]).exists()
    assert out["shrunk_sequence"]
    print(bug, "->", out["violations"], out["shrunk_sequence"])


def test_shrunk_sequence_is_short(tmp_path):
    out = run("I1", tmp_path)
    assert len(out["shrunk_sequence"]) <= 4  # create, delete, (search)


def test_suite_never_imports_engine():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "liveindex_check"
    for f in root.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not (n == "liveindex" or n.startswith("liveindex.")), f"{f.name} imports {n}"
