import importlib
import os
from pathlib import Path

import psycopg
import pytest

MODULES = [
    "liveindex.api.app", "liveindex.api.models", "liveindex.chunking", "liveindex.embed",
    "liveindex.wal.log", "liveindex.wal.epoch", "liveindex.index.bm25", "liveindex.index.hnsw",
    "liveindex.index.visibility", "liveindex.index.segments", "liveindex.acl",
    "liveindex.search", "liveindex.rerank", "liveindex.indexer", "liveindex.snapshot",
    "liveindex.telemetry", "liveindex.eval.metrics", "liveindex.eval.beir_runner",
    "liveindex_check.adapter", "liveindex_check.invariants", "liveindex_check.machine",
    "liveindex_check.report", "liveindex_check.cli",
    "liveindex_check.adapters.liveindex", "liveindex_check.adapters.qdrant",
    "liveindex_check.adapters.naive_ann_baseline",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_hnswlib_has_mark_deleted():
    import hnswlib

    assert hasattr(hnswlib.Index, "mark_deleted")


def test_migrations_applied():
    url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/liveindex")
    with psycopg.connect(url) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )
        }
        assert {"epoch_counter", "documents", "wal", "acl_entries", "group_members"} <= tables
        assert conn.execute("SELECT epoch FROM epoch_counter WHERE id=1").fetchone() is not None
    assert (Path(__file__).parent.parent / "db" / "migrations" / "001_init.sql").exists()
