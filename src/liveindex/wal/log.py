"""Write path: one Postgres transaction per write = document rows + WAL rows + epoch bump."""
from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

from liveindex.wal.epoch import INF, next_epoch

DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/liveindex"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


@dataclass(frozen=True)
class WalRow:
    epoch: int
    seq: int
    op: str
    doc_id: str
    version: int | None
    payload: dict


class WriteLog:
    """All writes go through here. `on_commit` is called after the transaction is durable."""

    def __init__(self, url: str | None = None, on_commit: Callable[[int], None] | None = None):
        self.url = url or database_url()
        self._conn = psycopg.connect(self.url, autocommit=True)
        self._lock = threading.Lock()
        self.on_commit = on_commit

    def close(self) -> None:
        self._conn.close()

    # ---- helpers (run inside a transaction) ----
    @staticmethod
    def _wal(tx, e: int, seq: int, op: str, doc_id: str, version: int | None, payload: dict) -> None:
        tx.execute(
            "INSERT INTO wal (epoch, seq, op, doc_id, version, payload) VALUES (%s,%s,%s,%s,%s,%s)",
            (e, seq, op, doc_id, version, Jsonb(payload)),
        )

    def _acl_diff(self, tx, e: int, seq: int, doc_id: str, allowed: set[str]) -> int:
        current = {
            r[0]
            for r in tx.execute(
                "SELECT principal FROM acl_entries WHERE doc_id=%s AND revoked_epoch=%s", (doc_id, INF)
            )
        }
        for p in sorted(allowed - current):
            tx.execute("INSERT INTO acl_entries (doc_id, principal, granted_epoch) VALUES (%s,%s,%s)", (doc_id, p, e))
            self._wal(tx, e, seq, "acl_grant", doc_id, None, {"principal": p})
            seq += 1
        for p in sorted(current - allowed):
            tx.execute(
                "UPDATE acl_entries SET revoked_epoch=%s WHERE doc_id=%s AND principal=%s AND revoked_epoch=%s",
                (e, doc_id, p, INF),
            )
            self._wal(tx, e, seq, "acl_revoke", doc_id, None, {"principal": p})
            seq += 1
        return seq

    def _commit(self, fn: Callable[[psycopg.Connection, int], None]) -> int:
        with self._lock, self._conn.transaction():
            e = next_epoch(self._conn)
            fn(self._conn, e)
        if self.on_commit:
            self.on_commit(e)
        return e

    # ---- public writes ----
    def upsert(self, doc_id: str, text: str, tenant: str = "default", allowed: Iterable[str] | None = None) -> int:
        def fn(tx, e):
            row = tx.execute("SELECT max(version) FROM documents WHERE doc_id=%s", (doc_id,)).fetchone()
            version = (row[0] or 0) + 1
            tx.execute(
                "UPDATE documents SET dead_epoch=%s WHERE doc_id=%s AND dead_epoch=%s", (e, doc_id, INF)
            )
            tx.execute(
                "INSERT INTO documents (doc_id, version, tenant, text, born_epoch) VALUES (%s,%s,%s,%s,%s)",
                (doc_id, version, tenant, text, e),
            )
            self._wal(tx, e, 0, "upsert", doc_id, version, {"text": text, "tenant": tenant})
            if allowed is not None:
                self._acl_diff(tx, e, 1, doc_id, set(allowed))

        return self._commit(fn)

    def delete(self, doc_id: str) -> int:
        def fn(tx, e):
            n = tx.execute(
                "UPDATE documents SET dead_epoch=%s WHERE doc_id=%s AND dead_epoch=%s", (e, doc_id, INF)
            ).rowcount
            if n == 0:
                raise KeyError(doc_id)
            self._wal(tx, e, 0, "delete", doc_id, None, {})

        return self._commit(fn)

    def set_acl(self, doc_id: str, allowed: Iterable[str]) -> int:
        def fn(tx, e):
            if not tx.execute("SELECT 1 FROM documents WHERE doc_id=%s AND dead_epoch=%s", (doc_id, INF)).fetchone():
                raise KeyError(doc_id)
            self._acl_diff(tx, e, 0, doc_id, set(allowed))

        return self._commit(fn)

    def group_add(self, grp: str, member: str) -> int:
        def fn(tx, e):
            tx.execute("INSERT INTO group_members (grp, member, added_epoch) VALUES (%s,%s,%s)", (grp, member, e))
            self._wal(tx, e, 0, "acl_grant", "", None, {"group": grp, "member": member})

        return self._commit(fn)

    def group_remove(self, grp: str, member: str) -> int:
        def fn(tx, e):
            tx.execute(
                "UPDATE group_members SET removed_epoch=%s WHERE grp=%s AND member=%s AND removed_epoch=%s",
                (e, grp, member, INF),
            )
            self._wal(tx, e, 0, "acl_revoke", "", None, {"group": grp, "member": member})

        return self._commit(fn)

    # ---- reads ----
    def committed_epoch(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT epoch FROM epoch_counter WHERE id=1").fetchone()
        return int(row[0]) if row else 0

    def read_after(self, epoch: int) -> list[WalRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT epoch, seq, op, doc_id, version, payload FROM wal WHERE epoch > %s ORDER BY epoch, seq",
                (epoch,),
            ).fetchall()
        return [WalRow(r[0], r[1], r[2], r[3], r[4], r[5] or {}) for r in rows]

    def reset(self) -> None:
        """Test/suite helper: wipe all state and restart epochs at 0."""
        with self._lock, self._conn.transaction():
            self._conn.execute("TRUNCATE wal, documents, acl_entries, group_members")
            self._conn.execute("UPDATE epoch_counter SET epoch = 0 WHERE id = 1")
