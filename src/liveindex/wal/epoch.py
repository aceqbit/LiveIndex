"""Epoch counter helpers. The counter row is locked by UPDATE, which serialises writers."""
from __future__ import annotations

import psycopg

INF = 9223372036854775807


def current_epoch(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT epoch FROM epoch_counter WHERE id = 1").fetchone()
    return int(row[0]) if row else 0


def next_epoch(conn: psycopg.Connection) -> int:
    """Increment inside the caller's transaction; the row lock is held until commit."""
    row = conn.execute("UPDATE epoch_counter SET epoch = epoch + 1 WHERE id = 1 RETURNING epoch").fetchone()
    assert row is not None
    return int(row[0])
