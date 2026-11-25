"""Apply db/migrations/*.sql in order, recording applied files in schema_migrations."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/liveindex"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def migrate() -> list[str]:
    applied: list[str] = []
    with psycopg.connect(database_url(), autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted((ROOT / "db" / "migrations").glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


def check() -> None:
    with psycopg.connect(database_url(), connect_timeout=5) as conn:
        row = conn.execute("SHOW server_version").fetchone()
        print(f"postgres reachable, server_version={row[0] if row else '?'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        print("applied:", migrate() or "nothing new")
