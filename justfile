set shell := ["powershell.exe", "-NoProfile", "-Command"]

# Postgres 18 already runs on localhost:5432 (no Docker); `up` only verifies it.
up:
    uv run python scripts/migrate.py check

down:
    Write-Host "nothing to stop: Postgres is an external service"

migrate:
    uv run python scripts/migrate.py

test:
    uv run pytest -q

lint:
    uv run ruff check .
    uv run mypy src

eval DATASET="scifact" RETRIEVER="all":
    Write-Error "eval: not implemented (prompt 1.1)"; exit 1

suite SYSTEM="liveindex" EXPECT="pass":
    Write-Error "suite: not implemented (prompt 1.5)"; exit 1

suite-all:
    Write-Error "suite-all: not implemented (prompt 2.2)"; exit 1

scale N="100000":
    Write-Error "scale: not implemented (prompt 2.3)"; exit 1

bench:
    Write-Error "bench: not implemented (prompt 2.3)"; exit 1

check: lint test

report:
    Write-Error "report: not implemented (prompt 2.8)"; exit 1
