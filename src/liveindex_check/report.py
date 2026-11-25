"""Result file and plain-language failure rendering."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from liveindex_check.machine import FailureHolder, Stats


def utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    v = sorted(values)
    rank = -(-q * len(v) // 100)  # ceil(q/100 * n)
    return v[min(len(v) - 1, max(0, int(rank) - 1))]


def metrics(stats: Stats) -> dict:
    q = max(stats.queries, 1)
    leak = stats.violations.get("I3", 0) + stats.violations.get("I4", 0)
    stale = stats.violations.get("I1", 0) + stats.violations.get("I2", 0)
    all_tti = [t for v in stats.tti.values() for t in v]
    return {
        "ops": stats.ops,
        "queries": stats.queries,
        "examples": stats.examples,
        "violations": dict(sorted(stats.violations.items())),
        "leak_rate": leak / q,
        "stale_hit_rate": stale / q,
        "time_to_invisible": {
            "samples": len(all_tti),
            "p50": percentile(all_tti, 50),
            "p95": percentile(all_tti, 95),
            "p99": percentile(all_tti, 99),
            "by_kind": {
                k: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95)} for k, v in stats.tti.items()
            },
        },
    }


def render_steps(steps: list[dict]) -> list[str]:
    out = []
    for n, s in enumerate(steps, 1):
        op = s["op"]
        if op == "upsert_new":
            out.append(f"{n}. create {s['doc']} v1 allowed to {s['allowed']}")
        elif op == "upsert_update":
            out.append(f"{n}. update {s['doc']} to v{s['version']}")
        elif op == "delete":
            out.append(f"{n}. delete {s['doc']}")
        elif op == "revoke":
            out.append(f"{n}. revoke {s['principal']} on {s['doc']}")
        elif op == "grant":
            out.append(f"{n}. grant {s['principal']} on {s['doc']}")
        elif op == "search_present":
            out.append(f"{n}. search as {s['principal']} for {s['doc']}'s text (expect present)")
        elif op == "search_absent":
            out.append(
                f"{n}. search as {s['principal']} for the text of {s['kind']} {s['doc']} v{s['version']} (expect absent)"
            )
        else:
            out.append(f"{n}. search as {s['principal']} for {s['doc']}'s text without settling")
    return out


def write_failure(system: str, holder: FailureHolder, seed: int, results_dir: Path) -> Path | None:
    if holder.violation is None:
        return None
    d = results_dir / "failures"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{system}_{utc()}.json"
    payload = {
        "system": system,
        "git_sha": git_sha(),
        "seed": seed,
        "violation": {"invariant": holder.violation.invariant, "detail": holder.violation.detail},
        "steps": holder.steps,
        "plain": render_steps(holder.steps),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_report(system: str, payload: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc()
    path = results_dir / f"suite_{system}_{stamp}.json"
    payload = {"system": system, "utc": stamp, "git_sha": git_sha(), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
