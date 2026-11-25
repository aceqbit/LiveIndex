"""`liveindex-check`: run the conformance suite against a system and enforce --expect."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

from liveindex_check.machine import TIMING, FailureHolder, Stats, run_machine
from liveindex_check.report import metrics, render_steps, write_failure, write_report

SYSTEMS = ["liveindex", "naive_ann_baseline", "qdrant"]


def run_suite(adapter, *, system: str, ops: int, steps: int, seeds: list[int], results_dir: Path) -> dict:
    stats = Stats()
    holder = FailureHolder()
    per_seed = max(1, ops // len(seeds))
    for seed in seeds:  # measurement pass: count everything, never raise
        random.seed(seed)
        run_machine(
            adapter, stats, holder, max_examples=max(1, per_seed // steps), steps=steps, raise_on_violation=False
        )
    m = metrics(stats)
    failure_path = None
    shrunk: list[str] = []
    if stats.violations:  # shrink pass: Hypothesis minimises a failing sequence
        shrink_stats, shrink_holder = Stats(), FailureHolder()
        random.seed(seeds[0])
        try:
            run_machine(
                adapter, shrink_stats, shrink_holder, max_examples=max(50, per_seed // steps), steps=steps,
                raise_on_violation=True,
            )
        except Exception as exc:  # noqa: BLE001 - Hypothesis re-raises the shrunk failure; holder has the steps
            print("shrink pass ended with:", type(exc).__name__, file=sys.stderr)
        use = shrink_holder if shrink_holder.violation else holder
        failure_path = write_failure(system, use, seeds[0], results_dir)
        shrunk = render_steps(use.steps)
    return {
        **m,
        "seeds": seeds,
        "steps_per_example": steps,
        "violated": sorted(stats.violations),
        "shrunk_sequence": shrunk,
        "failure_file": str(failure_path) if failure_path else None,
        "sample_violations": stats.samples,
    }


def build_adapter(name: str, args):
    mod = importlib.import_module(f"liveindex_check.adapters.{name}")
    return mod.make(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="liveindex-check")
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--expect", choices=["pass", "fail"], default="pass")
    ap.add_argument("--ops", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--url", default=None)
    ap.add_argument("--fixed", action="store_true", help="baseline only: lag 0 and pre-filter")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--settle-timeout", type=float, default=60.0, help="seconds to wait for a write to settle")
    ap.add_argument("--tti-timeout", type=float, default=5.0, help="seconds to poll for invisibility")
    args = ap.parse_args(argv)
    TIMING.tti_s = args.tti_timeout
    TIMING.settle_s = args.settle_timeout
    adapter = build_adapter(args.system, args)
    name = args.system + ("_fixed" if args.fixed else "")
    out = run_suite(
        adapter, system=name, ops=args.ops, steps=args.steps, seeds=args.seeds, results_dir=Path(args.results_dir)
    )
    violated = bool(out["violated"])
    out["expect"] = args.expect
    path = write_report(name, out, Path(args.results_dir))
    keys = ("ops", "queries", "violations", "leak_rate", "stale_hit_rate", "time_to_invisible")
    print(json.dumps({k: out[k] for k in keys}, indent=2))
    for line in out["shrunk_sequence"]:
        print("  ", line)
    print("report:", path)
    if args.expect == "pass":
        return 1 if violated else 0
    return 0 if violated else 1


if __name__ == "__main__":
    sys.exit(main())
