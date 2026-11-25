"""Run the conformance suite; for `liveindex`, start the API (test principals on) if it is not up."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

from liveindex_check.cli import main as suite_main

URL = "http://127.0.0.1:8000"


def up() -> bool:
    try:
        urllib.request.urlopen(URL + "/v1/epoch", timeout=2).read()
        return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="liveindex")
    args, rest = ap.parse_known_args()
    proc = None
    if args.system == "liveindex" and not up():
        env = {**os.environ, "ALLOW_TEST_PRINCIPAL": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "--factory", "liveindex.api.app:create_app", "--port", "8000", "--log-level", "warning"],
            env=env,
        )
        for _ in range(240):
            if up():
                break
            time.sleep(0.5)
        else:
            proc.kill()
            raise SystemExit("API did not start")
    try:
        return suite_main(["--system", args.system, *rest, *(["--url", URL] if args.system == "liveindex" else [])])
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
