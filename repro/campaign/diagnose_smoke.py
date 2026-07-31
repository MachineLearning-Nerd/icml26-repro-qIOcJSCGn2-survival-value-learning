#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


CASES = [
    ("hsvl", "antmaze-giant-navigate-v0"),
    ("hiql", "humanoidmaze-giant-navigate-v0"),
    ("visual_hsvl", "visual-antmaze-giant-navigate-v0"),
    ("flat_svl", "humanoidmaze-giant-navigate-v0"),
    ("crl", "humanoidmaze-giant-navigate-v0"),
]


def main() -> int:
    for method, dataset in CASES:
        print(f"===== {method} {dataset} =====", flush=True)
        result = subprocess.run(
            [sys.executable, "repro/campaign/smoke_method.py", method, dataset],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout, flush=True)
        print(f"returncode={result.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
