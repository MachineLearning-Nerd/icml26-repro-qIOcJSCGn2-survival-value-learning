#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-producers", action="store_true")
    args = parser.parse_args()
    if not args.skip_producers:
        print("Refusing to claim a producer run; pass --skip-producers for the scoped publication gate.", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[2]
    verifier = subprocess.run(
        [sys.executable, str(root / "repro/src/verify_results.py"), "--no-write"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if verifier.returncode != 0:
        print(verifier.stdout)
        print(verifier.stderr, file=sys.stderr)
        return verifier.returncode
    verification = json.loads(verifier.stdout)
    result = {
        "status": "SCOPED_PASS" if verification["status"] == "PASS" else "FAIL",
        "producers": {},
        "verification_status": verification["status"],
        "scope": "C2 exact identity and repository publication surface; not an official score.",
    }
    (root / "outputs/publication_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "SCOPED_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
