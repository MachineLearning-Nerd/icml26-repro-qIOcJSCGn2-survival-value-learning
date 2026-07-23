#!/usr/bin/env python3
"""Fail-closed baseline evaluator for the qIO OpenResearch experiment tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / ".openresearch" / "judge_contract.json"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
ROUTE_AUDIT = ROOT / "repro" / "configs" / "route_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    claims = contract["claims"]
    if [claim["id"] for claim in claims] != [f"C{i}" for i in range(1, 7)]:
        raise RuntimeError("judge contract must contain exactly C1-C6 in order")

    points = sum(int(claim["current_points"]) for claim in claims)
    live = contract["live_verdict"]
    if points != live["score"] or live["maximum"] != 12:
        raise RuntimeError("claim points do not match the recorded live verdict")

    checks = [
        run([sys.executable, "-m", "pytest", "-q", "repro/tests/test_svl.py"]),
        run([sys.executable, "repro/src/verify_survival.py"]),
    ]
    checks_pass = all(check["returncode"] == 0 for check in checks)

    local_hashes = {}
    for relative in [
        "repro/src/verify_survival.py",
        "repro/src/svl.py",
        "repro/tests/test_svl.py",
        "outputs/svl_summary.json",
    ]:
        path = ROOT / relative
        local_hashes[relative] = sha256(path) if path.is_file() else None

    verdict_rows = [
        {
            "id": claim["id"],
            "current_verdict": claim["current_verdict"],
            "current_points": claim["current_points"],
            "candidate_change": "none",
        }
        for claim in claims
    ]
    report = {
        "schema_version": 1,
        "orid": contract["orid"],
        "judged_sha": live["judged_sha"],
        "live_score": live["score"],
        "maximum": live["maximum"],
        "claims": verdict_rows,
        "protected_c2_regression_pass": checks_pass,
        "checks": checks,
        "local_hashes": local_hashes,
        "release_candidate": False,
        "publication_authorized": False,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "seed_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (ARTIFACTS / "live_claim_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# qIOcJSCGn2 baseline evaluation",
        "",
        f"- Live judged revision: `{live['judged_sha']}`",
        f"- Live judged score: **{live['score']}/{live['maximum']}**",
        f"- Protected C2 regression: **{'PASS' if checks_pass else 'FAIL'}**",
        "- Candidate score: **not estimated**",
        "- Release candidate: **no**",
        "- HF publication authorized: **no**",
        "",
        "| Claim | Live verdict | Points | Candidate change |",
        "|---|---:|---:|---|",
    ]
    for row in verdict_rows:
        lines.append(
            f"| {row['id']} | {row['current_verdict']} | {row['current_points']} | none |"
        )
    lines.extend(
        [
            "",
            "This baseline intentionally reproduces the current judge contract. Passing the local",
            "checks preserves Claim 2; it does not upgrade any other claim.",
            "",
        ]
    )
    (ROOT / "EVAL.md").write_text("\n".join(lines))

    for check in checks:
        sys.stdout.write(check["stdout"])
        sys.stderr.write(check["stderr"])
    print(f"BASELINE_SCORE={live['score']}/{live['maximum']}")
    print(f"PROTECTED_C2_REGRESSION={'PASS' if checks_pass else 'FAIL'}")
    if ROUTE_AUDIT.is_file():
        from evaluate_route_audit import evaluate_route_audit

        return evaluate_route_audit(contract, checks_pass)
    return 0 if checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
