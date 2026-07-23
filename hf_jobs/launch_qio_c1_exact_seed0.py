#!/usr/bin/env python3
"""Race-closing, exact-label launcher for the qIO C1 T4 route.

Default mode is read-only. Submission additionally requires --submit and two
exact approval environment variables. This program never cancels or alters an
existing job or local process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ROUTE = "qio-c1-exact-seed0-full-v1"
SPACE = "DineshAI/qIOcJSCGn2"
PARENT = "757e63d6377d6a3789657d75e354306c95d6e43f"
LABELS = {"paper": "qIOcJSCGn2", "claim": "C1", "phase": "exact-seed0-full", "route": ROUTE}
JOB_SCRIPT = ROOT / "repro/hf/run_qio_c1_exact_seed0_job.py"
INPUT_DIR = ROOT / "hf_jobs/qio-c1-exact-seed0-input"
GATE = ROOT / "repro/hf/check_qio_c1_exact_seed0_gates.py"
OPERATIONS = ROOT / "repro/hf/qio_c1_exact_seed0_operations_contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("jobs", "items", "data"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
    raise ValueError("unexpected HF Jobs JSON shape")


def route_jobs() -> list[dict]:
    command = ["hf", "jobs", "list", "--namespace", "DineshAI", "--all", "--limit", "0", "--format", "json"]
    for key, value in LABELS.items():
        command.extend(["--label", f"{key}={value}"])
    return records(json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout))


def stage(job: dict) -> str:
    return str(job.get("status", job.get("stage", ""))).upper()


def source_integrity() -> dict:
    path = INPUT_DIR / "qio-c1-exact-seed0-source-v1.integrity.json"
    value = json.loads(path.read_text())
    archive = INPUT_DIR / value["archive"]
    digest = sha256(archive)
    if digest != value["archive_sha256"] or value.get("frozen_manifest_file_count") != 31:
        raise ValueError("the frozen source input is missing or has drifted")
    operations = json.loads(OPERATIONS.read_text())
    for name, spec in operations["bound_files"].items():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"operational component drift: {name}")
    return {"source_bundle": value, "operations_contract_sha256": sha256(OPERATIONS)}


def full_gate() -> dict:
    result = subprocess.run([sys.executable, str(GATE), "--check-live-hf"], capture_output=True, text=True)
    report = json.loads(result.stdout)
    duplicates = [job for job in route_jobs() if stage(job) in {"COMPLETED", "RUNNING", "SCHEDULING"}]
    if duplicates:
        report["blockers"].append("exact route already has a COMPLETED/RUNNING/SCHEDULING job; retrieve it or wait")
    observed_parent = HfApi().space_info(SPACE).sha
    report["checks"]["space_parent"] = {"expected": PARENT, "observed": observed_parent}
    if observed_parent != PARENT:
        report["blockers"].append("Space parent drifted; refresh the repair contract instead of overwriting it")
    report["checks"]["route_duplicates"] = duplicates
    report["checks"]["source_integrity"] = source_integrity()
    report["ready_to_submit"] = not report["blockers"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    report = full_gate()
    if not args.submit:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready_to_submit"] else 4
    if os.environ.get("QIO_C1_EXECUTION_APPROVED") != "submit-exact-seed0-t4":
        raise PermissionError("QIO_C1_EXECUTION_APPROVED is absent or incorrect")
    if os.environ.get("QIO_C1_MAX_COST_USD") != "19.20":
        raise PermissionError("QIO_C1_MAX_COST_USD must be exactly 19.20")
    if report["blockers"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4

    # Close the admission race immediately before the only mutating command.
    final = full_gate()
    if final["blockers"]:
        print(json.dumps(final, indent=2, sort_keys=True))
        return 4
    token = "qio-c1-" + uuid.uuid4().hex
    command = [
        "hf", "jobs", "uv", "run", "--namespace", "DineshAI", "--detach",
        "--name", ROUTE, "--flavor", "t4-small", "--timeout", "48h",
        "--volume", f"{INPUT_DIR}:/job-input:ro",
        "--volume", "hf://buckets/DineshAI/qIOcJSCGn2-artifacts:/hf-bucket:rw",
    ]
    for key, value in {**LABELS, "submission": token}.items():
        command.extend(["--label", f"{key}={value}"])
    command.extend([
        str(JOB_SCRIPT), "--bucket-root", "/hf-bucket", "--input-root", "/job-input",
        "--submission-token", token,
    ])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(json.dumps({"job_cli_stdout": result.stdout.strip(), "submission_token": token, "labels": {**LABELS, "submission": token}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
