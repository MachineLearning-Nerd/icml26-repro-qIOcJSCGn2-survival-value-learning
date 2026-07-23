#!/usr/bin/env python3
"""Import one terminal exact-route job and recompute all returned evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
ROUTE = "qio-c1-exact-seed0-full-v1"
SPACE = "DineshAI/qIOcJSCGn2"
PARENT = "757e63d6377d6a3789657d75e354306c95d6e43f"
LABELS = {"paper": "qIOcJSCGn2", "claim": "C1", "phase": "exact-seed0-full", "route": ROUTE}
VERIFIER = ROOT / "repro/src/verify_qio_c1_exact_seed0_return.py"
IMPORT_ROOT = ROOT / "outputs/hf/qio-c1-exact-seed0-terminal-imports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(value: object) -> list[dict]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in ("jobs", "items", "data"):
            if isinstance(value.get(key), list):
                return [x for x in value[key] if isinstance(x, dict)]
    raise ValueError("unexpected HF Jobs JSON shape")


def labels_of(job: dict) -> dict[str, str]:
    value = job.get("labels", {})
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    output = {}
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and "=" in item:
                key, val = item.split("=", 1)
                output[key] = val
    return output


def job_id(job: dict) -> str:
    return str(job.get("id", job.get("job_id", "")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--submission-token", required=True)
    args = parser.parse_args()
    command = ["hf", "jobs", "list", "--namespace", "DineshAI", "--all", "--limit", "0", "--format", "json"]
    for key, value in LABELS.items():
        command.extend(["--label", f"{key}={value}"])
    jobs = records(json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout))
    active = [j for j in jobs if str(j.get("status", j.get("stage", ""))).upper() in {"RUNNING", "SCHEDULING"}]
    completed = [j for j in jobs if str(j.get("status", j.get("stage", ""))).upper() == "COMPLETED"]
    if active or len(completed) != 1:
        raise ValueError("route must have exactly one COMPLETED job and no RUNNING/SCHEDULING duplicate")
    job = completed[0]
    if job_id(job) != args.job_id or labels_of(job).get("submission") != args.submission_token:
        raise ValueError("completed job identity or submission token is wrong")
    if HfApi().space_info(SPACE).sha != PARENT:
        raise ValueError("Space parent drifted; do not prepare or publish a repair")

    remote = f"hf://buckets/DineshAI/qIOcJSCGn2-artifacts/hf-jobs/{ROUTE}"
    with tempfile.TemporaryDirectory(prefix="qio-c1-terminal-") as temporary:
        snapshot = Path(temporary) / "snapshot"
        subprocess.run(["hf", "buckets", "sync", remote, str(snapshot)], check=True)
        receipt_path = snapshot / "attempts" / args.submission_token / "route_receipt.json"
        results = snapshot / "state" / "results"
        if not receipt_path.is_file() or not (results / "completion.json").is_file():
            raise ValueError("partial bucket return: route receipt or completion is absent")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "COMPLETED" or receipt.get("submission_token") != args.submission_token:
            raise ValueError("bucket route receipt is not the selected terminal job")
        if receipt.get("hf_job_id") not in (None, "", args.job_id):
            raise ValueError("bucket-reported HF job identity differs from selected job")
        if receipt.get("completion_sha256") != sha256(results / "completion.json"):
            raise ValueError("completion binding mismatch")
        verification = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(VERIFIER), "--results-dir", str(results.resolve()), "--import-root", str(IMPORT_ROOT.resolve())],
            check=True, capture_output=True, text=True,
        )
        verified = json.loads(verification.stdout)
        destination = Path(verified["immutable_import_destination"])
        receipt_out = destination / "terminal_import_receipt.json"
        receipt_out.write_text(json.dumps({
            "schema_version": 1, "job_id": args.job_id, "submission_token": args.submission_token,
            "route": ROUTE, "space_parent_sha": PARENT, "route_receipt_sha256": sha256(receipt_path),
            "completion_sha256": sha256(results / "completion.json"), "local_recomputation": verified,
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "VERIFIED_AND_IMMUTABLY_IMPORTED", "destination": str(destination), "receipt_sha256": sha256(receipt_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
