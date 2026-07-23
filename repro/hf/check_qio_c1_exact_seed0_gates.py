#!/usr/bin/env python3
"""Read-only, fail-closed gates for the qIO C1 exact-scale static plan.

This program cannot submit, mutate, cancel, or monitor a Hugging Face job.  Its
optional live gate executes separate account-wide read-only `hf jobs list`
queries for the RUNNING and SCHEDULING states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name("qio_c1_exact_seed0_contract.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def check_file(
    report: dict[str, Any],
    label: str,
    spec: dict[str, Any],
    *,
    required: bool = True,
) -> None:
    path = resolve_repo_path(spec["path"])
    item: dict[str, Any] = {"path": str(path), "required": required}
    if not path.is_file():
        item["status"] = "missing"
        report["checks"][label] = item
        if required:
            report["blockers"].append(f"{label}: required file is missing: {path}")
        return
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    item.update({"status": "present", "bytes": actual_bytes, "sha256": actual_sha256})
    expected_bytes = spec.get("bytes")
    expected_sha256 = spec.get("sha256", spec.get("file_sha256"))
    if expected_bytes is not None and actual_bytes != expected_bytes:
        item["status"] = "mismatch"
        report["blockers"].append(
            f"{label}: byte mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        item["status"] = "mismatch"
        report["blockers"].append(
            f"{label}: SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    report["checks"][label] = item


def check_protocol(report: dict[str, Any], contract: dict[str, Any]) -> None:
    spec = contract["lineage"]["protocol"]
    path = resolve_repo_path(spec["path"])
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["blockers"].append(f"protocol: cannot decode JSON: {exc}")
        return
    declared = payload.pop("sha256", None)
    semantic = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = spec["semantic_sha256"]
    report["checks"]["protocol_semantic_identity"] = {
        "declared_sha256": declared,
        "computed_sha256": semantic,
        "expected_sha256": expected,
    }
    if declared != expected or semantic != expected:
        report["blockers"].append(
            "protocol: declared or recomputed semantic SHA-256 does not match the contract"
        )
    expected_fields = {
        "dataset_name": contract["scope"]["environment"],
        "seed": contract["scope"]["seed"],
        "backend": "gpu",
        "hsvl_revision": contract["source_and_data"]["hsvl_commit"],
        "ogbench_revision": contract["source_and_data"]["ogbench_commit"],
    }
    for key, expected_value in expected_fields.items():
        if payload.get(key) != expected_value:
            report["blockers"].append(
                f"protocol: {key} mismatch: expected {expected_value!r}, found {payload.get(key)!r}"
            )
    if payload.get("dataset_hashes") != {
        "train": contract["source_and_data"]["train_data"]["sha256"],
        "validation": contract["source_and_data"]["validation_data"]["sha256"],
    }:
        report["blockers"].append("protocol: dataset hashes do not match the contract")


def check_checkpoint_links(report: dict[str, Any], contract: dict[str, Any]) -> None:
    manifest_path = resolve_repo_path(contract["lineage"]["checkpoint_manifest"]["path"])
    completion_path = resolve_repo_path(contract["lineage"]["completion"]["path"])
    if not manifest_path.is_file() or not completion_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["blockers"].append(f"checkpoint lineage: cannot decode JSON: {exc}")
        return
    checkpoint = contract["lineage"]["checkpoint"]
    protocol = contract["lineage"]["protocol"]
    required_manifest = {
        "active_checkpoint": Path(checkpoint["path"]).name,
        "checkpoint_bytes": checkpoint["bytes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "completed_updates": checkpoint["completed_updates"],
        "protocol_sha256": protocol["semantic_sha256"],
        "state_sha256": checkpoint["state_sha256"],
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            report["blockers"].append(
                f"checkpoint manifest: {key} mismatch: expected {expected!r}, found {manifest.get(key)!r}"
            )
    required_completion = {
        "status": "complete",
        "checkpoint_reload_exact": True,
        "completed_updates": 100,
        "target_updates": 100,
        "paper_scale_target_updates": 1000000,
        "checkpoint_sha256": checkpoint["sha256"],
        "state_sha256": checkpoint["state_sha256"],
        "restored_state_sha256": checkpoint["state_sha256"],
        "protocol_sha256": protocol["semantic_sha256"],
    }
    for key, expected in required_completion.items():
        if completion.get(key) != expected:
            report["blockers"].append(
                f"completion: {key} mismatch: expected {expected!r}, found {completion.get(key)!r}"
            )


def process_table() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for raw in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)$", raw)
        if match:
            rows.append((int(match.group(1)), match.group(2)))
    return rows


def check_local_nonduplication(report: dict[str, Any], contract: dict[str, Any]) -> None:
    try:
        rows = process_table()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["blockers"].append(f"local process gate: unable to inspect process table: {exc}")
        return
    protected = contract["nonduplication"]["protected_pids"]
    patterns = tuple(contract["nonduplication"]["block_if_command_contains"])
    matches: list[dict[str, Any]] = []
    for pid, command in rows:
        if pid == os.getpid():
            continue
        protected_expected = protected.get(str(pid))
        pattern_match = next((pattern for pattern in patterns if pattern in command), None)
        if protected_expected is not None or pattern_match is not None:
            matches.append({"pid": pid, "command": command})
            if protected_expected is not None and protected_expected not in command:
                report["blockers"].append(
                    f"protected PID {pid} exists but command identity changed; refuse closed"
                )
            else:
                report["blockers"].append(
                    f"protected or duplicate local work is active at PID {pid}; leave it untouched"
                )
    report["checks"]["local_nonduplication"] = {"active_matches": matches}


def parse_money(text: str) -> float | None:
    match = re.search(r"USD\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def check_central_reservation(report: dict[str, Any], contract: dict[str, Any]) -> None:
    spec = contract["central_reservation"]
    ledger = resolve_repo_path(spec["ledger_path"])
    check: dict[str, Any] = {"path": str(ledger), "reservation_id": spec["reservation_id"]}
    report["checks"]["central_reservation"] = check
    if not ledger.is_file():
        report["blockers"].append(f"central reservation: ledger missing: {ledger}")
        return
    text = ledger.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    matching = [line for line in rows if f"`{spec['reservation_id']}`" in line]
    if len(matching) != 1:
        check["matching_rows"] = len(matching)
        report["blockers"].append(
            f"central reservation: expected exactly one {spec['reservation_id']} row, found {len(matching)}"
        )
        return
    row = matching[0]
    check["row"] = row
    lowered = row.lower()
    required_tokens = [spec["required_flavor"], spec["purpose_substring"]]
    required_tokens.extend(spec["required_state_substrings"])
    for token in required_tokens:
        if token.lower() not in lowered:
            report["blockers"].append(f"central reservation: row lacks required token {token!r}")
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if len(cells) < 6:
        report["blockers"].append("central reservation: malformed ledger row")
        return
    runtime_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*h", cells[4], flags=re.IGNORECASE)
    runtime_hours = float(runtime_match.group(1)) if runtime_match else None
    committed = parse_money(cells[5])
    check.update({"runtime_hours": runtime_hours, "committed_usd": committed})
    if runtime_hours is None or runtime_hours > spec["max_runtime_hours"]:
        report["blockers"].append("central reservation: missing or excessive runtime ceiling")
    if committed is None or committed > spec["max_committed_usd"]:
        report["blockers"].append("central reservation: missing or excessive USD ceiling")
    cap_match = re.search(r"Authorized cap:\s*\*\*USD\s*([0-9.]+)", text)
    committed_match = re.search(r"Billed or committed:\s*\*\*USD\s*([0-9.]+)", text)
    cap = float(cap_match.group(1)) if cap_match else None
    total = float(committed_match.group(1)) if committed_match else None
    check.update({"ledger_cap_usd": cap, "ledger_billed_or_committed_usd": total})
    if cap is None or cap > spec["campaign_cap_usd"]:
        report["blockers"].append("central reservation: campaign cap missing or above authorization")
    if total is None or total > spec["campaign_cap_usd"]:
        report["blockers"].append("central reservation: billed/committed total missing or above cap")


def check_frozen_components(report: dict[str, Any], contract: dict[str, Any]) -> None:
    details: dict[str, Any] = {}
    for label, spec in contract["required_frozen_execution_components"].items():
        path = resolve_repo_path(spec["path"])
        expected = spec.get("sha256")
        item = {"path": str(path), "expected_sha256": expected, "status": spec.get("status")}
        if expected is None:
            report["blockers"].append(f"frozen component: {label} has no approved SHA-256")
        elif not path.is_file():
            report["blockers"].append(f"frozen component: {label} is missing")
        else:
            actual = sha256_file(path)
            item["actual_sha256"] = actual
            if actual != expected:
                report["blockers"].append(f"frozen component: {label} SHA-256 mismatch")
        details[label] = item
    report["checks"]["frozen_execution_components"] = details


def active_hf_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for status in ("RUNNING", "SCHEDULING"):
        command = [
            "hf",
            "jobs",
            "list",
            "--namespace",
            "DineshAI",
            "--status",
            status,
            "--limit",
            "100",
            "--format",
            "json",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        if isinstance(payload, list):
            jobs.extend(payload)
            continue
        if isinstance(payload, dict):
            for key in ("jobs", "items", "data"):
                if isinstance(payload.get(key), list):
                    jobs.extend(payload[key])
                    break
            else:
                raise ValueError(
                    f"unexpected hf jobs list JSON shape for status {status}"
                )
            continue
        raise ValueError(f"unexpected hf jobs list JSON shape for status {status}")
    return jobs


def flavor_of(job: dict[str, Any]) -> str:
    for key in ("flavor", "hardware", "hardware_flavor"):
        value = job.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested in ("flavor", "name", "id"):
                if isinstance(value.get(nested), str):
                    return value[nested]
    return ""


def check_live_hf(report: dict[str, Any]) -> None:
    try:
        jobs = active_hf_jobs()
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        report["blockers"].append(f"live HF gate: read-only account query failed: {exc}")
        report["checks"]["live_hf_t4_exclusion"] = {"query_succeeded": False}
        return
    active_t4 = [job for job in jobs if flavor_of(job).lower().startswith("t4")]
    report["checks"]["live_hf_t4_exclusion"] = {
        "query_succeeded": True,
        "active_job_count": len(jobs),
        "active_t4_jobs": active_t4,
    }
    if active_t4:
        report["blockers"].append(
            f"live HF gate: {len(active_t4)} running/scheduling T4 job(s) exist; do not submit or mutate"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-live-hf",
        action="store_true",
        help=(
            "Perform one read-only account-wide HF active-state gate using "
            "separate RUNNING and SCHEDULING queries."
        ),
    )
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": str(CONTRACT_PATH),
        "mode": "read_only_gate_check",
        "hf_live_check_requested": args.check_live_hf,
        "checks": {},
        "blockers": [],
    }

    required_files = {
        "protocol": contract["lineage"]["protocol"],
        "checkpoint": contract["lineage"]["checkpoint"],
        "checkpoint_manifest": contract["lineage"]["checkpoint_manifest"],
        "completion": contract["lineage"]["completion"],
        "training_trace": contract["lineage"]["training_trace"],
        "sessions": contract["lineage"]["sessions"],
        "preflight_report": contract["lineage"]["preflight_report"],
        "preflight_integrity": contract["lineage"]["preflight_integrity"],
        "source_manifest": contract["source_and_data"]["source_manifest"],
        "uv_lock": contract["source_and_data"]["uv_lock"],
        "train_data": contract["source_and_data"]["train_data"],
        "validation_data": contract["source_and_data"]["validation_data"],
        "frozen_preflight_runner": contract["source_and_data"]["frozen_preflight_runner"],
    }
    for label, spec in required_files.items():
        check_file(report, label, spec)
    check_file(
        report,
        "returned_archive",
        contract["lineage"]["returned_archive"],
        required=contract["lineage"]["returned_archive"]["required_local"],
    )
    check_protocol(report, contract)
    check_checkpoint_links(report, contract)
    check_local_nonduplication(report, contract)
    check_central_reservation(report, contract)
    check_frozen_components(report, contract)
    if args.check_live_hf:
        check_live_hf(report)
    else:
        report["checks"]["live_hf_t4_exclusion"] = {
            "status": "not_checked",
            "required_immediately_before_submission": True,
        }

    report["ready_to_submit"] = not report["blockers"] and args.check_live_hf
    if not args.check_live_hf:
        report["blockers"].append(
            "live HF gate was not requested; a successful immediate pre-submit check is mandatory"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_to_submit"] else 4


if __name__ == "__main__":
    sys.exit(main())
