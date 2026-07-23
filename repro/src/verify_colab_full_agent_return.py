#!/usr/bin/env python3
"""Independently verify a returned qIO Colab archive without importing JAX.

The verifier never extracts or unpickles the checkpoint.  It validates the
archive envelope, raw metrics, GPU/session/resume ledger, canonical protocol,
official source/data identities, and the active checkpoint byte hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any

from audit_full_agent_protocol import verify_upstream_source


PAPER_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DATASET_HASHES = {
    "train": "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    "validation": "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
}
EXPECTED_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"
EXPECTED_INPUT_INTEGRITY = (
    PAPER_ROOT
    / "outputs"
    / "colab_handoff"
    / "qio-full-agent-colab-input-v1.tar.gz.integrity.json"
)
EXPECTED_INPUT_ARCHIVE = (
    PAPER_ROOT
    / "outputs"
    / "colab_handoff"
    / "qio-full-agent-colab-input-v1.tar.gz"
)


def strict_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def safe_archive_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def read_and_hash(handle: Any, retain: bool) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    retained = bytearray() if retain else None
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
        if retained is not None:
            retained.extend(chunk)
    return digest.hexdigest(), size, bytes(retained) if retained is not None else None


def verify_archive(
    archive_path: Path,
    integrity_path: Path,
    input_archive_path: Path,
    input_integrity_path: Path,
) -> dict[str, Any]:
    integrity = strict_json(integrity_path.read_bytes())
    archive_sha256 = sha256_file(archive_path)
    if (
        integrity.get("schema_version") != 1
        or integrity.get("archive") != archive_path.name
        or integrity.get("archive_sha256") != archive_sha256
        or integrity.get("archive_bytes") != archive_path.stat().st_size
    ):
        raise ValueError("return archive integrity sidecar mismatch")

    retained_names = {
        "RETURN_MANIFEST.json",
        "results/COMPLETE.json",
        "results/protocol.json",
        "results/training_trace.json",
        "results/sessions.json",
        "results/checkpoint.manifest.json",
        "provenance/BUNDLE_MANIFEST.json",
        "provenance/upstream_source_manifest.json",
        "provenance/run_full_agent_probe_colab.py",
        "provenance/run_full_agent_probe.py",
        "provenance/audit_full_agent_protocol.py",
        "provenance/package_colab_return.py",
    }
    observed: dict[str, dict[str, Any]] = {}
    retained: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or not all(safe_archive_name(name) for name in names):
            raise ValueError("return archive has duplicate or unsafe member names")
        for member in members:
            if not member.isfile():
                raise ValueError(f"return archive contains a non-file: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read return member: {member.name}")
            digest, size, payload = read_and_hash(handle, member.name in retained_names)
            observed[member.name] = {"sha256": digest, "bytes": size}
            if payload is not None:
                retained[member.name] = payload

    return_manifest = strict_json(retained["RETURN_MANIFEST.json"])
    declared = return_manifest.get("members")
    if return_manifest.get("schema_version") != 1 or not isinstance(declared, dict):
        raise ValueError("unsupported return manifest")
    if set(observed) != set(declared) | {"RETURN_MANIFEST.json"}:
        raise ValueError("return archive member inventory differs from its manifest")
    for name, record in declared.items():
        if observed.get(name) != record:
            raise ValueError(f"return member identity mismatch: {name}")
    manifest_sha256 = hashlib.sha256(retained["RETURN_MANIFEST.json"]).hexdigest()
    if integrity.get("return_manifest_sha256") != manifest_sha256:
        raise ValueError("return manifest hash differs from integrity sidecar")

    complete = strict_json(retained["results/COMPLETE.json"])
    protocol = strict_json(retained["results/protocol.json"])
    trace = strict_json(retained["results/training_trace.json"])
    sessions = strict_json(retained["results/sessions.json"])
    checkpoint_manifest = strict_json(retained["results/checkpoint.manifest.json"])
    bundle_manifest = strict_json(retained["provenance/BUNDLE_MANIFEST.json"])
    source_manifest = strict_json(retained["provenance/upstream_source_manifest.json"])

    if not (
        complete.get("schema_version") == 1
        and complete.get("status") == "complete"
        and complete.get("scope") == "execution_gate_not_paper_scale"
        and complete.get("target_updates") == complete.get("completed_updates")
        and complete.get("completed_updates", 0) >= 100
        and complete.get("checkpoint_reload_exact") is True
        and complete.get("state_sha256") == complete.get("restored_state_sha256")
        and complete.get("dataset_hashes") == EXPECTED_DATASET_HASHES
        and complete.get("requested_bins") == 800
        and complete.get("actual_bins") == 499
        and complete.get("includes_twin_critic") is True
        and complete.get("includes_low_actor") is True
        and complete.get("includes_high_actor") is True
        and complete.get("jax_backend") == "gpu"
        and complete.get("resume_event_count", 0) >= 1
    ):
        raise ValueError("returned completion does not satisfy the full CUDA gate")

    if observed["results/protocol.json"]["sha256"] != complete.get("protocol_artifact_sha256"):
        raise ValueError("completion does not bind the protocol member")
    if observed["results/training_trace.json"]["sha256"] != complete.get("training_trace_sha256"):
        raise ValueError("completion does not bind the raw metric trace")
    if observed["results/sessions.json"]["sha256"] != complete.get("sessions_sha256"):
        raise ValueError("completion does not bind the session ledger")
    if observed["results/checkpoint.manifest.json"]["sha256"] != complete.get("checkpoint_manifest_sha256"):
        raise ValueError("completion does not bind the checkpoint manifest")

    protocol_sha256 = protocol.pop("sha256", None)
    if (
        protocol_sha256 != complete.get("protocol_sha256")
        or canonical_digest(protocol) != protocol_sha256
        or protocol.get("backend") != "gpu"
        or protocol.get("dataset_hashes") != EXPECTED_DATASET_HASHES
        or protocol.get("hsvl_revision") != EXPECTED_REVISION
        or protocol.get("upstream_source") != complete.get("upstream_source")
    ):
        raise ValueError("canonical CUDA protocol identity mismatch")
    execution_members = {
        "colab_runner_sha256": "provenance/run_full_agent_probe_colab.py",
        "cpu_reference_runner_sha256": "provenance/run_full_agent_probe.py",
        "provenance_auditor_sha256": "provenance/audit_full_agent_protocol.py",
        "return_packer_sha256": "provenance/package_colab_return.py",
    }
    for key, name in execution_members.items():
        if protocol.get("execution_files", {}).get(key) != observed[name]["sha256"]:
            raise ValueError(f"archived execution source differs from protocol: {key}")
    local_execution_files = {
        "colab_runner_sha256": PAPER_ROOT / "repro" / "colab" / "run_full_agent_probe_colab.py",
        "cpu_reference_runner_sha256": PAPER_ROOT / "repro" / "src" / "run_full_agent_probe.py",
        "provenance_auditor_sha256": PAPER_ROOT / "repro" / "src" / "audit_full_agent_protocol.py",
        "return_packer_sha256": PAPER_ROOT / "repro" / "colab" / "package_colab_return.py",
    }
    for key, path in local_execution_files.items():
        if protocol.get("execution_files", {}).get(key) != sha256_file(path):
            raise ValueError(f"returned execution source differs from local preregistration: {key}")

    local_source = verify_upstream_source()
    if (
        local_source != complete.get("upstream_source")
        or source_manifest.get("commit") != EXPECTED_REVISION
        or source_manifest != strict_json(
            (PAPER_ROOT / "repro" / "configs" / "upstream_source_manifest.json").read_bytes()
        )
    ):
        raise ValueError("returned primary-source identity differs from independent local audit")

    input_integrity = strict_json(input_integrity_path.read_bytes())
    input_bundle = protocol.get("input_bundle", {})
    if (
        sha256_file(input_archive_path) != input_integrity.get("archive_sha256")
        or input_archive_path.stat().st_size != input_integrity.get("archive_bytes")
        or input_integrity.get("archive_sha256") != input_bundle.get("archive_sha256")
        or input_integrity.get("archive_bytes") != input_bundle.get("archive_bytes")
        or input_integrity.get("bundle_manifest_sha256")
        != input_bundle.get("bundle_manifest_sha256")
        or hashlib.sha256(retained["provenance/BUNDLE_MANIFEST.json"]).hexdigest()
        != input_bundle.get("bundle_manifest_sha256")
        or bundle_manifest.get("official_source", {}).get("commit") != EXPECTED_REVISION
    ):
        raise ValueError("returned execution is not bound to the prepared input bundle")
    bundle_dataset_hashes = bundle_manifest.get("official_dataset_hashes", {})
    expected_bundle_datasets = {
        "data/ogbench/pointmaze-large-navigate-v0.npz": EXPECTED_DATASET_HASHES["train"],
        "data/ogbench/pointmaze-large-navigate-v0-val.npz": EXPECTED_DATASET_HASHES["validation"],
    }
    if bundle_dataset_hashes != expected_bundle_datasets:
        raise ValueError("input bundle dataset identities differ from preregistration")

    if not isinstance(trace, list) or len(trace) != complete.get("training_trace_rows"):
        raise ValueError("raw metric trace length mismatch")
    updates = []
    session_ids = set()
    for row in trace:
        update = int(row["update"])
        updates.append(update)
        session_ids.add(row.get("session_id"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"update {update} has no raw metrics")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"update {update} has nonfinite raw metrics")
        if not math.isfinite(float(row.get("updates_per_second", float("nan")))):
            raise ValueError(f"update {update} has invalid throughput")
    if updates != sorted(set(updates)) or updates[-1] != complete["completed_updates"]:
        raise ValueError("raw metric updates are not strictly increasing through completion")

    if not isinstance(sessions, list) or len(sessions) != complete.get("session_count"):
        raise ValueError("session ledger length mismatch")
    resume_count = 0
    allowed_statuses = {"complete", "interrupted_recovered"}
    for index, session in enumerate(sessions):
        if session.get("status") not in allowed_statuses:
            raise ValueError("session ledger contains an unreconciled invocation")
        gpu = session.get("gpu", {})
        devices = gpu.get("jax_devices")
        if (
            gpu.get("jax_backend") != "gpu"
            or not gpu.get("nvidia_smi_rows")
            or not isinstance(devices, list)
            or not devices
            or any(device.get("platform") != "gpu" for device in devices)
        ):
            raise ValueError("session lacks a valid GPU hardware ledger")
        if (
            session.get("status") == "complete"
            and session.get("session_id") not in session_ids
            and session.get("starting_completed_updates")
            != session.get("finished_completed_updates")
        ):
            raise ValueError("completed session has no raw metric row")
        resumed = session.get("resumed_from")
        if resumed is not None:
            resume_count += 1
            if (
                resumed.get("completed_updates") != session.get("starting_completed_updates")
                or index == 0
                or resumed.get("checkpoint_sha256")
                != sessions[index - 1].get("ending_checkpoint_sha256")
                or resumed.get("state_sha256")
                != sessions[index - 1].get("ending_state_sha256")
            ):
                raise ValueError("cross-invocation resume chain is not exact")
    if resume_count != complete.get("resume_event_count") or resume_count < 1:
        raise ValueError("required exact resume evidence is absent")
    if not sessions or sessions[-1].get("status") != "complete":
        raise ValueError("final session did not complete")

    active_name = checkpoint_manifest.get("active_checkpoint")
    active_member = f"results/{active_name}"
    if (
        not isinstance(active_name, str)
        or Path(active_name).name != active_name
        or active_member not in observed
        or checkpoint_manifest.get("protocol_sha256") != protocol_sha256
        or checkpoint_manifest.get("completed_updates") != complete.get("completed_updates")
        or checkpoint_manifest.get("checkpoint_sha256") != complete.get("checkpoint_sha256")
        or checkpoint_manifest.get("state_sha256") != complete.get("state_sha256")
        or observed[active_member]["sha256"] != complete.get("checkpoint_sha256")
        or observed[active_member]["bytes"] != complete.get("checkpoint_bytes")
    ):
        raise ValueError("returned active checkpoint chain is invalid")

    summary = return_manifest.get("result", {})
    for key in (
        "protocol_sha256",
        "completed_updates",
        "checkpoint_sha256",
        "state_sha256",
        "training_trace_sha256",
        "sessions_sha256",
        "resume_event_count",
    ):
        if summary.get(key) != complete.get(key):
            raise ValueError(f"return envelope summary mismatch: {key}")
    if summary.get("input_archive_sha256") != input_bundle.get("archive_sha256"):
        raise ValueError("return envelope input archive identity mismatch")

    return {
        "status": "verified",
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "input_archive_sha256": input_bundle["archive_sha256"],
        "protocol_sha256": protocol_sha256,
        "completed_updates": complete["completed_updates"],
        "parameter_count": complete["parameter_count"],
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "checkpoint_bytes": complete["checkpoint_bytes"],
        "training_trace_sha256": complete["training_trace_sha256"],
        "session_count": complete["session_count"],
        "resume_event_count": resume_count,
        "gpu_rows": [row["gpu"]["nvidia_smi_rows"] for row in sessions],
        "primary_source_commit": EXPECTED_REVISION,
        "official_datasets_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--integrity", type=Path)
    parser.add_argument("--input-archive", type=Path, default=EXPECTED_INPUT_ARCHIVE)
    parser.add_argument("--input-integrity", type=Path, default=EXPECTED_INPUT_INTEGRITY)
    args = parser.parse_args()
    integrity_path = args.integrity or args.archive.with_name(
        args.archive.name + ".integrity.json"
    )
    for path in (
        args.archive,
        integrity_path,
        args.input_archive,
        args.input_integrity,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = verify_archive(
        args.archive,
        integrity_path,
        args.input_archive,
        args.input_integrity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
