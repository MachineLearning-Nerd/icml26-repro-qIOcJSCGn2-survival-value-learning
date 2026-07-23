#!/usr/bin/env python3
"""Validate and deterministically package a qIO Colab execution-gate return."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "outputs" / "full_agent_probe_colab" / "seed-0"
DEFAULT_ARCHIVE = PAPER_ROOT / "outputs" / "colab_handoff" / "qio-full-agent-colab-return-v1.tar.gz"
EXPECTED_DATASET_HASHES = {
    "train": "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    "validation": "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
}
EXPECTED_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def safe_child(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"unsafe artifact basename: {name!r}")
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_output(output_dir: Path, require_resume: bool) -> dict[str, Path]:
    complete_path = output_dir / "COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text())
    required = (
        complete.get("schema_version") == 1
        and complete.get("status") == "complete"
        and complete.get("scope") == "execution_gate_not_paper_scale"
        and complete.get("target_updates") == complete.get("completed_updates")
        and complete.get("completed_updates", 0) > 0
        and complete.get("checkpoint_reload_exact") is True
        and complete.get("dataset_hashes") == EXPECTED_DATASET_HASHES
        and complete.get("requested_bins") == 800
        and complete.get("actual_bins") == 499
        and complete.get("includes_twin_critic") is True
        and complete.get("includes_low_actor") is True
        and complete.get("includes_high_actor") is True
        and complete.get("jax_backend") == "gpu"
        and complete.get("state_sha256") == complete.get("restored_state_sha256")
        and complete.get("upstream_source", {}).get("commit") == EXPECTED_REVISION
        and complete.get("upstream_source", {}).get("all_git_blobs_match") is True
    )
    if not required:
        raise ValueError("completion artifact does not satisfy the CUDA full-agent gate")
    if require_resume and complete.get("resume_event_count", 0) < 1:
        raise ValueError(
            "return packaging requires a validated cross-invocation resume; "
            "run the 10-update stage before extending to 100"
        )

    protocol_path = safe_child(output_dir, complete.get("protocol_artifact"))
    checkpoint_manifest_path = safe_child(
        output_dir, complete.get("checkpoint_manifest")
    )
    trace_path = safe_child(output_dir, complete.get("training_trace"))
    sessions_path = safe_child(output_dir, complete.get("sessions"))
    if sha256_file(protocol_path) != complete.get("protocol_artifact_sha256"):
        raise ValueError("protocol artifact hash mismatch")
    if sha256_file(checkpoint_manifest_path) != complete.get("checkpoint_manifest_sha256"):
        raise ValueError("checkpoint manifest hash mismatch")
    if sha256_file(trace_path) != complete.get("training_trace_sha256"):
        raise ValueError("raw metric trace hash mismatch")
    if sha256_file(sessions_path) != complete.get("sessions_sha256"):
        raise ValueError("session ledger hash mismatch")

    protocol = json.loads(protocol_path.read_text())
    protocol_digest = protocol.pop("sha256", None)
    if protocol_digest != complete.get("protocol_sha256") or canonical_digest(protocol) != protocol_digest:
        raise ValueError("protocol canonical digest mismatch")
    if protocol.get("backend") != "gpu" or protocol.get("dataset_hashes") != EXPECTED_DATASET_HASHES:
        raise ValueError("protocol backend or dataset identity mismatch")
    expected_execution_files = {
        "colab_runner_sha256": PAPER_ROOT / "repro" / "colab" / "run_full_agent_probe_colab.py",
        "cpu_reference_runner_sha256": PAPER_ROOT / "repro" / "src" / "run_full_agent_probe.py",
        "provenance_auditor_sha256": PAPER_ROOT / "repro" / "src" / "audit_full_agent_protocol.py",
        "return_packer_sha256": Path(__file__),
    }
    for key, path in expected_execution_files.items():
        if protocol.get("execution_files", {}).get(key) != sha256_file(path):
            raise ValueError(f"protocol execution-file identity mismatch: {key}")

    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    checkpoint_path = safe_child(output_dir, checkpoint_manifest.get("active_checkpoint"))
    if (
        checkpoint_manifest.get("protocol_sha256") != protocol_digest
        or checkpoint_manifest.get("completed_updates") != complete.get("completed_updates")
        or checkpoint_manifest.get("checkpoint_sha256") != complete.get("checkpoint_sha256")
        or checkpoint_manifest.get("state_sha256") != complete.get("state_sha256")
        or sha256_file(checkpoint_path) != complete.get("checkpoint_sha256")
    ):
        raise ValueError("active checkpoint chain mismatch")

    trace = json.loads(trace_path.read_text())
    sessions = json.loads(sessions_path.read_text())
    if not isinstance(trace, list) or len(trace) != complete.get("training_trace_rows"):
        raise ValueError("raw metric trace length mismatch")
    updates = [int(row["update"]) for row in trace]
    if not trace or updates != sorted(set(updates)) or updates[-1] != complete["completed_updates"]:
        raise ValueError("raw metric trace update ordering mismatch")
    if not isinstance(sessions, list) or len(sessions) != complete.get("session_count"):
        raise ValueError("session ledger length mismatch")
    allowed_statuses = {"complete", "interrupted_recovered"}
    if any(row.get("status") not in allowed_statuses for row in sessions):
        raise ValueError("session ledger contains an unreconciled invocation")
    if not sessions or sessions[-1].get("status") != "complete":
        raise ValueError("final session did not complete")
    resumed = [row for row in sessions if row.get("resumed_from") is not None]
    if len(resumed) != complete.get("resume_event_count"):
        raise ValueError("resume evidence count mismatch")

    return {
        "COMPLETE.json": complete_path,
        "PROGRESS.json": output_dir / "PROGRESS.json",
        "protocol.json": protocol_path,
        "training_trace.json": trace_path,
        "sessions.json": sessions_path,
        "checkpoint.manifest.json": checkpoint_manifest_path,
        checkpoint_path.name: checkpoint_path,
    }


def tar_info(name: str, size: int, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def write_deterministic_archive(
    archive_path: Path, selected: dict[str, Path], summary: dict[str, Any]
) -> tuple[str, str]:
    records = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(selected.items())
    }
    manifest = {
        "schema_version": 1,
        "purpose": "qio_full_agent_colab_cuda_return",
        "result": summary,
        "members": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                archive.addfile(
                    tar_info("RETURN_MANIFEST.json", len(manifest_bytes)),
                    io.BytesIO(manifest_bytes),
                )
                for name, path in sorted(selected.items()):
                    with path.open("rb") as handle:
                        archive.addfile(
                            tar_info(name, path.stat().st_size, path.suffix == ".py"),
                            handle,
                        )
    temporary.replace(archive_path)
    return sha256_file(archive_path), hashlib.sha256(manifest_bytes).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--require-resume-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    result_files = validate_output(args.output_dir, args.require_resume_evidence)
    complete = json.loads(result_files["COMPLETE.json"].read_text())
    provenance = {
        "provenance/BUNDLE_MANIFEST.json": PAPER_ROOT / "BUNDLE_MANIFEST.json",
        "provenance/upstream_source_manifest.json": PAPER_ROOT / "repro" / "configs" / "upstream_source_manifest.json",
        "provenance/hsvl.yaml": PAPER_ROOT / "upstream" / "hsvl" / "config" / "agent" / "hsvl.yaml",
        "provenance/upstream-pyproject.toml": PAPER_ROOT / "upstream" / "pyproject.toml",
        "provenance/upstream-uv.lock": PAPER_ROOT / "upstream" / "uv.lock",
        "provenance/run_full_agent_probe_colab.py": PAPER_ROOT / "repro" / "colab" / "run_full_agent_probe_colab.py",
        "provenance/run_full_agent_probe.py": PAPER_ROOT / "repro" / "src" / "run_full_agent_probe.py",
        "provenance/audit_full_agent_protocol.py": PAPER_ROOT / "repro" / "src" / "audit_full_agent_protocol.py",
        "provenance/package_colab_return.py": Path(__file__),
    }
    selected = {f"results/{name}": path for name, path in result_files.items()}
    selected.update(provenance)
    missing = [str(path) for path in selected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"return provenance file(s) missing: {missing}")
    summary = {
        "protocol_sha256": complete["protocol_sha256"],
        "completed_updates": complete["completed_updates"],
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "state_sha256": complete["state_sha256"],
        "training_trace_sha256": complete["training_trace_sha256"],
        "sessions_sha256": complete["sessions_sha256"],
        "resume_event_count": complete["resume_event_count"],
        "input_archive_sha256": complete["input_bundle"]["archive_sha256"],
    }
    archive_sha256, manifest_sha256 = write_deterministic_archive(
        args.archive, selected, summary
    )
    integrity = {
        "schema_version": 1,
        "archive": args.archive.name,
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "return_manifest_sha256": manifest_sha256,
    }
    integrity_path = args.archive.with_name(args.archive.name + ".integrity.json")
    temporary = integrity_path.with_suffix(integrity_path.suffix + ".tmp")
    temporary.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    temporary.replace(integrity_path)
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
