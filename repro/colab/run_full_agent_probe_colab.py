#!/usr/bin/env python3
"""Fail-closed CUDA execution gate for the released full HSVL agent.

This is the accelerator counterpart of ``repro/src/run_full_agent_probe.py``.
It preserves the released width-512 twin critic, low-level actor, high-level
actor, batch size, temporal basis library, and official PointMaze data.  The
target update count is intentionally excluded from the protocol identity so a
validated checkpoint can be extended after a Colab interruption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tarfile
import time
import uuid
from typing import Any


# These values must be set before the reference runner imports JAX.  Defining
# XLA_FLAGS (even as an empty string) prevents the CPU reference runner from
# installing its CPU-only XLA policy during import.
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "")

PAPER_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PAPER_ROOT / "repro" / "src"
sys.path.insert(0, str(SRC_ROOT))

import run_full_agent_probe as core  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = PAPER_ROOT / "outputs" / "full_agent_probe_colab" / "seed-0"
INPUT_MANIFEST_NAME = "BUNDLE_MANIFEST.json"
EXPECTED_DATASET_HASHES = core.EXPECTED_DATASET_HASHES
LEDGER_ENV_KEYS = (
    "COLAB_RELEASE_TAG",
    "COLAB_GPU",
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORM_NAME",
    "JAX_ENABLE_X64",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_archive_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def verify_input_archive(archive_path: Path, integrity_path: Path) -> dict[str, Any]:
    """Verify the deterministic upload archive before trusting extracted files."""
    if not archive_path.is_file() or not integrity_path.is_file():
        raise FileNotFoundError("input archive and integrity sidecar are both required")
    archive_sha256 = core.sha256_file(archive_path)
    integrity = json.loads(integrity_path.read_text())
    if integrity.get("archive_sha256") != archive_sha256:
        raise ValueError("input archive SHA-256 does not match its integrity sidecar")
    if integrity.get("archive_bytes") != archive_path.stat().st_size:
        raise ValueError("input archive byte count does not match its integrity sidecar")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or not all(safe_archive_name(name) for name in names):
            raise ValueError("input archive has duplicate or unsafe member names")
        manifest_member = archive.getmember(INPUT_MANIFEST_NAME)
        extracted = archive.extractfile(manifest_member)
        if extracted is None:
            raise ValueError("input bundle manifest is not a regular file")
        manifest_bytes = extracted.read()
        manifest = json.loads(manifest_bytes)
        declared = manifest.get("members")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("input bundle manifest has no member inventory")
        if set(names) != set(declared) | {INPUT_MANIFEST_NAME}:
            raise ValueError("input archive member inventory does not match its manifest")
        for member in members:
            if member.name == INPUT_MANIFEST_NAME:
                continue
            if not member.isfile():
                raise ValueError(f"input archive member is not a file: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read input archive member: {member.name}")
            payload = handle.read()
            record = declared[member.name]
            if record.get("bytes") != len(payload) or record.get("sha256") != sha256_bytes(payload):
                raise ValueError(f"input archive member failed identity check: {member.name}")

    # Rehash all extracted files that can affect this execution.  Data and the
    # source tree are covered too, not merely the wrapper scripts.
    for relative, record in declared.items():
        extracted_path = (PAPER_ROOT / relative).resolve()
        try:
            extracted_path.relative_to(PAPER_ROOT)
        except ValueError as exc:
            raise ValueError(f"unsafe extracted member path: {relative}") from exc
        if not extracted_path.is_file():
            raise FileNotFoundError(extracted_path)
        if (
            extracted_path.stat().st_size != record.get("bytes")
            or core.sha256_file(extracted_path) != record.get("sha256")
        ):
            raise ValueError(f"extracted input differs from archive: {relative}")

    manifest_sha256 = sha256_bytes(manifest_bytes)
    if integrity.get("bundle_manifest_sha256") != manifest_sha256:
        raise ValueError("input bundle manifest SHA-256 does not match sidecar")
    return {
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "bundle_manifest_sha256": manifest_sha256,
        "member_count": len(declared),
        "schema_version": manifest.get("schema_version"),
    }


def gpu_ledger() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, text=True, capture_output=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    devices = []
    for device in core.jax.devices():
        devices.append(
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(getattr(device, "device_kind", "")),
                "process_index": int(device.process_index),
                "string": str(device),
            }
        )
    if core.jax.default_backend() != "gpu" or not devices:
        raise RuntimeError(
            f"CUDA handoff requires a JAX GPU backend, got {core.jax.default_backend()}"
        )
    return {
        "captured_at": utc_now(),
        "nvidia_smi_command": query,
        "nvidia_smi_rows": rows,
        "jax_backend": core.jax.default_backend(),
        "jax_devices": devices,
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": core.package_versions(),
        "environment": {key: os.environ.get(key) for key in LEDGER_ENV_KEYS},
    }


def protocol_payload(
    config: dict[str, Any],
    dataset_hashes: dict[str, str],
    seed: int,
    upstream_source: dict[str, Any],
    input_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "released_full_hsvl_real_data_colab_cuda_execution_gate",
        "dataset_name": core.DATASET_NAME,
        "dataset_hashes": dataset_hashes,
        "hsvl_revision": core.HSVL_REVISION,
        "ogbench_revision": core.OGBENCH_REVISION,
        "upstream_source": upstream_source,
        "input_bundle": input_bundle,
        "execution_files": {
            "colab_runner_sha256": core.sha256_file(Path(__file__)),
            "cpu_reference_runner_sha256": core.sha256_file(
                SRC_ROOT / "run_full_agent_probe.py"
            ),
            "provenance_auditor_sha256": core.sha256_file(
                SRC_ROOT / "audit_full_agent_protocol.py"
            ),
            "return_packer_sha256": core.sha256_file(
                PAPER_ROOT / "repro" / "colab" / "package_colab_return.py"
            ),
        },
        "agent_config": config,
        "seed": seed,
        "backend": "gpu",
        "precision_policy": {
            "jax_enable_x64": bool(core.jax.config.jax_enable_x64),
            "default_matmul_precision": str(
                core.jax.config.jax_default_matmul_precision
            ),
        },
        "package_versions": core.package_versions(),
    }


def load_trace(path: Path, completed: int) -> list[dict[str, Any]]:
    if not path.is_file() or completed == 0:
        return []
    trace = json.loads(path.read_text())
    if not isinstance(trace, list):
        raise ValueError("training trace must be a list")
    retained = [row for row in trace if int(row["update"]) <= completed]
    updates = [int(row["update"]) for row in retained]
    if updates != sorted(set(updates)):
        raise ValueError("training trace update sequence is not strictly increasing")
    return retained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-integrity", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--data-dir", type=Path, default=PAPER_ROOT / "data" / "ogbench")
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "upstream" / "hsvl" / "config" / "agent" / "hsvl.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.updates < 1 or args.log_interval < 1 or args.checkpoint_interval < 1:
        parser.error("updates and intervals must be positive")
    return args


def main() -> None:
    args = parse_args()
    if core.jax.default_backend() != "gpu":
        raise RuntimeError(
            f"this preregistered Colab probe requires GPU, got {core.jax.default_backend()}"
        )
    input_bundle = verify_input_archive(args.input_archive, args.input_integrity)
    gpu = gpu_ledger()

    train_path = args.data_dir / f"{core.DATASET_NAME}.npz"
    validation_path = args.data_dir / f"{core.DATASET_NAME}-val.npz"
    for path in (train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_hashes = {
        "train": core.sha256_file(train_path),
        "validation": core.sha256_file(validation_path),
    }
    if dataset_hashes != EXPECTED_DATASET_HASHES:
        raise ValueError(
            f"official OGBench dataset hash mismatch: expected {EXPECTED_DATASET_HASHES}, "
            f"observed {dataset_hashes}"
        )
    config = core.load_released_config(args.config)
    agent_config = core.FrozenDict(config)
    upstream_source = core.verify_upstream_source()
    protocol = protocol_payload(config, dataset_hashes, args.seed, upstream_source, input_bundle)
    protocol_sha256 = core.canonical_digest(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    new_protocol_artifact = {**protocol, "sha256": protocol_sha256}
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != new_protocol_artifact:
        raise ValueError("existing output protocol differs; refusing checkpoint reuse")
    atomic_json(protocol_path, new_protocol_artifact)

    print("Loading official PointMaze train split...", flush=True)
    train_host, _ = core.ogbench.make_env_and_datasets(
        core.DATASET_NAME,
        dataset_path=str(train_path),
        compact_dataset=True,
        dataset_only=True,
    )
    train_dataset = core.prepare_hgc_dataset_for_jax(train_host, do_terminals=True)
    train_size = core.dataset_size_jax(train_dataset)
    batch_size = int(config["batch_size"])
    rng = core.jax.random.PRNGKey(args.seed)
    example, _ = core.HGCDataset_sample(
        rng, train_dataset, batch_size, train_size, agent_config
    )
    template = core.HSVL.create(
        seed=args.seed,
        ex_observations=example["observations"],
        ex_actions=example["actions"],
        ex_goals=example["oracle_reps"] if "oracle_reps" in train_dataset else None,
        config=agent_config,
    )
    parameter_count = core.count_parameters(template.network.params)
    module_parameter_counts = {
        key: core.count_parameters(value) for key, value in template.network.params.items()
    }
    checkpoint_path = args.output_dir / "checkpoint.pkl"
    checkpoint_manifest_path = checkpoint_path.with_suffix(".manifest.json")
    completed = 0
    agent = template
    resumed_from: dict[str, Any] | None = None
    if args.resume and checkpoint_manifest_path.is_file():
        agent, manifest = core.restore_checkpoint(template, protocol_sha256, checkpoint_path)
        completed = int(manifest["completed_updates"])
        resumed_from = {
            "completed_updates": completed,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "state_sha256": manifest["state_sha256"],
            "active_checkpoint": manifest["active_checkpoint"],
            "checkpoint_manifest_sha256": core.sha256_file(checkpoint_manifest_path),
        }
        if completed > args.updates:
            raise ValueError(
                f"checkpoint has {completed} updates, beyond requested target {args.updates}"
            )
        print(f"Resumed validated checkpoint at update {completed}", flush=True)
    elif checkpoint_path.exists() or checkpoint_manifest_path.exists():
        raise FileExistsError("checkpoint exists but --no-resume was requested")

    session_id = str(uuid.uuid4())
    sessions_path = args.output_dir / "sessions.json"
    sessions: list[dict[str, Any]] = []
    if sessions_path.is_file():
        sessions = json.loads(sessions_path.read_text())
        if not isinstance(sessions, list):
            raise ValueError("session ledger must be a list")
    running_sessions = [row for row in sessions if row.get("status") == "running"]
    if running_sessions:
        if running_sessions != [sessions[-1]]:
            raise ValueError("only the final ledger session may be interrupted")
        # A process cannot update its ledger after a hard Colab disconnect.  The
        # next invocation closes that record against the checkpoint it just
        # validated.  No progress beyond this durable generation is credited.
        sessions[-1].update(
            {
                "status": "interrupted_recovered",
                "recovered_at": utc_now(),
                "finished_completed_updates": completed,
                "ending_checkpoint_sha256": (
                    resumed_from["checkpoint_sha256"] if resumed_from else None
                ),
                "ending_state_sha256": (
                    resumed_from["state_sha256"] if resumed_from else None
                ),
                "recovery_note": (
                    "Previous process ended without a terminal ledger write; "
                    "this invocation credited only the independently validated "
                    "durable checkpoint."
                ),
            }
        )
        atomic_json(sessions_path, sessions)
    session = {
        "session_id": session_id,
        "started_at": utc_now(),
        "argv": sys.argv,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "target_updates": args.updates,
        "starting_completed_updates": completed,
        "resumed_from": resumed_from,
        "gpu": gpu,
        "status": "running",
    }
    sessions.append(session)
    atomic_json(sessions_path, sessions)

    trace_path = args.output_dir / "training_trace.json"
    trace = load_trace(trace_path, completed)
    start = time.perf_counter()
    interval_start = start
    interval_step = completed
    print(
        f"Training released full CUDA agent: {completed}->{args.updates} updates, "
        f"parameters={parameter_count:,}, batch={batch_size}",
        flush=True,
    )
    for update in range(completed + 1, args.updates + 1):
        agent, info = agent.sample_and_update(train_dataset, batch_size, train_size)
        if update % args.log_interval == 0 or update == args.updates:
            core.jax.block_until_ready(agent)
            now = time.perf_counter()
            metrics = core.finite_metrics(info)
            row = {
                "update": update,
                "session_id": session_id,
                "interval_seconds": now - interval_start,
                "updates_per_second": (update - interval_step) / (now - interval_start),
                "elapsed_seconds_this_session": now - start,
                "metrics": metrics,
            }
            if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("empty or nonfinite raw metric record")
            trace.append(row)
            atomic_json(trace_path, trace)
            atomic_json(
                args.output_dir / "PROGRESS.json",
                {
                    "status": "running",
                    "protocol_sha256": protocol_sha256,
                    "target_updates": args.updates,
                    "completed_updates": update,
                    "parameter_count": parameter_count,
                    "session_id": session_id,
                    "last_trace": row,
                },
            )
            print(json.dumps(row, sort_keys=True), flush=True)
            interval_start = now
            interval_step = update
        if update % args.checkpoint_interval == 0 or update == args.updates:
            core.save_checkpoint(agent, update, protocol_sha256, checkpoint_path)

    restored, manifest = core.restore_checkpoint(template, protocol_sha256, checkpoint_path)
    restored_state_sha256 = core.state_digest(
        core.flax.serialization.to_state_dict(restored)
    )
    completed_seconds = time.perf_counter() - start
    session.update(
        {
            "finished_at": utc_now(),
            "status": "complete",
            "finished_completed_updates": int(manifest["completed_updates"]),
            "ending_checkpoint_sha256": manifest["checkpoint_sha256"],
            "ending_state_sha256": manifest["state_sha256"],
            "runtime_seconds": completed_seconds,
        }
    )
    atomic_json(sessions_path, sessions)
    trace_sha256 = core.sha256_file(trace_path)
    sessions_sha256 = core.sha256_file(sessions_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "execution_gate_not_paper_scale",
        "protocol_sha256": protocol_sha256,
        "protocol_artifact": "protocol.json",
        "protocol_artifact_sha256": core.sha256_file(protocol_path),
        "upstream_source": upstream_source,
        "input_bundle": input_bundle,
        "target_updates": args.updates,
        "completed_updates": int(manifest["completed_updates"]),
        "seed": args.seed,
        "dataset_rows": int(len(train_host["observations"])),
        "dataset_valid_pool": train_size,
        "dataset_hashes": dataset_hashes,
        "parameter_count": parameter_count,
        "module_parameter_counts": module_parameter_counts,
        "requested_bins": int(config["num_log_bins"]),
        "actual_bins": int(len(core.np.asarray(restored._bins)) - 1),
        "includes_twin_critic": bool(config["ensemble"]),
        "includes_low_actor": "low_actor" in module_parameter_counts,
        "includes_high_actor": "high_actor" in module_parameter_counts,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_manifest": "checkpoint.manifest.json",
        "checkpoint_manifest_sha256": core.sha256_file(checkpoint_manifest_path),
        "checkpoint_bytes": manifest["checkpoint_bytes"],
        "state_sha256": manifest["state_sha256"],
        "restored_state_sha256": restored_state_sha256,
        "checkpoint_reload_exact": restored_state_sha256 == manifest["state_sha256"],
        "training_trace": "training_trace.json",
        "training_trace_sha256": trace_sha256,
        "training_trace_rows": len(trace),
        "sessions": "sessions.json",
        "sessions_sha256": sessions_sha256,
        "session_count": len(sessions),
        "resume_event_count": sum(row.get("resumed_from") is not None for row in sessions),
        "runtime_seconds_this_invocation": completed_seconds,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "jax_backend": core.jax.default_backend(),
        "jax_devices": [str(device) for device in core.jax.devices()],
        "package_versions": core.package_versions(),
        "paper_scale_target_updates": 1_000_000,
        "deviations": [
            f"Execution gate runs {args.updates} updates rather than 1,000,000.",
            "PointMaze-large is official OGBench navigation data but is not a Table 1 headline task.",
            "No policy-success claim is made by this training-and-reload gate.",
        ],
    }
    if not result["checkpoint_reload_exact"]:
        raise AssertionError("checkpoint reload was not exact")
    atomic_json(args.output_dir / "COMPLETE.json", result)
    atomic_json(args.output_dir / "PROGRESS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
