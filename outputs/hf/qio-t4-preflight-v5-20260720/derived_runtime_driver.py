#!/usr/bin/env python3
"""Fail-closed HF Jobs T4 preflight for the frozen qIO CUDA bundle.

The driver is deliberately standard-library-only.  It validates and extracts the
frozen Colab input, installs the authors' exact ``uv.lock`` environment, executes
the preregistered 10-update checkpoint followed by an independent resume to 100
updates, packages the return, and writes raw logs plus runtime, GPU-memory,
throughput, cost, and integrity evidence to a mounted HF bucket.

Attempt 5 retains Attempt 4's exact source, lockfile, bounded dependency
transport, external UV environment/cache isolation, and derived-runtime tuple
shim.  It adds one fail-closed result-inventory normalization: the registered
runner checks unprefixed actor names even though Flax emits ``modules_*`` keys.
The normalizer requires exact actor parameter counts, actor-loss trace metrics,
and checkpoint reload identity before correcting those two false-negative
booleans.  The official package gate remains unchanged.  The four earlier
attempts remain immutable and are referenced rather than overwritten.

This is an execution/checkpoint/throughput gate, not paper-scale Claim 3 evidence.
It must be submitted on ``t4-small`` with a Jobs timeout below one hour.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from typing import Any, TextIO


SCHEMA_VERSION = 1
ATTEMPT_VERSION = 5
INPUT_SERIES = "qio-t4-preflight-v1"
OUTPUT_SERIES = "qio-t4-preflight-v5"
DERIVED_RUNNER_FLAG = "--_derived-runner"
APPROVED_SEQUENCE_FIELDS = ("actor_hidden_dims", "value_hidden_dims")
EXPECTED_MODULE_PARAMETER_COUNTS = {
    "modules_goal_rep": 662_272,
    "modules_high_actor": 1_712_896,
    "modules_low_actor": 1_712_642,
    "modules_value": 5_690_888,
}
EXPECTED_PARAMETER_COUNT = 9_778_698
DEPENDENCY_TRANSPORT_SETTINGS = {
    "UV_HTTP_TIMEOUT": "300",
    "UV_HTTP_RETRIES": "5",
    "UV_CONCURRENT_DOWNLOADS": "4",
}
PINNED_CUDNN_PACKAGE = {
    "name": "nvidia-cudnn-cu11",
    "version": "8.9.7.29",
    "source": {"registry": "https://pypi.nvidia.com/"},
    "wheels": [
        {
            "url": (
                "https://pypi.nvidia.com/nvidia-cudnn-cu11/"
                "nvidia_cudnn_cu11-8.9.7.29-py3-none-manylinux1_x86_64.whl"
            ),
            "hash": (
                "sha256:ff7ec4a73b762ace4dc50850d16fe76d9519a7c2d56ac293955d611e900eba00"
            ),
        }
    ],
}
EXPECTED_COLAB_RUNNER_SHA256 = (
    "bed6a8f8bdf44a4c716beb1e0dca8d194faaaccfb10581f9950e4d038e4f2b32"
)
PRIOR_ATTEMPTS = (
    {
        "job_id": "DineshAI/6a5dbfefd216bd6f3a202e00",
        "run_id": "qio-t4-preflight-v1-20260720",
        "bucket_path": (
            "hf://buckets/DineshAI/qIOcJSCGn2-artifacts/hf-jobs/"
            "qio-t4-preflight-v1/runs/qio-t4-preflight-v1-20260720"
        ),
        "status": "failed_closed",
        "failure": (
            "uv sync completed, then the source verifier rejected upstream/.venv "
            "as an unexpected mutation before any scientific update executed"
        ),
        "driver_sha256": (
            "993c47d674eb7081dbc99cfe2990486664dc99570c9011c1861994cf52558523"
        ),
        "uv_sync_log_sha256": (
            "001ffc4c206a332f3bf2516c06086527cf9a00a95aad9ee58733f692f8183750"
        ),
        "phase_10_failure_log_sha256": (
            "f0c8e5fae53ae73f59bd2acc764db0e476567b19a0d587e29c610ba209c58710"
        ),
    },
    {
        "job_id": "DineshAI/6a5dc3acd216bd6f3a202e64",
        "run_id": "qio-t4-preflight-v2-20260720",
        "bucket_path": (
            "hf://buckets/DineshAI/qIOcJSCGn2-artifacts/hf-jobs/"
            "qio-t4-preflight-v2/runs/qio-t4-preflight-v2-20260720"
        ),
        "status": "failed_closed",
        "failure": (
            "external UV environment and pristine-source gates passed; the first "
            "HGCDataset_sample call then rejected list-valued hidden dimensions in "
            "the static JAX FrozenDict before any scientific update executed"
        ),
        "driver_sha256": (
            "f55fbcdeb8064e9ce15929d7b942bfa6d4db7d5d858f9176c741a87ed34349bc"
        ),
        "preflight_report_sha256": (
            "1764457e800eae339ccc34c821bde4bb70ff02f396756873a7e130e99e1bd65e"
        ),
        "preflight_integrity_sha256": (
            "0708fc59d8eb43ed8d981cb499aeb399b18ac76ad3636cf96913877306d2cbe5"
        ),
        "events_sha256": (
            "7da79aa3e60119aa34fd603110719bc402ede85089d53cf9370566405cbcc3a5"
        ),
        "resource_samples_sha256": (
            "ade646f7b5938eb32a82443acb418500b96ba16ed3cc9c753d1f6b687df01418"
        ),
        "uv_sync_log_sha256": (
            "58e68ae3bd7ff371af7792b83539e55c9631cfdc90b8856fe4b53ae8dddb5098"
        ),
        "phase_10_failure_log_sha256": (
            "1889c5e5b25a2ace8f7cb51963b245640642b07b680b1a96ddcb6e155b03158f"
        ),
        "unpatched_protocol_sha256": (
            "32d1c2f063ca96a04cf2e4a163f65dba0aa041dc6da109344dd0ae7eddbfc9f1"
        ),
    },
    {
        "job_id": "DineshAI/6a5dc7c2bee6ee1cf4ed20ff",
        "run_id": "qio-t4-preflight-v3-20260720",
        "bucket_path": (
            "hf://buckets/DineshAI/qIOcJSCGn2-artifacts/hf-jobs/"
            "qio-t4-preflight-v3/runs/qio-t4-preflight-v3-20260720"
        ),
        "status": "failed_closed",
        "failure": (
            "the frozen UV sync downloaded and extracted most dependencies, then "
            "the pinned 667.5 MiB nvidia-cudnn-cu11 wheel exceeded UV's 30-second "
            "HTTP read timeout before any scientific update executed"
        ),
        "driver_sha256": (
            "013b02f6d6d6964fb2b8216852210a0ef832d59cf1ad917af9ac8153d2701288"
        ),
        "preflight_report_sha256": (
            "879c0bde049e449fc508592e205497173f3bef55e3fa237163d43fdcfa9ed8f7"
        ),
        "preflight_integrity_sha256": (
            "ac70cd0accb4c4e5bc88108e5cfb94f2fc1e4c1e0385f79de1da6a06de1f0ea2"
        ),
        "events_sha256": (
            "8752565ab8d18b5c9a2b92e94b4d1e9a67ae94dd64c0691877c22ea1db87d018"
        ),
        "resource_samples_sha256": (
            "e96ac1e49c88e408703b2c203c29ce5d0590e569284c0cba63c7e872f35f3792"
        ),
        "uv_sync_log_sha256": (
            "c6cd4ae7bb0de45bd87ab5d5684ee353043dfe7999e6c1c4ab48c74fe063df88"
        ),
        "uv_sync_log_bytes": 3452,
        "uv_sync_runtime_seconds": 50.45219963499994,
        "job_running_seconds": 61,
        "job_total_seconds": 70,
        "scientific_updates_executed": 0,
    },
    {
        "job_id": "DineshAI/6a5dca82d216bd6f3a202f21",
        "run_id": "qio-t4-preflight-v4-20260720",
        "bucket_path": (
            "hf://buckets/DineshAI/qIOcJSCGn2-artifacts/hf-jobs/"
            "qio-t4-preflight-v4/runs/qio-t4-preflight-v4-20260720"
        ),
        "status": "failed_closed_after_scientific_gate",
        "failure": (
            "the frozen environment, 0-to-10 CUDA stage, exact 10-to-100 resume, "
            "checkpoint reload, and pristine-source gates passed; the unchanged "
            "official packager then rejected false actor-presence booleans produced "
            "by unprefixed lookups against Flax modules_* parameter keys"
        ),
        "driver_sha256": (
            "2d8eef3ea343cc25c420328533b5ee8228fbcaa25a109e0dc60120872a58f9bf"
        ),
        "preflight_report_sha256": (
            "3d42a8d31252f8e301872d53b4cfa19e9481210eb31062795399294bdf845197"
        ),
        "preflight_integrity_sha256": (
            "bdb99435d162614efc111ff688b214f7d615bd8a11e84a76767feb6efc8a1c67"
        ),
        "package_failure_log_sha256": (
            "b9f9cb850521ba0612b651170b27f5b44ab89a9da9b6a27769be0757cb4a00ed"
        ),
        "complete_json_sha256": (
            "7ec127d7a420158e06a01ec14334e026493aab4275b2b26db9fc76ebf772a459"
        ),
        "protocol_artifact_sha256": (
            "a62d89b3f75956b90b6d3e69da2754bc8f3ddc04b92c56d797934ed5dbc16e07"
        ),
        "training_trace_sha256": (
            "be6595c2df95f4dba0d1f66f0b5c97f227c1ee38de1cd6d177c201d75bb0769f"
        ),
        "sessions_sha256": (
            "045a131355c03b6e7782a7928a3eb517af6eac23cff9b474afe393a662aaf322"
        ),
        "checkpoint_sha256": (
            "5cdd4706dcf9e4dd38afdfcfe0a09d02af18dd9cbe43e45f6eeda7e7a78791eb"
        ),
        "state_sha256": (
            "38271517ea571a90bc8344e18a9a03017366b46b035f9b6ca5a9537a6af0f056"
        ),
        "uv_sync_log_sha256": (
            "b4bf77ddc4d1498076da445db19d5d916ecb714ca4d7a95122c4fd6756281143"
        ),
        "phase_10_log_sha256": (
            "d126119f3bfaeb6ea94c5485d0c0e8671ed3aed724bae12b6ea4927ff15a948e"
        ),
        "phase_100_log_sha256": (
            "babe97d69fdb0b1a8fabaff4d7270b64cdabf135cd9f03a9773f4842a3742781"
        ),
        "resource_samples_sha256": (
            "79c0b9035add9e74c0bff62956d695cde161f9897b44a3da9cb845d376773e96"
        ),
        "completed_updates": 100,
        "resume_event_count": 1,
        "checkpoint_reload_exact": True,
        "module_parameter_counts": EXPECTED_MODULE_PARAMETER_COUNTS,
        "reported_actor_presence": {"low_actor": False, "high_actor": False},
        "job_running_seconds": 150,
        "job_total_seconds": 158,
    },
)
EXPECTED_ARCHIVE_NAME = "qio-full-agent-colab-input-v1.tar.gz"
EXPECTED_ARCHIVE_BYTES = 22_528_563
EXPECTED_ARCHIVE_SHA256 = (
    "d7f8e07b72f828d8f39100533510a60858dab8b7cc7debcc2e2409260da634ee"
)
EXPECTED_SIDECAR_SHA256 = (
    "aeb4c83a3d369dff6c9956e9e6731736b7980b33dd12c8a0ecc70ceec196e9fd"
)
EXPECTED_BUNDLE_MANIFEST_SHA256 = (
    "d2e5706fe5c8f12d904234fd6b23b624624501da05e60954b291372c3e67fa42"
)
EXPECTED_MEMBER_COUNT = 26
EXPECTED_SOURCE_COMMIT = "5f13cf22d397be42a87b7d35336db7662879d6db"
EXPECTED_DATASET_HASHES = {
    "data/ogbench/pointmaze-large-navigate-v0.npz": (
        "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61"
    ),
    "data/ogbench/pointmaze-large-navigate-v0-val.npz": (
        "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4"
    ),
}
EXPECTED_BUNDLE_EXECUTION_HASHES = {
    "repro/colab/run_full_agent_probe_colab.py": (
        "bed6a8f8bdf44a4c716beb1e0dca8d194faaaccfb10581f9950e4d038e4f2b32"
    ),
    "repro/colab/package_colab_return.py": (
        "124599a99e116f1001e4adefb9455bc06518b255b4768fc1c6439a414b4967e9"
    ),
    "repro/src/run_full_agent_probe.py": (
        "3bcc449d317cb00030ebcf2394370a84a9fdbb487cd75ebcc87b451dc4350237"
    ),
    "repro/src/audit_full_agent_protocol.py": (
        "cea07c8a68685d949e8149e1a6366d0bf3a98a346f6e7035e7df6b7236702c1a"
    ),
    "repro/configs/upstream_source_manifest.json": (
        "e9a279e09fa5952002b8f37a2aa54c7e02a711fcff3f86c362101d6e406131a6"
    ),
    "upstream/uv.lock": (
        "8b7c535e583ed86ef3966f9e5b52b07900a86d864fc5f3dc8c99b8fed235d069"
    ),
    "upstream/pyproject.toml": (
        "d9a041488d6b04a232d8599b5744d0f055e554afca56132358e6b6898737ac49"
    ),
}
T4_RATE_USD_PER_HOUR = 0.40
PREFLIGHT_TIMEOUT_HOURS = 59.0 / 60.0
PREFLIGHT_TIMEOUT_COST_USD = T4_RATE_USD_PER_HOUR * PREFLIGHT_TIMEOUT_HOURS
ALLOWED_T4_NAMES = {"Tesla T4", "NVIDIA T4"}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_archive_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def validate_and_extract_bundle(
    archive_path: Path, sidecar_path: Path, extraction_root: Path
) -> dict[str, Any]:
    """Stream-verify every declared byte before extracting regular files."""
    if not archive_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("the frozen input archive and sidecar are required")
    if archive_path.name != EXPECTED_ARCHIVE_NAME:
        raise ValueError(f"unexpected input archive name: {archive_path.name}")
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("frozen input archive byte count mismatch")
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("frozen input archive SHA-256 mismatch")
    if sha256_file(sidecar_path) != EXPECTED_SIDECAR_SHA256:
        raise ValueError("frozen input sidecar SHA-256 mismatch")
    sidecar = json.loads(sidecar_path.read_text())
    if not (
        sidecar.get("schema_version") == 1
        and sidecar.get("archive") == EXPECTED_ARCHIVE_NAME
        and sidecar.get("archive_bytes") == EXPECTED_ARCHIVE_BYTES
        and sidecar.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and sidecar.get("bundle_manifest_sha256") == EXPECTED_BUNDLE_MANIFEST_SHA256
        and sidecar.get("member_count") == EXPECTED_MEMBER_COUNT
    ):
        raise ValueError(
            "frozen input integrity sidecar fields differ from preregistration"
        )

    if extraction_root.exists():
        raise FileExistsError(f"refusing to reuse extraction root: {extraction_root}")
    extraction_root.mkdir(parents=True)
    observed: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] | None = None
    manifest_bytes: bytes | None = None
    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or not all(
            safe_archive_name(name) for name in names
        ):
            raise ValueError("input bundle has duplicate or unsafe member names")
        if "BUNDLE_MANIFEST.json" not in names:
            raise ValueError("input bundle is missing BUNDLE_MANIFEST.json")
        manifest_handle = bundle.extractfile(bundle.getmember("BUNDLE_MANIFEST.json"))
        if manifest_handle is None:
            raise ValueError("bundle manifest is not readable")
        manifest_bytes = manifest_handle.read()
        if sha256_bytes(manifest_bytes) != EXPECTED_BUNDLE_MANIFEST_SHA256:
            raise ValueError("embedded bundle manifest SHA-256 mismatch")
        manifest = json.loads(manifest_bytes)
        declared = manifest.get("members")
        if not isinstance(declared, dict) or len(declared) != EXPECTED_MEMBER_COUNT:
            raise ValueError("bundle member manifest is absent or has the wrong size")
        if set(names) != set(declared) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("archive inventory differs from the embedded manifest")
        if manifest.get("official_source", {}).get("commit") != EXPECTED_SOURCE_COMMIT:
            raise ValueError("bundle source revision differs from preregistration")
        if manifest.get("official_dataset_hashes") != EXPECTED_DATASET_HASHES:
            raise ValueError("bundle dataset identities differ from preregistration")
        for relative, expected in EXPECTED_BUNDLE_EXECUTION_HASHES.items():
            if declared.get(relative, {}).get("sha256") != expected:
                raise ValueError(f"bundle execution identity mismatch: {relative}")

        for member in members:
            if not member.isfile():
                raise ValueError(f"non-file archive member rejected: {member.name}")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            destination = (extraction_root / member.name).resolve()
            destination.relative_to(extraction_root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with destination.open("xb") as output:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            record = {"bytes": size, "sha256": digest.hexdigest()}
            observed[member.name] = record
            if member.name == "BUNDLE_MANIFEST.json":
                if record["sha256"] != EXPECTED_BUNDLE_MANIFEST_SHA256:
                    raise ValueError("extracted bundle manifest identity mismatch")
            elif record != declared[member.name]:
                raise ValueError(f"extracted member identity mismatch: {member.name}")

    assert manifest is not None and manifest_bytes is not None
    return {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST_SHA256,
        "member_count": len(manifest["members"]),
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "all_members_stream_verified_before_use": True,
        "all_extracted_members_rehashed": True,
    }


def verify_source_tree_is_pristine(extraction_root: Path) -> dict[str, Any]:
    """Rehash audited source bytes and reject non-bytecode contamination."""
    manifest_path = extraction_root / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get("members")
    if not isinstance(declared, dict):
        raise ValueError("extracted bundle manifest has no member map")
    expected = {
        relative: record
        for relative, record in declared.items()
        if relative.startswith("upstream/")
    }
    upstream_root = extraction_root / "upstream"
    observed_paths = {
        path.relative_to(extraction_root).as_posix()
        for path in upstream_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    expected_paths = set(expected)
    if observed_paths != expected_paths:
        raise ValueError(
            "audited source tree inventory changed after environment setup: "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}"
        )
    observed = {
        relative: {
            "bytes": (extraction_root / relative).stat().st_size,
            "sha256": sha256_file(extraction_root / relative),
        }
        for relative in sorted(expected)
    }
    mismatches = {
        relative: {"expected": expected[relative], "observed": observed[relative]}
        for relative in observed
        if observed[relative] != expected[relative]
    }
    if mismatches:
        raise ValueError(
            f"audited source bytes changed after environment setup: {mismatches}"
        )
    return {
        "source_root": str(upstream_root),
        "file_count": len(observed),
        "inventory_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "all_declared_source_files_match": True,
        "no_unexpected_source_files": True,
        "generated_python_bytecode_ignored_like_official_auditor": True,
        "environment_contamination_absent": True,
    }


def canonical_json_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def result_inventory_normalization_contract(driver_sha256: str) -> dict[str, Any]:
    """Describe the exact false-negative correction without weakening its gate."""
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "normalize_flax_moduledict_actor_presence_false_negatives",
        "driver_sha256": driver_sha256,
        "official_colab_runner_sha256": EXPECTED_COLAB_RUNNER_SHA256,
        "official_return_packer_sha256": EXPECTED_BUNDLE_EXECUTION_HASHES[
            "repro/colab/package_colab_return.py"
        ],
        "registered_module_parameter_counts": EXPECTED_MODULE_PARAMETER_COUNTS,
        "registered_total_parameter_count": EXPECTED_PARAMETER_COUNT,
        "module_key_mapping": {
            "goal_rep": "modules_goal_rep",
            "value": "modules_value",
            "low_actor": "modules_low_actor",
            "high_actor": "modules_high_actor",
        },
        "registered_false_negative_predicates": {
            "includes_low_actor": "'low_actor' in module_parameter_counts",
            "includes_high_actor": "'high_actor' in module_parameter_counts",
        },
        "normalized_fields": ["includes_high_actor", "includes_low_actor"],
        "required_actor_trace_metrics": [
            "high_actor/actor_loss",
            "low_actor/actor_loss",
        ],
        "official_source_files_modified": False,
        "official_package_gate_modified": False,
        "scientific_values_modified": False,
    }


def result_artifact_child(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"unsafe result artifact basename: {name!r}")
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_full_agent_result_inventory(
    output_dir: Path,
    evidence_path: Path,
    *,
    driver_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    """Correct only proven Flax ModuleDict actor-presence false negatives."""
    if runner_sha256 != EXPECTED_COLAB_RUNNER_SHA256:
        raise ValueError("result normalizer refuses an unregistered Colab runner")
    output_root = output_dir.resolve()
    complete_path = output_root / "COMPLETE.json"
    progress_path = output_root / "PROGRESS.json"
    for path in (complete_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    complete = json.loads(complete_path.read_text())
    progress = json.loads(progress_path.read_text())
    if complete != progress:
        raise ValueError(
            "terminal COMPLETE and PROGRESS artifacts differ before normalization"
        )
    if not (
        complete.get("schema_version") == SCHEMA_VERSION
        and complete.get("status") == "complete"
        and complete.get("scope") == "execution_gate_not_paper_scale"
        and complete.get("jax_backend") == "gpu"
        and complete.get("checkpoint_reload_exact") is True
        and complete.get("state_sha256") == complete.get("restored_state_sha256")
        and complete.get("includes_twin_critic") is True
    ):
        raise ValueError("result failed prerequisite scientific and checkpoint gates")
    module_counts = complete.get("module_parameter_counts")
    if module_counts != EXPECTED_MODULE_PARAMETER_COUNTS:
        raise ValueError(
            "module parameter inventory differs from the registered full agent: "
            f"expected={EXPECTED_MODULE_PARAMETER_COUNTS}, observed={module_counts}"
        )
    if not (
        complete.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and sum(module_counts.values()) == EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError(
            "module parameter counts do not exactly partition the full agent"
        )
    original_flags = {
        "includes_high_actor": complete.get("includes_high_actor"),
        "includes_low_actor": complete.get("includes_low_actor"),
    }
    if original_flags != {"includes_high_actor": False, "includes_low_actor": False}:
        raise ValueError(
            "actor flags do not reproduce the registered ModuleDict false negative: "
            f"{original_flags}"
        )

    trace_path = result_artifact_child(output_root, complete.get("training_trace"))
    if sha256_file(trace_path) != complete.get("training_trace_sha256"):
        raise ValueError("actor-presence proof trace failed its registered hash")
    trace = json.loads(trace_path.read_text())
    required_metrics = result_inventory_normalization_contract(driver_sha256)[
        "required_actor_trace_metrics"
    ]
    if not isinstance(trace, list) or not trace:
        raise ValueError("actor-presence proof requires a nonempty training trace")
    for row in trace:
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, dict) or any(
            metric not in metrics or not math.isfinite(float(metrics[metric]))
            for metric in required_metrics
        ):
            raise ValueError(
                "every trace row must contain finite low/high actor losses"
            )

    protocol_path = result_artifact_child(
        output_root, complete.get("protocol_artifact")
    )
    if sha256_file(protocol_path) != complete.get("protocol_artifact_sha256"):
        raise ValueError("normalization refuses an unregistered protocol artifact")
    protocol_artifact = json.loads(protocol_path.read_text())
    protocol_sha256 = protocol_artifact.pop("sha256", None)
    contract = result_inventory_normalization_contract(driver_sha256)
    if not (
        isinstance(protocol_sha256, str)
        and canonical_json_digest(protocol_artifact) == protocol_sha256
        and protocol_sha256 == complete.get("protocol_sha256")
        and protocol_artifact.get("derived_result_inventory_normalization") == contract
        and protocol_artifact.get("execution_files", {}).get("colab_runner_sha256")
        == runner_sha256
    ):
        raise ValueError("protocol does not bind the registered inventory normalizer")

    before_complete_sha256 = sha256_file(complete_path)
    before_progress_sha256 = sha256_file(progress_path)
    normalization_record = {
        "schema_version": SCHEMA_VERSION,
        "purpose": contract["purpose"],
        "contract_sha256": canonical_json_digest(contract),
        "driver_sha256": driver_sha256,
        "official_colab_runner_sha256": runner_sha256,
        "official_return_packer_sha256": contract["official_return_packer_sha256"],
        "original_flags": original_flags,
        "observed_module_parameter_counts": module_counts,
        "observed_total_parameter_count": complete["parameter_count"],
        "actor_metric_trace_rows": len(trace),
        "actor_metric_trace_sha256": complete["training_trace_sha256"],
        "required_actor_trace_metrics_present_and_finite_in_every_row": True,
        "checkpoint_reload_exact": True,
        "checkpoint_state_sha256": complete["state_sha256"],
        "official_source_files_modified": False,
        "official_package_gate_modified": False,
        "scientific_values_modified": False,
    }
    normalized = dict(complete)
    normalized.update(
        {
            "includes_high_actor": True,
            "includes_low_actor": True,
            "derived_result_inventory_normalization": normalization_record,
        }
    )
    atomic_json(complete_path, normalized)
    atomic_json(progress_path, normalized)
    if (
        json.loads(complete_path.read_text()) != normalized
        or json.loads(progress_path.read_text()) != normalized
    ):
        raise AssertionError("normalized result artifact readback differs")
    evidence = {
        **normalization_record,
        "completed_updates": normalized["completed_updates"],
        "protocol_sha256": normalized["protocol_sha256"],
        "before_complete_sha256": before_complete_sha256,
        "before_progress_sha256": before_progress_sha256,
        "after_complete_sha256": sha256_file(complete_path),
        "after_progress_sha256": sha256_file(progress_path),
        "modified_fields": {
            "includes_high_actor": {"before": False, "after": True},
            "includes_low_actor": {"before": False, "after": True},
            "derived_result_inventory_normalization": {
                "before": "absent",
                "after_sha256": canonical_json_digest(normalization_record),
            },
        },
    }
    atomic_json(evidence_path, evidence)
    return evidence


def derive_immutable_config(
    config: dict[str, Any], *, driver_sha256: str, runner_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tuple-normalize only approved list fields and prove semantic identity."""
    if not isinstance(config, dict):
        raise TypeError("the released config loader must return a plain mapping")
    list_fields = sorted(
        key for key, value in config.items() if isinstance(value, list)
    )
    if list_fields != sorted(APPROVED_SEQUENCE_FIELDS):
        raise ValueError(
            "released list-valued config fields differ from the approved patch scope: "
            f"observed={list_fields}, approved={sorted(APPROVED_SEQUENCE_FIELDS)}"
        )
    derived = dict(config)
    field_checks: dict[str, Any] = {}
    for field in APPROVED_SEQUENCE_FIELDS:
        original = config[field]
        if type(original) is not list or not original:
            raise TypeError(
                f"{field} must be a nonempty built-in list before conversion"
            )
        if not all(type(value) is int and value > 0 for value in original):
            raise TypeError(f"{field} must contain only positive built-in integers")
        converted = tuple(original)
        derived[field] = converted
        field_checks[field] = {
            "before_type": "list",
            "after_type": "tuple",
            "length_unchanged": len(original) == len(converted),
            "element_values_unchanged": list(converted) == original,
            "element_types_unchanged": [type(value).__name__ for value in original]
            == [type(value).__name__ for value in converted],
            "values": list(converted),
        }
    scalar_fields = sorted(set(config) - set(APPROVED_SEQUENCE_FIELDS))
    scalar_checks = {
        field: {
            "type": type(config[field]).__name__,
            "value_unchanged": derived[field] == config[field],
            "type_unchanged": type(derived[field]) is type(config[field]),
        }
        for field in scalar_fields
    }
    before_digest = canonical_json_digest(config)
    after_digest = canonical_json_digest(derived)
    if set(config) != set(derived):
        raise AssertionError("derived config key set changed")
    if not all(
        check["value_unchanged"] and check["type_unchanged"]
        for check in scalar_checks.values()
    ):
        raise AssertionError("a scalar config field changed")
    if not all(
        check["length_unchanged"]
        and check["element_values_unchanged"]
        and check["element_types_unchanged"]
        for check in field_checks.values()
    ):
        raise AssertionError("a sequence field changed semantically")
    if before_digest != after_digest:
        raise AssertionError("JSON-semantic config identity changed")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "qio_static_jax_config_list_to_tuple_runtime_derivation",
        "approved_fields": list(APPROVED_SEQUENCE_FIELDS),
        "field_checks": field_checks,
        "scalar_field_checks": scalar_checks,
        "key_set_unchanged": True,
        "json_semantic_sha256_before": before_digest,
        "json_semantic_sha256_after": after_digest,
        "json_semantics_unchanged": True,
        "driver_sha256": driver_sha256,
        "official_colab_runner_sha256": runner_sha256,
        "official_source_files_modified": False,
        "scientific_values_modified": False,
        "type_only_change": "built-in list to tuple for the two approved hidden-dimension fields",
    }
    return derived, evidence


def derived_runner_entry(argv: list[str]) -> None:
    """Run the exact bundled wrapper with a hash-bound config-loader shim."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--inventory-evidence", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = (
        args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    )
    output_positions = [
        index for index, value in enumerate(runner_args) if value == "--output-dir"
    ]
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(runner_args):
        raise ValueError("derived runtime shim requires exactly one --output-dir")
    output_dir = Path(runner_args[output_positions[0] + 1]).resolve()
    runner_path = args.runner.resolve()
    if sha256_file(runner_path) != EXPECTED_COLAB_RUNNER_SHA256:
        raise ValueError("derived runtime shim refuses an unregistered Colab runner")
    driver_path = Path(__file__).resolve()
    driver_sha256 = sha256_file(driver_path)
    spec = importlib.util.spec_from_file_location("qio_exact_colab_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exact Colab runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_loader = runner.core.load_released_config
    original_protocol_payload = runner.protocol_payload

    def immutable_loader(path: Path) -> dict[str, Any]:
        released = original_loader(path)
        derived, evidence = derive_immutable_config(
            released,
            driver_sha256=driver_sha256,
            runner_sha256=EXPECTED_COLAB_RUNNER_SHA256,
        )
        before_hash_error: str | None = None
        try:
            hash(runner.core.FrozenDict(released))
        except TypeError as exc:
            before_hash_error = f"{type(exc).__name__}: {exc}"
        if (
            before_hash_error is None
            or "unhashable type: 'list'" not in before_hash_error
        ):
            raise AssertionError(
                "the registered pre-patch JAX hash failure was not reproduced"
            )
        hash(runner.core.FrozenDict(derived))
        evidence.update(
            {
                "pre_patch_frozendict_hash_failure_reproduced": True,
                "pre_patch_hash_error": before_hash_error,
                "post_patch_frozendict_hash_succeeds": True,
            }
        )
        atomic_json(args.evidence, evidence)
        return derived

    def patch_bound_protocol(
        *protocol_args: Any, **protocol_kwargs: Any
    ) -> dict[str, Any]:
        payload = original_protocol_payload(*protocol_args, **protocol_kwargs)
        if not args.evidence.is_file():
            raise FileNotFoundError("derived runtime patch evidence was not persisted")
        evidence = json.loads(args.evidence.read_text())
        if not evidence.get("json_semantics_unchanged"):
            raise ValueError("derived runtime config did not preserve JSON semantics")
        # The official runner compares a protocol loaded from JSON with the
        # newly built Python object before resuming.  Keep tuples only in the
        # runtime FrozenDict; materialize the protocol copy through JSON so the
        # fresh and resume processes compare identical built-in types.
        protocol_config = payload.get("agent_config")
        if (
            canonical_json_digest(protocol_config)
            != evidence["json_semantic_sha256_after"]
        ):
            raise ValueError("protocol config differs from derived runtime config")
        payload["agent_config"] = json.loads(
            json.dumps(protocol_config, sort_keys=True, separators=(",", ":"))
        )
        if (
            canonical_json_digest(payload["agent_config"])
            != evidence["json_semantic_sha256_before"]
        ):
            raise ValueError("JSON-native protocol config changed scientific values")
        payload["derived_runtime_patch"] = {
            "purpose": evidence["purpose"],
            "approved_fields": evidence["approved_fields"],
            "driver_sha256": driver_sha256,
            "evidence_sha256": sha256_file(args.evidence),
            "json_semantic_sha256": evidence["json_semantic_sha256_after"],
            "runtime_sequence_type": "tuple",
            "protocol_sequence_type": "list",
            "protocol_materialized_as_json_native_types_for_resume_identity": True,
            "official_source_files_modified": False,
            "scientific_values_modified": False,
        }
        payload["derived_result_inventory_normalization"] = (
            result_inventory_normalization_contract(driver_sha256)
        )
        return payload

    runner.core.load_released_config = immutable_loader
    runner.protocol_payload = patch_bound_protocol
    patch_env_key = "QIO_DERIVED_RUNTIME_PATCH_SHA256"
    if patch_env_key not in runner.LEDGER_ENV_KEYS:
        runner.LEDGER_ENV_KEYS = (*runner.LEDGER_ENV_KEYS, patch_env_key)
    os.environ[patch_env_key] = driver_sha256
    sys.argv = [str(runner_path), *runner_args]
    runner.main()
    inventory_evidence = normalize_full_agent_result_inventory(
        output_dir,
        args.inventory_evidence,
        driver_sha256=driver_sha256,
        runner_sha256=EXPECTED_COLAB_RUNNER_SHA256,
    )
    print(json.dumps(inventory_evidence, indent=2, sort_keys=True), flush=True)


def enforce_external_environment_paths(
    source_root: Path, environment_root: Path, cache_root: Path
) -> dict[str, Any]:
    """Prove environment/cache roots are neither inside nor equal to source root."""
    source = source_root.resolve()
    environment = environment_root.resolve()
    cache = cache_root.resolve()
    for label, candidate in (("environment", environment), ("cache", cache)):
        if candidate == source:
            raise ValueError(f"{label} root equals the audited source root")
        try:
            candidate.relative_to(source)
        except ValueError:
            pass
        else:
            raise ValueError(f"{label} root is inside the audited source tree")
        try:
            source.relative_to(candidate)
        except ValueError:
            pass
        else:
            raise ValueError(f"audited source tree is inside the {label} root")
    if environment == cache:
        raise ValueError("UV project environment and cache must also be separate")
    return {
        "source_root": str(source),
        "uv_project_environment": str(environment),
        "uv_cache_dir": str(cache),
        "environment_is_external_to_source": True,
        "cache_is_external_to_source": True,
        "environment_and_cache_are_separate": True,
    }


def parse_nvidia_csv(output: str, fields: list[str]) -> list[dict[str, str]]:
    return [
        {field: value.strip() for field, value in zip(fields, row, strict=True)}
        for row in csv.reader(line for line in output.splitlines() if line.strip())
    ]


def command_output(command: list[str], timeout: float = 30.0) -> str:
    completed = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def dependency_transport_contract(
    extraction_root: Path, uv_executable: str
) -> dict[str, Any]:
    """Bind bounded transport settings to the exact frozen cuDNN artifact."""
    # The outer HF driver is Python >=3.11.  Keep this import local because the
    # same file later serves as a shim under the authors' pinned Python 3.10.
    import tomllib

    lock_path = extraction_root / "upstream" / "uv.lock"
    expected_lock_sha256 = EXPECTED_BUNDLE_EXECUTION_HASHES["upstream/uv.lock"]
    if sha256_file(lock_path) != expected_lock_sha256:
        raise ValueError("dependency transport refuses an unregistered uv.lock")
    lock = tomllib.loads(lock_path.read_text())
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("registered uv.lock has no package inventory")
    matches = [
        package
        for package in packages
        if isinstance(package, dict)
        and package.get("name") == PINNED_CUDNN_PACKAGE["name"]
    ]
    if len(matches) != 1:
        raise ValueError("pinned cuDNN package is not unique in uv.lock")
    observed = {
        key: matches[0].get(key) for key in ("name", "version", "source", "wheels")
    }
    if observed != PINNED_CUDNN_PACKAGE:
        raise ValueError(
            "pinned cuDNN dependency provenance differs from Attempt 3: "
            f"expected={PINNED_CUDNN_PACKAGE}, observed={observed}"
        )
    timeout_seconds = int(DEPENDENCY_TRANSPORT_SETTINGS["UV_HTTP_TIMEOUT"])
    retries = int(DEPENDENCY_TRANSPORT_SETTINGS["UV_HTTP_RETRIES"])
    concurrent_downloads = int(DEPENDENCY_TRANSPORT_SETTINGS["UV_CONCURRENT_DOWNLOADS"])
    if not (
        30 < timeout_seconds <= 600
        and 0 <= retries <= 8
        and 1 <= concurrent_downloads <= 8
    ):
        raise ValueError("dependency transport settings exceed bounded limits")
    return {
        "purpose": "attempt4_bounded_transport_hardening_for_pinned_cudnn_wheel",
        "uv_version": command_output([uv_executable, "--version"]).strip(),
        "settings": DEPENDENCY_TRANSPORT_SETTINGS,
        "settings_sha256": canonical_json_digest(DEPENDENCY_TRANSPORT_SETTINGS),
        "bounds": {
            "http_timeout_seconds_at_most": 600,
            "http_retries_at_most": 8,
            "concurrent_downloads_at_most": 8,
            "overall_internal_deadline_seconds_at_most": 3300,
        },
        "uv_lock_sha256": expected_lock_sha256,
        "pinned_artifact": observed,
        "frozen_lock_dependency_resolution_disabled": True,
        "official_source_files_modified": False,
        "scientific_values_modified": False,
        "derived_runtime_tuple_shim_modified_from_attempt3": False,
    }


def enforce_single_t4() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "compute_capability",
    ]
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    rows = parse_nvidia_csv(command_output(query), fields)
    if len(rows) != 1:
        raise RuntimeError(
            f"exactly one physical T4 is required, observed {len(rows)} GPUs"
        )
    if rows[0]["name"] not in ALLOWED_T4_NAMES:
        raise RuntimeError(
            f"Tesla T4 required; nvidia-smi reported {rows[0]['name']!r}"
        )
    if rows[0]["index"] != "0":
        raise RuntimeError(f"single T4 must be index 0, observed {rows[0]['index']!r}")
    return {
        "required_flavor": "t4-small",
        "required_gpu": "Tesla T4",
        "accepted_nvidia_smi_aliases": sorted(ALLOWED_T4_NAMES),
        "query": query,
        "rows": rows,
        "exactly_one_t4": True,
    }


def process_tree_rss_bytes(root_pid: int) -> int:
    """Best-effort Linux RSS sum for a process and all of its descendants."""
    records: dict[int, tuple[int, int]] = {}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        ppid = 0
        rss_kib = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
        records[int(entry.name)] = (ppid, rss_kib * 1024)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in records.items():
            if pid not in descendants and ppid in descendants:
                descendants.add(pid)
                changed = True
    return sum(records.get(pid, (0, 0))[1] for pid in descendants)


def gpu_sample(root_pid: int | None, stage: str) -> dict[str, Any]:
    fields = [
        "timestamp",
        "index",
        "name",
        "uuid",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
        "temperature_c",
        "power_watts",
    ]
    query = [
        "nvidia-smi",
        (
            "--query-gpu=timestamp,index,name,uuid,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw"
        ),
        "--format=csv,noheader,nounits",
    ]
    try:
        rows = parse_nvidia_csv(command_output(query), fields)
        return {
            "captured_at": utc_now(),
            "stage": stage,
            "rows": rows,
            "process_tree_rss_bytes": (
                process_tree_rss_bytes(root_pid) if root_pid is not None else 0
            ),
        }
    except Exception as exc:  # Evidence must record sampler failures, not hide them.
        return {
            "captured_at": utc_now(),
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
        }


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    samples_path: Path,
    stage: str,
    absolute_deadline: float,
    sample_period_seconds: float,
) -> dict[str, Any]:
    remaining = absolute_deadline - time.monotonic()
    if remaining <= 180:
        raise TimeoutError(
            f"refusing to start {stage}: less than 180 seconds remain for evidence finalization"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    stream_error: list[str] = []

    def copy_output(source: TextIO, target: TextIO) -> None:
        try:
            for line in source:
                target.write(line)
                target.flush()
                print(line, end="", flush=True)
        except Exception as exc:  # Retain the error for the main thread.
            stream_error.append(f"{type(exc).__name__}: {exc}")

    with log_path.open("x", encoding="utf-8") as log_handle:
        thread = threading.Thread(
            target=copy_output, args=(process.stdout, log_handle), daemon=True
        )
        thread.start()
        timed_out = False
        while process.poll() is None:
            append_jsonl(samples_path, gpu_sample(process.pid, stage))
            if time.monotonic() >= absolute_deadline - 180:
                terminate_process(process)
                timed_out = True
                break
            time.sleep(sample_period_seconds)
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError(f"{stage} output thread did not finish")
    append_jsonl(samples_path, gpu_sample(process.pid, stage + ":terminal"))
    finished = time.monotonic()
    record = {
        "stage": stage,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": finished - started,
        "command": command,
        "cwd": str(cwd),
        "returncode": process.returncode,
        "log": log_path.name,
        "log_bytes": log_path.stat().st_size,
        "log_sha256": sha256_file(log_path),
        "stream_errors": stream_error,
        "timed_out_for_evidence_finalization": timed_out,
    }
    if timed_out or process.returncode != 0 or stream_error:
        raise RuntimeError(f"stage failed: {json.dumps(record, sort_keys=True)}")
    return record


def read_memory_evidence(samples_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    valid = [row for row in rows if isinstance(row.get("rows"), list) and row["rows"]]
    if not valid:
        raise ValueError("no valid nvidia-smi resource samples were captured")
    if any(len(row["rows"]) != 1 for row in valid):
        raise ValueError("resource samples do not consistently show exactly one GPU")
    names = {row["rows"][0]["name"] for row in valid}
    if not names.issubset(ALLOWED_T4_NAMES):
        raise ValueError(f"resource samples contain a non-T4 GPU name: {sorted(names)}")
    memory_used = [float(row["rows"][0]["memory_used_mib"]) for row in valid]
    memory_total = [float(row["rows"][0]["memory_total_mib"]) for row in valid]
    rss = [int(row.get("process_tree_rss_bytes", 0)) for row in valid]
    peak_used = max(memory_used)
    minimum_total = min(memory_total)
    return {
        "sample_count": len(rows),
        "valid_sample_count": len(valid),
        "sampler_error_count": len(rows) - len(valid),
        "gpu_names": sorted(names),
        "peak_gpu_memory_used_mib": peak_used,
        "minimum_gpu_memory_total_mib": minimum_total,
        "peak_gpu_memory_fraction": peak_used / minimum_total,
        "peak_process_tree_rss_bytes": max(rss),
        "t4_memory_feasible_below_95_percent": peak_used / minimum_total < 0.95,
        "samples_file": samples_path.name,
        "samples_sha256": sha256_file(samples_path),
    }


def estimate_campaign(result_dir: Path, remaining_budget_usd: float) -> dict[str, Any]:
    complete = json.loads((result_dir / "COMPLETE.json").read_text())
    trace = json.loads((result_dir / "training_trace.json").read_text())
    sessions = json.loads((result_dir / "sessions.json").read_text())
    if not (
        complete.get("status") == "complete"
        and complete.get("scope") == "execution_gate_not_paper_scale"
        and complete.get("completed_updates") == 100
        and complete.get("resume_event_count", 0) >= 1
        and complete.get("checkpoint_reload_exact") is True
        and complete.get("jax_backend") == "gpu"
        and len(sessions) >= 2
    ):
        raise ValueError(
            "100-update output does not satisfy the preregistered resume gate"
        )
    final_session = sessions[-1]
    if (
        final_session.get("status") != "complete"
        or final_session.get("starting_completed_updates") != 10
    ):
        raise ValueError("final phase did not resume the exact update-10 generation")
    session_rows = [
        row for row in trace if row.get("session_id") == final_session.get("session_id")
    ]
    if len(session_rows) < 3:
        raise ValueError("insufficient resumed trace rows for steady-state estimation")
    # The first interval of the second process includes JIT compilation.  Exclude
    # it from steady-state, but retain it and all raw rows in the output archive.
    steady_rows = session_rows[1:]
    rates = [float(row["updates_per_second"]) for row in steady_rows]
    if not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("non-positive or non-finite steady-state throughput")
    median_rate = statistics.median(rates)
    conservative_rate = min(rates)
    one_seed_hours = 1_000_000 / conservative_rate / 3600
    hsvl_four_seed_hours = one_seed_hours * 4
    # Planning estimate for the minimum matched campaign: four HSVL seeds and
    # four matched comparator seeds, with a 25% setup/evaluation contingency.
    # The comparator itself is not measured by this preflight, so the report
    # marks this explicitly as a planning estimate, not observed evidence.
    full_campaign_hours = one_seed_hours * 8 * 1.25
    full_campaign_cost = full_campaign_hours * T4_RATE_USD_PER_HOUR
    return {
        "paper_scale_updates_per_seed": 1_000_000,
        "reported_seed_count": 4,
        "steady_trace_row_count": len(steady_rows),
        "steady_updates_per_second": rates,
        "median_steady_updates_per_second": median_rate,
        "conservative_updates_per_second": conservative_rate,
        "conservative_one_seed_training_hours": one_seed_hours,
        "conservative_hsvl_four_seed_training_hours": hsvl_four_seed_hours,
        "conservative_hsvl_four_seed_training_cost_usd": (
            hsvl_four_seed_hours * T4_RATE_USD_PER_HOUR
        ),
        "full_claim3_planning_assumptions": {
            "hsvl_seeds": 4,
            "matched_comparator_seeds": 4,
            "comparator_throughput_assumed_no_faster_than_measured_hsvl": True,
            "setup_and_evaluation_contingency_fraction": 0.25,
            "policy_success_and_comparator_scientific_outcomes_unmeasured": True,
        },
        "full_claim3_planning_hours": full_campaign_hours,
        "full_claim3_planning_cost_usd": full_campaign_cost,
        "remaining_campaign_budget_usd_at_submission": remaining_budget_usd,
        "full_claim3_planning_estimate_fits_remaining_budget": (
            full_campaign_cost <= remaining_budget_usd
        ),
        "interpretation": (
            "Measured throughput and memory establish execution feasibility only. "
            "The full campaign must not launch unless its centrally rechecked cost "
            "fits the shared ledger; policy evaluation and matched-comparator outcomes "
            "remain unmeasured scientific requirements."
        ),
    }


def artifact_inventory(root: Path, excluded: set[Path]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remaining-budget-usd", type=float, required=True)
    parser.add_argument(
        "--work-root", type=Path, default=Path("/tmp/qio-t4-preflight-work")
    )
    parser.add_argument("--internal-deadline-seconds", type=int, default=3300)
    parser.add_argument("--sample-period-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        parser.error("run-id must use 1-80 safe filename characters")
    if args.remaining_budget_usd < 0:
        parser.error("remaining budget must be non-negative")
    if not 600 <= args.internal_deadline_seconds <= 3300:
        parser.error("internal deadline must be between 600 and 3300 seconds")
    if not 1.0 <= args.sample_period_seconds <= 30.0:
        parser.error("sample period must be between 1 and 30 seconds")
    return args


def main() -> None:
    args = parse_args()
    if PREFLIGHT_TIMEOUT_COST_USD > 0.40:
        raise AssertionError(
            "preflight timeout cost exceeds the authorized per-job bound"
        )
    bucket_root = args.bucket_root.resolve()
    if not bucket_root.is_dir():
        raise FileNotFoundError(f"mounted HF bucket root is absent: {bucket_root}")
    input_root = bucket_root / "hf-jobs" / INPUT_SERIES / "inputs"
    archive_path = input_root / EXPECTED_ARCHIVE_NAME
    sidecar_path = input_root / (EXPECTED_ARCHIVE_NAME + ".integrity.json")
    run_root = bucket_root / "hf-jobs" / OUTPUT_SERIES / "runs" / args.run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite prior run evidence: {run_root}")
    run_root.mkdir(parents=True)
    events_path = run_root / "events.jsonl"
    samples_path = run_root / "resource_samples.jsonl"
    report_path = run_root / "preflight_report.json"
    integrity_path = run_root / "preflight_integrity.json"
    work_run_root = args.work_root.resolve() / args.run_id
    extraction_root = work_run_root / "qio-full-agent"
    author_environment = work_run_root / "author-environment"
    uv_cache = work_run_root / "uv-cache"
    result_dir = run_root / "full_agent_probe" / "seed-0"
    return_archive = run_root / "qio-full-agent-t4-return-v1.tar.gz"
    submitted_driver = Path(__file__).resolve()
    derived_driver = run_root / "derived_runtime_driver.py"
    shutil.copyfile(submitted_driver, derived_driver)
    if sha256_file(derived_driver) != sha256_file(submitted_driver):
        raise ValueError(
            "persisted derived runtime driver differs from submitted bytes"
        )
    patch_evidence = run_root / "derived_runtime_config_patch.json"
    phase_10_patch_evidence = run_root / "derived_runtime_config_patch_phase_10.json"
    inventory_evidence = run_root / "derived_result_inventory_patch.json"
    phase_10_inventory_evidence = (
        run_root / "derived_result_inventory_patch_phase_10.json"
    )
    absolute_deadline = time.monotonic() + args.internal_deadline_seconds
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_version": ATTEMPT_VERSION,
        "purpose": (
            "qio_full_agent_t4_actor_inventory_normalized_execution_resume_preflight"
        ),
        "status": "running",
        "started_at": utc_now(),
        "run_id": args.run_id,
        "scope": "execution_checkpoint_throughput_gate_not_claim3_evidence",
        "prior_attempt_provenance": PRIOR_ATTEMPTS,
        "driver": {
            "submitted_name": submitted_driver.name,
            "persisted_name": derived_driver.name,
            "bytes": derived_driver.stat().st_size,
            "sha256": sha256_file(derived_driver),
            "persisted_in_bucket_before_scientific_execution": True,
        },
        "derived_runtime_patch_contract": {
            "approved_fields": list(APPROVED_SEQUENCE_FIELDS),
            "only_type_change": "built-in list to tuple",
            "json_semantics_must_remain_identical": True,
            "official_source_files_may_be_modified": False,
            "scientific_values_may_be_modified": False,
            "official_colab_runner_sha256": EXPECTED_COLAB_RUNNER_SHA256,
        },
        "result_inventory_normalization_contract": (
            result_inventory_normalization_contract(sha256_file(derived_driver))
        ),
        "hf_job_contract": {
            "required_flavor": "t4-small",
            "external_timeout": "59m",
            "internal_deadline_seconds": args.internal_deadline_seconds,
            "rate_usd_per_hour": T4_RATE_USD_PER_HOUR,
            "timeout_worst_case_cost_usd": PREFLIGHT_TIMEOUT_COST_USD,
            "aggregate_budget_must_be_rechecked_in_central_ledger": True,
        },
        "remaining_budget_usd_at_submission": args.remaining_budget_usd,
        "commands": [],
    }
    atomic_json(report_path, report)
    append_jsonl(events_path, {"at": utc_now(), "event": "preflight_started"})
    try:
        hardware = enforce_single_t4()
        report["hardware"] = hardware
        append_jsonl(events_path, {"at": utc_now(), "event": "single_t4_verified"})
        append_jsonl(samples_path, gpu_sample(None, "hardware_preflight"))

        bundle = validate_and_extract_bundle(
            archive_path, sidecar_path, extraction_root
        )
        report["input_bundle"] = bundle
        append_jsonl(
            events_path,
            {"at": utc_now(), "event": "bundle_verified_and_extracted", **bundle},
        )

        uv = shutil.which("uv")
        if uv is None:
            raise FileNotFoundError("uv executable is required in the HF Jobs image")
        transport = dependency_transport_contract(extraction_root, uv)
        report["dependency_transport_contract"] = transport
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "bounded_dependency_transport_verified",
                **transport,
            },
        )
        environment_separation = enforce_external_environment_paths(
            extraction_root / "upstream", author_environment, uv_cache
        )
        report["environment_separation"] = environment_separation
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "external_environment_paths_verified",
                **environment_separation,
            },
        )
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "JAX_PLATFORM_NAME": "gpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_FLAGS": "",
                "WANDB_MODE": "disabled",
                "MUJOCO_GL": "egl",
                "PYTHONUNBUFFERED": "1",
                "UV_PROJECT_ENVIRONMENT": str(author_environment),
                "UV_CACHE_DIR": str(uv_cache),
                **DEPENDENCY_TRANSPORT_SETTINGS,
            }
        )
        atomic_json(report_path, report)
        stages: list[dict[str, Any]] = []
        sync_command = [
            uv,
            "sync",
            "--frozen",
            "--python",
            "3.10",
            "--project",
            str(extraction_root / "upstream"),
        ]
        stages.append(
            run_logged(
                sync_command,
                cwd=extraction_root,
                env=env,
                log_path=run_root / "uv_sync.log",
                samples_path=samples_path,
                stage="uv_sync_frozen",
                absolute_deadline=absolute_deadline,
                sample_period_seconds=args.sample_period_seconds,
            )
        )
        report["commands"] = stages
        source_after_sync = verify_source_tree_is_pristine(extraction_root)
        report["source_tree_after_uv_sync"] = source_after_sync
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "source_tree_pristine_after_uv_sync",
                **source_after_sync,
            },
        )
        atomic_json(report_path, report)
        python = author_environment / "bin" / "python"
        runner = extraction_root / "repro" / "colab" / "run_full_agent_probe_colab.py"
        packer = extraction_root / "repro" / "colab" / "package_colab_return.py"
        for path in (python, runner, packer):
            if not path.is_file():
                raise FileNotFoundError(path)
        runner_args = [
            "--input-archive",
            str(archive_path),
            "--input-integrity",
            str(sidecar_path),
            "--seed",
            "0",
            "--log-interval",
            "10",
            "--output-dir",
            str(result_dir),
        ]
        common = [
            str(python),
            str(derived_driver),
            DERIVED_RUNNER_FLAG,
            "--runner",
            str(runner),
            "--evidence",
            str(patch_evidence),
            "--inventory-evidence",
            str(inventory_evidence),
            "--",
            *runner_args,
        ]
        phase_10 = common + [
            "--updates",
            "10",
            "--checkpoint-interval",
            "10",
        ]
        stages.append(
            run_logged(
                phase_10,
                cwd=extraction_root,
                env=env,
                log_path=run_root / "phase_10.log",
                samples_path=samples_path,
                stage="phase_10_fresh_process",
                absolute_deadline=absolute_deadline,
                sample_period_seconds=args.sample_period_seconds,
            )
        )
        report["commands"] = stages
        if not patch_evidence.is_file():
            raise FileNotFoundError("phase 10 did not persist derived runtime evidence")
        shutil.copyfile(patch_evidence, phase_10_patch_evidence)
        report["derived_runtime_patch_phase_10"] = {
            "evidence": phase_10_patch_evidence.name,
            "evidence_sha256": sha256_file(phase_10_patch_evidence),
        }
        if not inventory_evidence.is_file():
            raise FileNotFoundError(
                "phase 10 did not persist result inventory normalization evidence"
            )
        phase_10_inventory = json.loads(inventory_evidence.read_text())
        if phase_10_inventory.get("completed_updates") != 10:
            raise ValueError("phase 10 result inventory evidence has the wrong target")
        shutil.copyfile(inventory_evidence, phase_10_inventory_evidence)
        report["result_inventory_normalization_phase_10"] = {
            "evidence": phase_10_inventory_evidence.name,
            "evidence_sha256": sha256_file(phase_10_inventory_evidence),
            "protocol_sha256": phase_10_inventory["protocol_sha256"],
            "modified_fields": phase_10_inventory["modified_fields"],
        }
        source_after_phase_10 = verify_source_tree_is_pristine(extraction_root)
        report["source_tree_after_phase_10"] = source_after_phase_10
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "source_tree_pristine_after_phase_10",
                **source_after_phase_10,
            },
        )
        atomic_json(report_path, report)
        phase_100 = common + [
            "--updates",
            "100",
            "--checkpoint-interval",
            "50",
        ]
        stages.append(
            run_logged(
                phase_100,
                cwd=extraction_root,
                env=env,
                log_path=run_root / "phase_100.log",
                samples_path=samples_path,
                stage="phase_100_independent_resume_process",
                absolute_deadline=absolute_deadline,
                sample_period_seconds=args.sample_period_seconds,
            )
        )
        report["commands"] = stages
        if sha256_file(patch_evidence) != sha256_file(phase_10_patch_evidence):
            raise ValueError("derived runtime evidence changed across resume processes")
        report["derived_runtime_patch_final"] = {
            "evidence": patch_evidence.name,
            "evidence_sha256": sha256_file(patch_evidence),
            "identical_across_fresh_and_resume_processes": True,
        }
        final_inventory = json.loads(inventory_evidence.read_text())
        if not (
            final_inventory.get("completed_updates") == 100
            and final_inventory.get("protocol_sha256")
            == phase_10_inventory["protocol_sha256"]
        ):
            raise ValueError(
                "final result inventory evidence is not bound to the resumed protocol"
            )
        report["result_inventory_normalization_final"] = {
            "evidence": inventory_evidence.name,
            "evidence_sha256": sha256_file(inventory_evidence),
            "protocol_sha256": final_inventory["protocol_sha256"],
            "modified_fields": final_inventory["modified_fields"],
        }
        source_after_phase_100 = verify_source_tree_is_pristine(extraction_root)
        report["source_tree_after_phase_100"] = source_after_phase_100
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "source_tree_pristine_after_phase_100",
                **source_after_phase_100,
            },
        )
        atomic_json(report_path, report)
        package_command = [
            str(python),
            str(packer),
            "--output-dir",
            str(result_dir),
            "--archive",
            str(return_archive),
            "--require-resume-evidence",
        ]
        stages.append(
            run_logged(
                package_command,
                cwd=extraction_root,
                env=env,
                log_path=run_root / "package_return.log",
                samples_path=samples_path,
                stage="package_fail_closed_return",
                absolute_deadline=absolute_deadline,
                sample_period_seconds=args.sample_period_seconds,
            )
        )
        report["commands"] = stages
        report["memory_feasibility"] = read_memory_evidence(samples_path)
        report["campaign_estimate"] = estimate_campaign(
            result_dir, args.remaining_budget_usd
        )
        return_sidecar = return_archive.with_name(
            return_archive.name + ".integrity.json"
        )
        if not return_archive.is_file() or not return_sidecar.is_file():
            raise FileNotFoundError("return packer did not persist both envelope files")
        report["return_artifact"] = {
            "archive": return_archive.name,
            "archive_bytes": return_archive.stat().st_size,
            "archive_sha256": sha256_file(return_archive),
            "integrity_sidecar": return_sidecar.name,
            "integrity_sidecar_bytes": return_sidecar.stat().st_size,
            "integrity_sidecar_sha256": sha256_file(return_sidecar),
            "complete_json_sha256": sha256_file(result_dir / "COMPLETE.json"),
            "training_trace_sha256": sha256_file(result_dir / "training_trace.json"),
            "sessions_sha256": sha256_file(result_dir / "sessions.json"),
        }
        if not report["memory_feasibility"]["t4_memory_feasible_below_95_percent"]:
            raise MemoryError(
                "preflight peak GPU memory reached at least 95% of T4 memory"
            )
        report.update({"status": "passed", "finished_at": utc_now()})
        atomic_json(report_path, report)
        append_jsonl(events_path, {"at": utc_now(), "event": "preflight_passed"})
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        if samples_path.is_file():
            try:
                report["memory_feasibility"] = read_memory_evidence(samples_path)
            except Exception as memory_exc:
                report["memory_evidence_error"] = (
                    f"{type(memory_exc).__name__}: {memory_exc}"
                )
        atomic_json(report_path, report)
        append_jsonl(
            events_path,
            {
                "at": utc_now(),
                "event": "preflight_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        try:
            excluded = {integrity_path.resolve()}
            inventory = artifact_inventory(run_root, excluded)
            atomic_json(
                integrity_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "generated_at": utc_now(),
                    "run_id": args.run_id,
                    "root": run_root.name,
                    "member_count_excluding_this_manifest": len(inventory),
                    "members": inventory,
                },
            )
        except Exception as integrity_exc:
            print(
                f"FATAL: could not write final integrity inventory: "
                f"{type(integrity_exc).__name__}: {integrity_exc}",
                file=sys.stderr,
                flush=True,
            )

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == DERIVED_RUNNER_FLAG:
        derived_runner_entry(sys.argv[2:])
    else:
        main()
