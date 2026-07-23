#!/usr/bin/env python3
"""Persistent HF Job wrapper for the frozen qIO C1 seed-0 continuation."""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


ROUTE = "qio-c1-exact-seed0-full-v1"
SOURCE_MANIFEST_SHA256 = "8638e6fffb347a1998772d0771e291a41b6d7ebe1c67b0145098688ddd2ea851"
CONTRACT_SHA256 = "5890f2d07ffeee4e58a99b279f506656816c7469644ad26d1a82c311ed3d4181"
RUNNER_SHA256 = "90ea2ee78cd918d1e39cd2659ce7eeac09e433543a38f8c0a61dab36d8c4fce0"
AUDITOR_SHA256 = "12274e0c550ce0795585c120e6be37ec6a4fb47b6818ba9b86dc97ba606e4198"
VERIFIER_SHA256 = "58c11f186c0075d9d1cc072b5b1a0212e372ebf0d941202e634f766c2ac92c4f"
SOURCE_ARCHIVE_SHA256 = "ff76d286ac579fbdaae777c9c2a306d3b2039bd084ab45da999c2ad406a904be"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in (target, *target.parents) or member.issym() or member.islnk():
                raise ValueError(f"unsafe source archive member: {member.name}")
        tar.extractall(destination)


def verify_sources(root: Path) -> None:
    manifest_path = root / "repro/hf/qio_c1_exact_seed0_source_manifest.json"
    contract_path = root / "repro/hf/qio_c1_exact_seed0_contract.json"
    expected = {
        manifest_path: SOURCE_MANIFEST_SHA256,
        contract_path: CONTRACT_SHA256,
        root / "repro/hf/run_qio_c1_exact_seed0.py": RUNNER_SHA256,
        root / "repro/src/audit_qio_c1_exact_seed0.py": AUDITOR_SHA256,
        root / "repro/src/verify_qio_c1_exact_seed0_return.py": VERIFIER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen component mismatch: {path}")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 31:
        raise ValueError("the frozen source manifest must contain exactly 31 files")
    for relative, digest in files.items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"31-file manifest mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--submission-token", required=True)
    args = parser.parse_args()
    if not args.bucket_root.is_dir() or not os.access(args.bucket_root, os.W_OK):
        raise ValueError("the persistent HF bucket mount is absent or read-only")
    if not args.submission_token.startswith("qio-c1-"):
        raise ValueError("unexpected submission token")

    archive = args.input_root / "qio-c1-exact-seed0-source-v1.tar.gz"
    integrity = json.loads((args.input_root / "qio-c1-exact-seed0-source-v1.integrity.json").read_text())
    if integrity.get("archive_sha256") != SOURCE_ARCHIVE_SHA256 or sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 mismatch")

    route_root = args.bucket_root / "hf-jobs" / ROUTE
    source_root = Path("/tmp/qio-c1-exact-seed0-source")
    runtime_root = Path("/tmp/qio-c1-exact-seed0-runtime")
    results_root = route_root / "state" / "results"
    attempt_root = route_root / "attempts" / args.submission_token
    receipt_path = attempt_root / "route_receipt.json"
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if previous.get("status") == "COMPLETED":
            print(json.dumps(previous, indent=2, sort_keys=True))
            return
        if previous.get("submission_token") != args.submission_token:
            raise ValueError("persistent route identity mismatch")

    if source_root.exists():
        shutil.rmtree(source_root)
    temporary = source_root.with_name(source_root.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    safe_extract(archive, temporary)
    verify_sources(temporary)
    temporary.replace(source_root)
    verify_sources(source_root)

    base = {
        "schema_version": 1,
        "route": ROUTE,
        "submission_token": args.submission_token,
        "source_archive_sha256": integrity["archive_sha256"],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "hf_job_id": os.environ.get("HF_JOB_ID"),
    }
    atomic_json(receipt_path, {**base, "status": "RUNNING", "started_unix": time.time()})
    try:
        gpu_names = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        if len(gpu_names) != 1 or gpu_names[0].strip() not in {"Tesla T4", "NVIDIA T4"}:
            raise RuntimeError(f"exactly one T4 is required, observed {gpu_names!r}")
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
        environment = runtime_root / "author-env"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORM_NAME": "gpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "",
            "WANDB_MODE": "disabled", "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1",
            "UV_PROJECT_ENVIRONMENT": str(environment), "UV_CACHE_DIR": str(runtime_root / "uv-cache"),
            "UV_HTTP_TIMEOUT": "300", "UV_HTTP_RETRIES": "5", "UV_CONCURRENT_DOWNLOADS": "4",
        })
        subprocess.run(
            ["uv", "sync", "--frozen", "--python", "3.10", "--project", str(source_root / "upstream")],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                str(environment / "bin/python"), str(source_root / "repro/hf/run_qio_c1_exact_seed0.py"),
                "--work-dir", str(results_root.resolve()),
            ],
            check=True,
            env=env,
        )
        verification = subprocess.run(
            [
                str(environment / "bin/python"), str(source_root / "repro/src/verify_qio_c1_exact_seed0_return.py"),
                "--results-dir", str(results_root.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        completion = json.loads((results_root / "completion.json").read_text())
        receipt = {
            **base,
            "status": "COMPLETED",
            "completed_unix": time.time(),
            "completion_sha256": sha256(results_root / "completion.json"),
            "checkpoint_sha256": completion["checkpoint_sha256"],
            "c1_support_candidate": completion["c1_support_candidate"],
            "verification_stdout_sha256": hashlib.sha256(verification.stdout.encode()).hexdigest(),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    except BaseException as exc:
        atomic_json(receipt_path, {**base, "status": "INTERRUPTED_OR_FAILED", "failed_unix": time.time(), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
