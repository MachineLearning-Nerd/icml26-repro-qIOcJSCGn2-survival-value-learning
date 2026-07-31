#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "throughput"
TIMED_UPDATES = 20
ORX_EXPERIMENT_ID = "88cdd586-9702-455b-9945-34afeb682c40"
CASES = [
    ("hsvl_antmaze", "hsvl", "antmaze-giant-navigate-v0", 1_000_000),
    ("hsvl_humanoid", "hsvl", "humanoidmaze-giant-navigate-v0", 1_000_000),
    ("hsvl_visual_antmaze", "visual_hsvl", "visual-antmaze-giant-navigate-v0", 500_000),
    ("hiql_antmaze", "hiql", "antmaze-giant-navigate-v0", 1_000_000),
    ("hiql_humanoid", "hiql", "humanoidmaze-giant-navigate-v0", 1_000_000),
    ("hiql_visual_antmaze", "hiql", "visual-antmaze-giant-navigate-v0", 500_000),
    ("flat_svl_humanoid", "flat_svl", "humanoidmaze-giant-navigate-v0", 1_000_000),
    ("crl_humanoid", "crl", "humanoidmaze-giant-navigate-v0", 1_000_000),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(path: Path) -> None:
    relative = path.relative_to(ROOT)
    print(f"ORX_ARTIFACT_BEGIN {relative} {sha256(path)} {path.stat().st_size}")
    print(path.read_text(), end="")
    print(f"ORX_ARTIFACT_END {relative}")


def main() -> int:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("JAX_PLATFORMS must be cpu")
    import jax

    devices = [f"{device.platform}:{device.device_kind}" for device in jax.devices()]
    if not devices or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"Expected CPU-only JAX devices, got {devices}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    for name, method, dataset, paper_updates in CASES:
        output_path = ARTIFACTS / f"{name}.json"
        command = [
            sys.executable,
            "repro/campaign/smoke_method.py",
            method,
            dataset,
            str(TIMED_UPDATES),
            str(output_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (ARTIFACTS / f"{name}.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            failures.append(name)
            print(f"===== {name} failed =====", file=sys.stderr)
            print(completed.stdout + completed.stderr, file=sys.stderr)
            continue
        result = json.loads(output_path.read_text())
        seconds = result["steady_update_seconds"]
        result["paper_updates"] = paper_updates
        result["estimated_training_days"] = round(seconds * paper_updates / 86_400, 3)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results[name] = result

    report = {
        "benchmark_kind": "post_jit_paper_scale_updates",
        "campaign_contract_sha256": sha256(ROOT / "repro/campaign/campaign_contract.json"),
        "cpu_count": os.cpu_count(),
        "failures": failures,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "jax_devices": devices,
        "orx_experiment_id": ORX_EXPERIMENT_ID,
        "platform": platform.platform(),
        "results": results,
        "timed_updates_per_case": TIMED_UPDATES,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
    }
    report_path = ARTIFACTS / "throughput.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    for path in sorted(ARTIFACTS.glob("*.json")):
        emit(path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
