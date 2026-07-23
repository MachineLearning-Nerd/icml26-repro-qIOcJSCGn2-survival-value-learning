#!/usr/bin/env python3
"""Restartable real-data probe of the released, full HSVL agent.

This is an execution gate, not a paper-scale reproduction.  It deliberately
keeps the released twin survival critic, low-level actor, high-level actor,
network widths, basis library, horizon, bin request, and batch size.  The
default 100 updates establish that the complete agent can train, checkpoint,
reload, and continue on the official PointMaze data before committing this
CPU-only host to a much longer run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import resource
import sys
import time
from typing import Any


# These must be set before importing JAX.  The probe is intentionally a polite
# single-process CPU workload when it is eventually scheduled alongside other
# reproduction work.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import flax  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402
from flax.core import FrozenDict  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


PAPER_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PAPER_ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from hsvl.agent import HSVL  # noqa: E402
from hsvl.utils.datasets import (  # noqa: E402
    HGCDataset_sample,
    dataset_size_jax,
    prepare_hgc_dataset_for_jax,
)
from audit_full_agent_protocol import verify_upstream_source  # noqa: E402


DATASET_NAME = "pointmaze-large-navigate-v0"
HSVL_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"
OGBENCH_REVISION = "1d4140997f60c52c6fb0702ec100dc988b18c548"
EXPECTED_DATASET_HASHES = {
    "train": "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    "validation": "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def state_digest(state: Any) -> str:
    """Hash a state tree by leaf order, shape, dtype, and exact bytes."""
    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten(state)
    digest.update(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def count_parameters(tree: Any) -> int:
    return int(
        sum(int(np.prod(np.asarray(leaf).shape)) for leaf in jax.tree_util.tree_leaves(tree))
    )


def package_versions() -> dict[str, str]:
    packages = ["hsvl", "ogbench", "jax", "jaxlib", "flax", "optax", "numpy"]
    return {name: importlib.metadata.version(name) for name in packages}


def load_released_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("released agent configuration must be a mapping")
    required = {
        "agent_name": "hsvl",
        "batch_size": 1024,
        "value_hidden_dims": [512, 512, 512],
        "actor_hidden_dims": [512, 512, 512],
        "k_basis": 256,
        "num_basis_sets": 16,
        "rep_dim": 256,
        "surv_horizon": 10000,
        "num_log_bins": 800,
        "ensemble": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"released full-agent configuration mismatch: {mismatches}")
    return config


def protocol_payload(
    config: dict[str, Any],
    dataset_name: str,
    dataset_hashes: dict[str, str],
    seed: int,
    upstream_source: dict[str, Any],
) -> dict[str, Any]:
    """Return extension-safe protocol identity (the target update count is excluded)."""
    return {
        "schema_version": 1,
        "purpose": "released_full_hsvl_real_data_execution_gate",
        "dataset_name": dataset_name,
        "dataset_hashes": dataset_hashes,
        "hsvl_revision": HSVL_REVISION,
        "ogbench_revision": OGBENCH_REVISION,
        "upstream_source": upstream_source,
        "execution_wrapper_sha256": sha256_file(Path(__file__)),
        "provenance_auditor_sha256": sha256_file(
            PAPER_ROOT / "repro" / "src" / "audit_full_agent_protocol.py"
        ),
        "agent_config": config,
        "seed": seed,
        "backend": "cpu",
        "package_versions": package_versions(),
        "execution_environment": {
            key: os.environ.get(key)
            for key in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def checkpoint_reusable(
    manifest: dict[str, Any], protocol_sha256: str, checkpoint_path: Path
) -> bool:
    if manifest.get("schema_version") != 1:
        return False
    if manifest.get("protocol_sha256") != protocol_sha256:
        return False
    if not isinstance(manifest.get("completed_updates"), int):
        return False
    if int(manifest["completed_updates"]) < 0 or not checkpoint_path.is_file():
        return False
    expected_hash = manifest.get("checkpoint_sha256")
    return isinstance(expected_hash, str) and sha256_file(checkpoint_path) == expected_hash


def save_checkpoint(
    agent: HSVL,
    completed_updates: int,
    protocol_sha256: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    agent_state = flax.serialization.to_state_dict(agent)
    payload = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "completed_updates": completed_updates,
        "agent_state": agent_state,
    }
    generation_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}-{completed_updates:08d}{checkpoint_path.suffix}"
    )
    atomic_bytes(
        generation_path,
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "completed_updates": completed_updates,
        "active_checkpoint": generation_path.name,
        "checkpoint_sha256": sha256_file(generation_path),
        "state_sha256": state_digest(agent_state),
        "checkpoint_bytes": generation_path.stat().st_size,
    }
    atomic_json(generation_path.with_suffix(".manifest.json"), manifest)
    atomic_json(checkpoint_path.with_suffix(".manifest.json"), manifest)
    return manifest


def restore_checkpoint(
    template: HSVL,
    protocol_sha256: str,
    checkpoint_path: Path,
) -> tuple[HSVL, dict[str, Any]]:
    manifest_path = checkpoint_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    active_name = manifest.get("active_checkpoint", checkpoint_path.name)
    if not isinstance(active_name, str) or Path(active_name).name != active_name:
        raise ValueError("checkpoint manifest has an unsafe active checkpoint")
    active_path = checkpoint_path.with_name(active_name)
    if not checkpoint_reusable(manifest, protocol_sha256, active_path):
        raise ValueError("checkpoint or protocol manifest failed validation")
    payload = pickle.loads(active_path.read_bytes())
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("checkpoint payload protocol mismatch")
    restored = flax.serialization.from_state_dict(template, payload["agent_state"])
    if state_digest(flax.serialization.to_state_dict(restored)) != manifest["state_sha256"]:
        raise ValueError("restored agent state digest mismatch")
    return restored, manifest


def finite_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics = {key: float(np.asarray(value)) for key, value in info.items()}
    nonfinite = {key: value for key, value in metrics.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"nonfinite full-agent metrics: {nonfinite}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=PAPER_ROOT / "data" / "ogbench")
    parser.add_argument(
        "--config",
        type=Path,
        default=UPSTREAM_ROOT / "hsvl" / "config" / "agent" / "hsvl.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PAPER_ROOT / "outputs" / "full_agent_probe"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.updates < 1 or args.log_interval < 1 or args.checkpoint_interval < 1:
        parser.error("updates and intervals must be positive")
    return args


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"this preregistered probe requires CPU, got {jax.default_backend()}")
    train_path = args.data_dir / f"{DATASET_NAME}.npz"
    validation_path = args.data_dir / f"{DATASET_NAME}-val.npz"
    for path in (train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    if dataset_hashes != EXPECTED_DATASET_HASHES:
        raise ValueError(
            f"official OGBench dataset hash mismatch: expected {EXPECTED_DATASET_HASHES}, "
            f"observed {dataset_hashes}"
        )
    config = load_released_config(args.config)
    agent_config = FrozenDict(config)
    upstream_source = verify_upstream_source()
    protocol = protocol_payload(
        config, DATASET_NAME, dataset_hashes, args.seed, upstream_source
    )
    protocol_sha256 = canonical_digest(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    atomic_json(protocol_path, {**protocol, "sha256": protocol_sha256})

    print("Loading official PointMaze train split...", flush=True)
    train_host, _ = ogbench.make_env_and_datasets(
        DATASET_NAME,
        dataset_path=str(train_path),
        compact_dataset=True,
        dataset_only=True,
    )
    train_dataset = prepare_hgc_dataset_for_jax(train_host, do_terminals=True)
    train_size = dataset_size_jax(train_dataset)
    batch_size = int(config["batch_size"])
    rng = jax.random.PRNGKey(args.seed)
    example, _ = HGCDataset_sample(
        rng, train_dataset, batch_size, train_size, agent_config
    )
    template = HSVL.create(
        seed=args.seed,
        ex_observations=example["observations"],
        ex_actions=example["actions"],
        ex_goals=example["oracle_reps"] if "oracle_reps" in train_dataset else None,
        config=agent_config,
    )
    parameter_count = count_parameters(template.network.params)
    module_parameter_counts = {
        key: count_parameters(value) for key, value in template.network.params.items()
    }
    checkpoint_path = args.output_dir / "checkpoint.pkl"
    checkpoint_manifest_path = checkpoint_path.with_suffix(".manifest.json")
    completed = 0
    agent = template
    if args.resume and checkpoint_manifest_path.is_file():
        agent, manifest = restore_checkpoint(template, protocol_sha256, checkpoint_path)
        completed = int(manifest["completed_updates"])
        if completed > args.updates:
            raise ValueError(
                f"checkpoint has {completed} updates, beyond requested target {args.updates}"
            )
        print(f"Resumed validated checkpoint at update {completed}", flush=True)
    elif checkpoint_path.exists() or checkpoint_manifest_path.exists():
        raise FileExistsError("checkpoint exists but --no-resume was requested")

    trace_path = args.output_dir / "training_trace.json"
    trace = []
    if trace_path.is_file() and completed:
        trace = json.loads(trace_path.read_text())
        trace = [row for row in trace if int(row["update"]) <= completed]
    start = time.perf_counter()
    interval_start = start
    interval_step = completed
    print(
        f"Training released full agent: {completed}->{args.updates} updates, "
        f"parameters={parameter_count:,}, batch={batch_size}",
        flush=True,
    )
    for update in range(completed + 1, args.updates + 1):
        agent, info = agent.sample_and_update(train_dataset, batch_size, train_size)
        if update % args.log_interval == 0 or update == args.updates:
            jax.block_until_ready(agent)
            now = time.perf_counter()
            metrics = finite_metrics(info)
            row = {
                "update": update,
                "interval_seconds": now - interval_start,
                "updates_per_second": (update - interval_step) / (now - interval_start),
                "elapsed_seconds": now - start,
                "metrics": metrics,
            }
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
                    "last_trace": row,
                },
            )
            print(json.dumps(row, sort_keys=True), flush=True)
            interval_start = now
            interval_step = update
        if update % args.checkpoint_interval == 0 or update == args.updates:
            save_checkpoint(agent, update, protocol_sha256, checkpoint_path)

    restored, manifest = restore_checkpoint(template, protocol_sha256, checkpoint_path)
    restored_state_sha256 = state_digest(flax.serialization.to_state_dict(restored))
    completed_seconds = time.perf_counter() - start
    result = {
        "status": "complete",
        "scope": "execution_gate_not_paper_scale",
        "protocol_sha256": protocol_sha256,
        "protocol_artifact": str(protocol_path.relative_to(PAPER_ROOT)),
        "protocol_artifact_sha256": sha256_file(protocol_path),
        "upstream_source": upstream_source,
        "target_updates": args.updates,
        "completed_updates": int(manifest["completed_updates"]),
        "seed": args.seed,
        "dataset_rows": int(len(train_host["observations"])),
        "dataset_valid_pool": train_size,
        "dataset_hashes": dataset_hashes,
        "parameter_count": parameter_count,
        "module_parameter_counts": module_parameter_counts,
        "requested_bins": int(config["num_log_bins"]),
        "actual_bins": int(len(np.asarray(restored._bins)) - 1),
        "includes_twin_critic": bool(config["ensemble"]),
        "includes_low_actor": "low_actor" in module_parameter_counts,
        "includes_high_actor": "high_actor" in module_parameter_counts,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_manifest": str(
            checkpoint_manifest_path.relative_to(PAPER_ROOT)
        ),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "state_sha256": manifest["state_sha256"],
        "restored_state_sha256": restored_state_sha256,
        "checkpoint_reload_exact": restored_state_sha256 == manifest["state_sha256"],
        "runtime_seconds_this_invocation": completed_seconds,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "package_versions": package_versions(),
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
