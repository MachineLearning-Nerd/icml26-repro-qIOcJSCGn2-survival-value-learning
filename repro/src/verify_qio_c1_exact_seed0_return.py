#!/usr/bin/env python3
"""Verify and optionally import an exact-scale qIO C1 return.

The verifier never imports JAX or HSVL and never unpickles a checkpoint.  It
checks the frozen local contract, the complete returned file inventory, the
exact update-1,000,000 lineage, all deterministic policy episodes, raw held-out
arrays, and an independent NumPy recomputation.  A scientific control failure
is preserved as negative evidence rather than treated as corrupt execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PAPER_ROOT / "repro" / "hf" / "qio_c1_exact_seed0_contract.json"
SOURCE_MANIFEST_PATH = (
    PAPER_ROOT / "repro" / "hf" / "qio_c1_exact_seed0_source_manifest.json"
)
AUDITOR_PATH = PAPER_ROOT / "repro" / "src" / "audit_qio_c1_exact_seed0.py"
TARGET_UPDATE = 1_000_000
EXPECTED_TUPLES = 32_768
EXPECTED_EPISODES = 50
EXPECTED_PROTOCOL_SHA256 = "9c54d123aa361d8d6fcfe77a1049445490b89a67ae2749a36e7bf460ac0b11ea"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def safe_relative(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def require_hash(path: Path, expected: str, expected_bytes: int | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular file is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"byte-size mismatch for {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def files_under(root: Path, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    exclude = exclude or set()
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"return contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        if relative.endswith(".tmp") or relative.endswith(".tmp.npz"):
            raise ValueError(f"return contains an unfinished temporary file: {relative}")
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def checksum_inventory(results_dir: Path) -> dict[str, dict[str, Any]]:
    checksum_path = results_dir / "SHA256SUMS"
    declared: dict[str, str] = {}
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in line:
            raise ValueError(f"malformed SHA256SUMS line {number}")
        digest, name = line.split("  ", 1)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not safe_relative(name)
            or name in declared
            or name == "SHA256SUMS"
        ):
            raise ValueError(f"unsafe or duplicate SHA256SUMS line {number}")
        declared[name] = digest
    observed = files_under(results_dir, {"SHA256SUMS"})
    if set(declared) != set(observed):
        raise ValueError("SHA256SUMS does not exactly enumerate the returned files")
    for name, expected in declared.items():
        if observed[name]["sha256"] != expected:
            raise ValueError(f"returned file differs from SHA256SUMS: {name}")
    return observed


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = strict_json(CONTRACT_PATH)
    if not (
        contract.get("schema_version") == 1
        and contract.get("scope", {}).get("claim") == "C1"
        and contract["scope"].get("c1_only") is True
        and contract["scope"].get("claim3") is False
        and contract["scope"].get("seed") == 0
        and contract["exact_protocol"].get("target_update") == TARGET_UPDATE
    ):
        raise ValueError("unexpected local qIO C1 contract")
    components = contract.get("required_frozen_execution_components", {})
    for label, spec in components.items():
        if spec.get("status") != "frozen_static_reviewed":
            raise ValueError(f"local component is not frozen: {label}")
        require_hash(PAPER_ROOT / spec["path"], spec["sha256"])
    manifest = strict_json(SOURCE_MANIFEST_PATH)
    if sha256_file(SOURCE_MANIFEST_PATH) != components["immutable_job_source_manifest"]["sha256"]:
        raise ValueError("local source manifest differs from the frozen contract")
    for relative, expected in manifest.get("files", {}).items():
        if not safe_relative(relative):
            raise ValueError(f"unsafe source-manifest path: {relative}")
        require_hash(PAPER_ROOT / relative, expected)
    return contract, manifest


def verify_protocol(
    results_dir: Path,
    contract: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    protocol_path = results_dir / "execution_protocol.json"
    protocol = strict_json(protocol_path)
    semantic = dict(protocol)
    declared = semantic.pop("sha256", None)
    components = contract["required_frozen_execution_components"]
    expected_components = {label: spec["sha256"] for label, spec in components.items()}
    policy = contract["exact_protocol"]["policy_evaluation"]
    tasks = protocol.get("task_inventory")
    if (
        declared != canonical_digest(semantic)
        or protocol.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or protocol.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH)
        or protocol.get("source_manifest_schema_version") != source_manifest.get("schema_version")
        or protocol.get("returned_protocol_semantic_sha256") != EXPECTED_PROTOCOL_SHA256
        or protocol.get("input_checkpoint_sha256")
        != contract["lineage"]["checkpoint"]["sha256"]
        or protocol.get("input_state_sha256")
        != contract["lineage"]["checkpoint"]["state_sha256"]
        or protocol.get("start_update") != 100
        or protocol.get("target_update") != TARGET_UPDATE
        or protocol.get("checkpoint_interval_updates") != 50_000
        or protocol.get("log_interval_updates") != 1_000
        or protocol.get("component_hashes") != expected_components
        or protocol.get("heldout_twin_survival")
        != contract["exact_protocol"]["heldout_twin_survival"]
        or protocol.get("negative_controls")
        != contract["exact_protocol"]["negative_controls"]
        or protocol.get("policy_temperature") != 0.0
        or protocol.get("policy_gaussian_noise") is not None
        or not isinstance(tasks, list)
        or len(tasks) != 5
        or [task.get("task_name") for task in tasks]
        != [f"task{index}" for index in range(1, 6)]
        or protocol.get("task_inventory_sha256") != canonical_digest(tasks)
    ):
        raise ValueError("returned execution protocol differs from the frozen protocol")
    expected_schedule = [
        {
            "episode_index": index,
            "task_id": index % 5 + 1,
            "task_name": f"task{index % 5 + 1}",
            "reset_seed": policy["reset_seed_start"] + index,
            "actor_seed": policy["actor_seed_start"] + index,
        }
        for index in range(EXPECTED_EPISODES)
    ]
    if protocol.get("policy_schedule") != expected_schedule:
        raise ValueError("returned deterministic policy schedule differs")
    return protocol


def verify_checkpoint_and_ledgers(
    results_dir: Path, completion: dict[str, Any]
) -> dict[str, Any]:
    active_manifest = strict_json(results_dir / "checkpoint.manifest.json")
    generation_manifest = strict_json(results_dir / "checkpoint-01000000.manifest.json")
    if active_manifest != generation_manifest:
        raise ValueError("terminal active and generation checkpoint manifests differ")
    checkpoint_path = results_dir / "checkpoint-01000000.pkl"
    if not (
        active_manifest.get("schema_version") == 1
        and active_manifest.get("completed_updates") == TARGET_UPDATE
        and active_manifest.get("active_checkpoint") == checkpoint_path.name
        and active_manifest.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and completion.get("checkpoint_reload_exact") is True
        and completion.get("checkpoint_sha256") == active_manifest.get("checkpoint_sha256")
        and completion.get("checkpoint_bytes") == active_manifest.get("checkpoint_bytes")
        and completion.get("state_sha256") == active_manifest.get("state_sha256")
        and completion.get("restored_state_sha256") == active_manifest.get("state_sha256")
    ):
        raise ValueError("terminal checkpoint lineage or exact-reload evidence is invalid")
    require_hash(
        checkpoint_path,
        active_manifest["checkpoint_sha256"],
        active_manifest["checkpoint_bytes"],
    )

    trace = strict_json(results_dir / "training_trace.json")
    if not isinstance(trace, list) or not trace:
        raise ValueError("training trace is empty or invalid")
    updates: list[int] = []
    for row in trace:
        update = row.get("update")
        metrics = row.get("metrics")
        if type(update) is not int or not isinstance(metrics, dict) or not metrics:
            raise ValueError("training trace row has invalid structure")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"training trace contains non-finite metrics at update {update}")
        updates.append(update)
    if updates != sorted(set(updates)) or updates[0] > 100 or updates[-1] != TARGET_UPDATE:
        raise ValueError("training trace does not progress uniquely through update 1,000,000")

    sessions = strict_json(results_dir / "sessions.json")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("session ledger is empty or invalid")
    terminal = sessions[-1]
    if not (
        not any(session.get("status") == "running" for session in sessions)
        and terminal.get("status") == "complete"
        and terminal.get("finished_completed_updates") == TARGET_UPDATE
        and terminal.get("ending_checkpoint_sha256") == active_manifest["checkpoint_sha256"]
        and terminal.get("ending_state_sha256") == active_manifest["state_sha256"]
        and terminal.get("jax_backend") == "gpu"
    ):
        raise ValueError("terminal session does not bind the exact GPU completion")
    return active_manifest


def verify_policy(results_dir: Path, protocol: dict[str, Any]) -> None:
    path = results_dir / "policy_episodes.jsonl"
    rows = [strict_json_line(line, path) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError("deterministic policy ledger does not contain exactly 50 episodes")
    for expected, row in zip(protocol["policy_schedule"], rows, strict=True):
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("policy episode differs from the frozen schedule")
        if (
            type(row.get("length")) is not int
            or row["length"] < 1
            or type(row.get("reward")) not in (int, float)
            or type(row.get("success")) not in (int, float)
            or not math.isfinite(row["reward"])
            or not math.isfinite(row["success"])
            or not isinstance(row.get("terminal_info"), dict)
            or type(row.get("terminated")) is not bool
            or type(row.get("truncated")) is not bool
            or not (row["terminated"] or row["truncated"])
        ):
            raise ValueError("policy episode has invalid terminal metrics")
    expected_tasks = []
    for task_id in range(1, 6):
        subset = [row for row in rows if row["task_id"] == task_id]
        if len(subset) != 10:
            raise ValueError("raw policy ledger is not balanced across the five tasks")
        expected_tasks.append(
            {
                "task_id": task_id,
                "task_name": subset[0]["task_name"],
                "episodes": len(subset),
                "success_mean": float(np.mean([row["success"] for row in subset])),
                "reward_mean": float(np.mean([row["reward"] for row in subset])),
                "length_mean": float(np.mean([row["length"] for row in subset])),
            }
        )
    expected_summary = {
        "schema_version": 1,
        "scope": "operational_C1_context_not_C3_benchmark_claim",
        "episodes": EXPECTED_EPISODES,
        "temperature": 0.0,
        "gaussian_noise": None,
        "success_mean": float(np.mean([row["success"] for row in rows])),
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "tasks": expected_tasks,
    }
    summary = strict_json(results_dir / "policy_summary.json")
    if summary != expected_summary:
        raise ValueError("policy summary does not independently reconstruct from raw episodes")


def strict_json_line(line: str, path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    return json.loads(line, parse_constant=reject)


def load_auditor() -> Any:
    spec = importlib.util.spec_from_file_location("qio_c1_frozen_numpy_auditor", AUDITOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen NumPy auditor: {AUDITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_heldout(results_dir: Path) -> dict[str, Any]:
    indices_path = results_dir / "heldout_tuple_indices.npz"
    raw_path = results_dir / "twin_survival_raw.npz"
    with np.load(indices_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "observation_indices",
            "value_goal_indices",
            "tuple_sampler_seed",
            "tuple_count",
        }:
            raise ValueError("held-out index artifact keys differ")
        indices = {key: archive[key] for key in archive.files}
    if not (
        indices["observation_indices"].shape == (EXPECTED_TUPLES,)
        and indices["value_goal_indices"].shape == (EXPECTED_TUPLES,)
        and indices["tuple_sampler_seed"].shape == ()
        and indices["tuple_sampler_seed"].item() == 20_260_719
        and indices["tuple_count"].shape == ()
        and indices["tuple_count"].item() == EXPECTED_TUPLES
    ):
        raise ValueError("held-out tuple index contract differs")
    with np.load(raw_path, allow_pickle=False) as raw:
        if not (
            np.array_equal(raw["observation_indices"], indices["observation_indices"])
            and np.array_equal(raw["value_goal_indices"], indices["value_goal_indices"])
        ):
            raise ValueError("raw survival rows do not bind the frozen tuple indices")
    documents = load_auditor().audit_raw(raw_path)
    saved = {
        "metrics": strict_json(results_dir / "twin_survival_metrics.json"),
        "controls": strict_json(results_dir / "negative_controls.json"),
        "audit": strict_json(results_dir / "independent_audit.json"),
    }
    if documents != saved:
        raise ValueError("independent NumPy recomputation differs from returned audit")
    return documents["audit"]


def verify_bucket_inventories(results_dir: Path, observed: dict[str, dict[str, Any]]) -> None:
    before = strict_json(results_dir / "bucket_inventory_before.json")
    after = strict_json(results_dir / "bucket_inventory_after.json")
    if not (
        before.get("schema_version") == 1
        and before.get("capture") == "before_first_continuation_mutation"
        and isinstance(before.get("files"), dict)
        and after.get("schema_version") == 1
        and after.get("capture") == "after_scientific_outputs_before_terminal_seal"
        and isinstance(after.get("files"), dict)
    ):
        raise ValueError("persistent bucket inventories have invalid structure")
    for inventory in (before["files"], after["files"]):
        for name, record in inventory.items():
            if not safe_relative(name) or not isinstance(record, dict):
                raise ValueError("persistent bucket inventory contains an unsafe entry")
            if name in observed and observed[name] != record:
                raise ValueError(f"persistent bucket inventory hash differs: {name}")
    for name, record in after["files"].items():
        if name not in observed or observed[name] != record:
            raise ValueError(f"after inventory member is missing or changed: {name}")


def verify_completion_bindings(
    results_dir: Path,
    completion: dict[str, Any],
    observed: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    bindings = {
        "execution_protocol_sha256": "execution_protocol.json",
        "training_trace_sha256": "training_trace.json",
        "sessions_sha256": "sessions.json",
        "policy_episodes_sha256": "policy_episodes.jsonl",
        "policy_summary_sha256": "policy_summary.json",
        "heldout_indices_sha256": "heldout_tuple_indices.npz",
        "twin_survival_raw_sha256": "twin_survival_raw.npz",
        "twin_survival_metrics_sha256": "twin_survival_metrics.json",
        "negative_controls_sha256": "negative_controls.json",
        "independent_audit_sha256": "independent_audit.json",
    }
    if not (
        completion.get("schema_version") == 1
        and completion.get("status") == "COMPLETED"
        and completion.get("scope") == "C1_only_not_C3_not_score_guaranteed"
        and completion.get("completed_updates") == TARGET_UPDATE
        and completion.get("target_updates") == TARGET_UPDATE
        and completion.get("seed") == 0
        and completion.get("returned_protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and completion.get("contract_sha256") == sha256_file(CONTRACT_PATH)
        and completion.get("source_manifest_sha256") == sha256_file(SOURCE_MANIFEST_PATH)
        and completion.get("policy_episode_count") == EXPECTED_EPISODES
        and completion.get("heldout_tuple_count") == EXPECTED_TUPLES
        and completion.get("parameter_count") == 9_778_698
        and completion.get("c1_support_candidate") == audit["c1_support_candidate"]
        and completion.get("scientific_interpretation") == audit["scientific_interpretation"]
        and protocol.get("sha256") == canonical_digest({key: value for key, value in protocol.items() if key != "sha256"})
    ):
        raise ValueError("terminal completion sentinel differs from the exact C1 contract")
    for key, name in bindings.items():
        if completion.get(key) != observed[name]["sha256"]:
            raise ValueError(f"completion does not bind returned artifact: {name}")
    progress = strict_json(results_dir / "PROGRESS.json")
    if progress != completion:
        raise ValueError("terminal progress sentinel differs from completion")


def immutable_import(results_dir: Path, import_root: Path) -> Path:
    import_root.mkdir(parents=True, exist_ok=True)
    if import_root.is_symlink():
        raise ValueError("immutable import root may not be a symbolic link")
    identity = sha256_file(results_dir / "completion.json")
    destination = import_root / identity
    source_inventory = files_under(results_dir)
    if destination.exists():
        if not destination.is_dir() or files_under(destination) != source_inventory:
            raise FileExistsError("hash-named immutable import destination already differs")
        return destination
    temporary = import_root / f".{identity}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(results_dir, temporary, symlinks=False)
        if files_under(temporary) != source_inventory:
            raise IOError("immutable import copy verification failed")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def verify(results_dir: Path) -> dict[str, Any]:
    if not results_dir.is_dir() or results_dir.is_symlink():
        raise ValueError("--results-dir must be a regular directory")
    contract, source_manifest = load_frozen_inputs()
    required = set(contract["return_contract"]["required_artifacts"])
    required.update({"bucket_inventory_before.json", "bucket_inventory_after.json", "PROGRESS.json"})
    missing = sorted(name for name in required if not (results_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"required returned artifacts are missing: {missing}")
    if (results_dir / "source_manifest.json").read_bytes() != SOURCE_MANIFEST_PATH.read_bytes():
        raise ValueError("returned immutable source manifest differs byte-for-byte")
    observed = checksum_inventory(results_dir)
    completion = strict_json(results_dir / "completion.json")
    protocol = verify_protocol(results_dir, contract, source_manifest)
    verify_checkpoint_and_ledgers(results_dir, completion)
    verify_policy(results_dir, protocol)
    audit = verify_heldout(results_dir)
    verify_bucket_inventories(results_dir, observed)
    verify_completion_bindings(results_dir, completion, observed, protocol, audit)
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "scope": "C1_only_not_C3_not_score_guaranteed",
        "results_dir": str(results_dir.resolve()),
        "completion_sha256": sha256_file(results_dir / "completion.json"),
        "file_count_excluding_SHA256SUMS": len(observed),
        "completed_updates": TARGET_UPDATE,
        "policy_episode_count": EXPECTED_EPISODES,
        "heldout_tuple_count": EXPECTED_TUPLES,
        "c1_support_candidate": audit["c1_support_candidate"],
        "scientific_interpretation": audit["scientific_interpretation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--import-root", type=Path)
    args = parser.parse_args()
    report = verify(args.results_dir.resolve())
    if args.import_root is not None:
        destination = immutable_import(args.results_dir.resolve(), args.import_root.resolve())
        report["immutable_import_destination"] = str(destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
