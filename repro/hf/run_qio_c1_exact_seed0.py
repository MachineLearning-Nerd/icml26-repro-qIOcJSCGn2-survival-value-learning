#!/usr/bin/env python3
"""Exact-scale, resumable seed-0 HSVL continuation for C1 evidence only.

The runner has no Hugging Face control-plane operations. It expects an already
mounted persistent directory, resumes the exact returned update-100 state,
persists immutable checkpoint generations, and evaluates only after an exact
update-1,000,000 reload. It is not C3 and cannot guarantee a score increase.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
from typing import Any
import uuid


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402
from flax.core import FrozenDict  # noqa: E402


PAPER_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PAPER_ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from hsvl.agent import HSVL  # noqa: E402
from hsvl.utils.datasets import (  # noqa: E402
    Dataset_sample_idxs,
    HGCDataset_sample,
    dataset_size_jax,
    prepare_hgc_dataset_for_jax,
)


CONTRACT_PATH = PAPER_ROOT / "repro" / "hf" / "qio_c1_exact_seed0_contract.json"
SOURCE_MANIFEST_PATH = (
    PAPER_ROOT / "repro" / "hf" / "qio_c1_exact_seed0_source_manifest.json"
)
AUDITOR_PATH = PAPER_ROOT / "repro" / "src" / "audit_qio_c1_exact_seed0.py"
DATASET_NAME = "pointmaze-large-navigate-v0"
START_UPDATE = 100
TARGET_UPDATE = 1_000_000
LOG_INTERVAL = 1_000
EXPECTED_PROTOCOL_SHA256 = "9c54d123aa361d8d6fcfe77a1049445490b89a67ae2749a36e7bf460ac0b11ea"
EXPECTED_INPUT_STATE_SHA256 = "7bc393ffaac206b7deaa64de5e67634c4c0556aed1d0ff43e5dc258fbc25d9a0"
EXPECTED_INPUT_CHECKPOINT_SHA256 = "689ed10245e4926249316c79f7502ba692343c583e6fdfcc0de458020f606925"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_digest(state: Any) -> str:
    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten(state)
    digest.update(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def require_hash(path: Path, expected: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"byte-size mismatch for {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def resolve(spec: dict[str, Any]) -> Path:
    return (PAPER_ROOT / spec["path"]).resolve()


def load_and_verify_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = strict_json(CONTRACT_PATH)
    if not (
        contract.get("schema_version") == 1
        and contract.get("scope", {}).get("claim") == "C1"
        and contract["scope"].get("c1_only") is True
        and contract["scope"].get("claim3") is False
        and contract["scope"].get("seed") == 0
        and contract["exact_protocol"].get("start_update") == START_UPDATE
        and contract["exact_protocol"].get("target_update") == TARGET_UPDATE
    ):
        raise ValueError("unexpected qIO C1 execution contract")
    for group in ("lineage", "source_and_data"):
        for label, spec in contract[group].items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            if label == "returned_archive" and not spec.get("required_local", True):
                continue
            require_hash(
                resolve(spec),
                spec.get("sha256", spec.get("file_sha256")),
                spec.get("bytes"),
            )
    components = contract["required_frozen_execution_components"]
    for label, spec in components.items():
        expected = spec.get("sha256")
        if spec.get("status") != "frozen_static_reviewed" or not isinstance(expected, str):
            raise ValueError(f"execution component is not frozen: {label}")
        require_hash(resolve(spec), expected)
    source_manifest = strict_json(SOURCE_MANIFEST_PATH)
    if sha256_file(SOURCE_MANIFEST_PATH) != components["immutable_job_source_manifest"]["sha256"]:
        raise ValueError("source manifest differs from the frozen contract")
    for relative, expected in source_manifest.get("files", {}).items():
        require_hash((PAPER_ROOT / relative).resolve(), expected)

    protocol_spec = contract["lineage"]["protocol"]
    protocol = strict_json(resolve(protocol_spec))
    declared = protocol.pop("sha256", None)
    if declared != EXPECTED_PROTOCOL_SHA256 or canonical_digest(protocol) != declared:
        raise ValueError("returned protocol semantic identity mismatch")
    if protocol_spec["semantic_sha256"] != declared:
        raise ValueError("contract protocol semantic identity mismatch")
    checkpoint_manifest = strict_json(resolve(contract["lineage"]["checkpoint_manifest"]))
    checkpoint = contract["lineage"]["checkpoint"]
    if not (
        checkpoint_manifest.get("completed_updates") == START_UPDATE
        and checkpoint_manifest.get("protocol_sha256") == declared
        and checkpoint_manifest.get("checkpoint_sha256") == checkpoint["sha256"]
        and checkpoint_manifest.get("state_sha256") == checkpoint["state_sha256"]
        and checkpoint["sha256"] == EXPECTED_INPUT_CHECKPOINT_SHA256
        and checkpoint["state_sha256"] == EXPECTED_INPUT_STATE_SHA256
    ):
        raise ValueError("returned checkpoint lineage mismatch")
    return contract, source_manifest, {**protocol, "sha256": declared}


def runtime_agent_config(protocol: dict[str, Any]) -> FrozenDict:
    config = dict(protocol["agent_config"])
    for key in ("actor_hidden_dims", "value_hidden_dims"):
        value = config.get(key)
        if not isinstance(value, list) or not all(type(item) is int for item in value):
            raise ValueError(f"protocol {key} is not the approved JSON list")
        config[key] = tuple(value)
    return FrozenDict(config)


def package_versions() -> dict[str, str]:
    names = ("flax", "hsvl", "jax", "jaxlib", "numpy", "ogbench", "optax")
    return {name: importlib.metadata.version(name) for name in names}


def task_inventory(env: Any) -> list[dict[str, Any]]:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    rows = jsonable(base.task_infos)
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("PointMaze-large canonical task inventory must contain five tasks")
    if [row.get("task_name") for row in rows] != [f"task{index}" for index in range(1, 6)]:
        raise ValueError("PointMaze-large canonical task names changed")
    return rows


def execution_protocol(
    contract: dict[str, Any],
    source_manifest: dict[str, Any],
    returned_protocol: dict[str, Any],
    tasks: list[dict[str, Any]],
    log_interval: int,
) -> dict[str, Any]:
    policy = contract["exact_protocol"]["policy_evaluation"]
    schedule = [
        {
            "episode_index": index,
            "task_id": index % len(tasks) + 1,
            "task_name": tasks[index % len(tasks)]["task_name"],
            "reset_seed": policy["reset_seed_start"] + index,
            "actor_seed": policy["actor_seed_start"] + index,
        }
        for index in range(policy["episodes_total"])
    ]
    payload = {
        "schema_version": 1,
        "scope": "C1_only_not_C3_not_score_guaranteed",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "returned_protocol_file_sha256": sha256_file(
            resolve(contract["lineage"]["protocol"])
        ),
        "returned_protocol_semantic_sha256": returned_protocol["sha256"],
        "input_checkpoint_sha256": EXPECTED_INPUT_CHECKPOINT_SHA256,
        "input_state_sha256": EXPECTED_INPUT_STATE_SHA256,
        "start_update": START_UPDATE,
        "target_update": TARGET_UPDATE,
        "checkpoint_interval_updates": contract["exact_protocol"]["checkpoint_interval_updates"],
        "log_interval_updates": log_interval,
        "dataset_hashes": returned_protocol["dataset_hashes"],
        "component_hashes": {
            label: spec["sha256"]
            for label, spec in contract["required_frozen_execution_components"].items()
        },
        "task_inventory": tasks,
        "task_inventory_sha256": canonical_digest(tasks),
        "policy_schedule": schedule,
        "policy_temperature": policy["temperature"],
        "policy_gaussian_noise": policy["gaussian_noise"],
        "heldout_twin_survival": contract["exact_protocol"]["heldout_twin_survival"],
        "negative_controls": contract["exact_protocol"]["negative_controls"],
        "runtime_package_versions": package_versions(),
        "returned_package_versions": returned_protocol["package_versions"],
        "source_manifest_schema_version": source_manifest["schema_version"],
    }
    payload["sha256"] = canonical_digest(payload)
    return payload


def count_parameters(tree: Any) -> int:
    return int(sum(int(np.prod(np.asarray(leaf).shape)) for leaf in jax.tree_util.tree_leaves(tree)))


def build_template(
    train_dataset: FrozenDict, config: FrozenDict, seed: int
) -> tuple[HSVL, int, dict[str, int]]:
    size = dataset_size_jax(train_dataset)
    batch_size = int(config["batch_size"])
    example, _ = HGCDataset_sample(
        jax.random.PRNGKey(seed), train_dataset, batch_size, size, config
    )
    template = HSVL.create(
        seed=seed,
        ex_observations=example["observations"],
        ex_actions=example["actions"],
        ex_goals=example["oracle_reps"] if "oracle_reps" in train_dataset else None,
        config=config,
    )
    modules = {
        key: count_parameters(value) for key, value in template.network.params.items()
    }
    if sum(modules.values()) != 9_778_698:
        raise ValueError(f"full-agent parameter identity changed: {modules}")
    return template, size, modules


def checkpoint_reusable(manifest: dict[str, Any], directory: Path) -> Path:
    active = manifest.get("active_checkpoint")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or not isinstance(manifest.get("completed_updates"), int)
        or not isinstance(active, str)
        or Path(active).name != active
    ):
        raise ValueError("checkpoint manifest shape or protocol is invalid")
    path = directory / active
    require_hash(path, manifest["checkpoint_sha256"], manifest.get("checkpoint_bytes"))
    return path


def restore_checkpoint(template: HSVL, directory: Path) -> tuple[HSVL, dict[str, Any]]:
    manifest = strict_json(directory / "checkpoint.manifest.json")
    active = checkpoint_reusable(manifest, directory)
    payload = pickle.loads(active.read_bytes())
    if not (
        payload.get("schema_version") == 1
        and payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and payload.get("completed_updates") == manifest["completed_updates"]
    ):
        raise ValueError("checkpoint payload identity mismatch")
    restored = flax.serialization.from_state_dict(template, payload["agent_state"])
    digest = state_digest(flax.serialization.to_state_dict(restored))
    if digest != manifest["state_sha256"]:
        raise ValueError("restored checkpoint state digest mismatch")
    return restored, manifest


def save_checkpoint(agent: HSVL, completed_updates: int, directory: Path) -> dict[str, Any]:
    state = flax.serialization.to_state_dict(agent)
    payload = {
        "schema_version": 1,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "completed_updates": completed_updates,
        "agent_state": state,
    }
    generation = directory / f"checkpoint-{completed_updates:08d}.pkl"
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    serialized_sha256 = hashlib.sha256(serialized).hexdigest()
    if generation.exists():
        if (
            generation.stat().st_size != len(serialized)
            or sha256_file(generation) != serialized_sha256
        ):
            raise FileExistsError(f"immutable checkpoint generation differs: {generation}")
    else:
        atomic_bytes(generation, serialized)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "completed_updates": completed_updates,
        "active_checkpoint": generation.name,
        "checkpoint_sha256": serialized_sha256,
        "state_sha256": state_digest(state),
        "checkpoint_bytes": generation.stat().st_size,
    }
    generation_manifest = generation.with_suffix(".manifest.json")
    if generation_manifest.exists() and strict_json(generation_manifest) != manifest:
        raise FileExistsError(
            f"immutable checkpoint generation manifest differs: {generation_manifest}"
        )
    if not generation_manifest.exists():
        atomic_json(generation_manifest, manifest)
    atomic_json(directory / "checkpoint.manifest.json", manifest)
    return manifest


def initialize_persistent_state(
    work_dir: Path, contract: dict[str, Any]
) -> None:
    manifest_path = work_dir / "checkpoint.manifest.json"
    if manifest_path.exists():
        return
    conflicting = list(work_dir.glob("checkpoint-*.pkl"))
    if conflicting:
        raise FileExistsError("checkpoint generations exist without an active manifest")
    checkpoint = resolve(contract["lineage"]["checkpoint"])
    manifest = strict_json(resolve(contract["lineage"]["checkpoint_manifest"]))
    atomic_bytes(work_dir / "checkpoint-00000100.pkl", checkpoint.read_bytes())
    manifest = {**manifest, "active_checkpoint": "checkpoint-00000100.pkl"}
    atomic_json(work_dir / "checkpoint-00000100.manifest.json", manifest)
    atomic_json(manifest_path, manifest)
    atomic_bytes(
        work_dir / "training_trace.json",
        resolve(contract["lineage"]["training_trace"]).read_bytes(),
    )
    atomic_bytes(
        work_dir / "sessions.json",
        resolve(contract["lineage"]["sessions"]).read_bytes(),
    )


def finite_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics = {key: float(np.asarray(value)) for key, value in info.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("training metrics contain non-finite values")
    return metrics


def train_to_target(
    agent: HSVL,
    manifest: dict[str, Any],
    train_dataset: FrozenDict,
    train_size: int,
    work_dir: Path,
    batch_size: int,
    log_interval: int,
    checkpoint_interval: int,
    session_id: str,
) -> tuple[HSVL, dict[str, Any]]:
    completed = int(manifest["completed_updates"])
    if completed > TARGET_UPDATE:
        raise ValueError("persistent checkpoint is beyond the frozen target")
    trace_path = work_dir / "training_trace.json"
    trace = strict_json(trace_path)
    if not isinstance(trace, list):
        raise ValueError("training trace must be a list")
    trace = [row for row in trace if int(row["update"]) <= completed]
    started = time.perf_counter()
    interval_started = started
    interval_update = completed
    for update in range(completed + 1, TARGET_UPDATE + 1):
        agent, info = agent.sample_and_update(train_dataset, batch_size, train_size)
        if update % log_interval == 0 or update == TARGET_UPDATE:
            jax.block_until_ready(agent)
            now = time.perf_counter()
            row = {
                "update": update,
                "session_id": session_id,
                "interval_seconds": now - interval_started,
                "updates_per_second": (update - interval_update) / (now - interval_started),
                "elapsed_seconds_this_session": now - started,
                "metrics": finite_metrics(info),
            }
            trace.append(row)
            atomic_json(trace_path, trace)
            atomic_json(
                work_dir / "PROGRESS.json",
                {
                    "schema_version": 1,
                    "status": "TRAINING",
                    "completed_updates": update,
                    "target_updates": TARGET_UPDATE,
                    "active_session_id": session_id,
                    "last_trace": row,
                },
            )
            interval_started, interval_update = now, update
        if update % checkpoint_interval == 0 or update == TARGET_UPDATE:
            manifest = save_checkpoint(agent, update, work_dir)
    return agent, manifest


def deterministic_policy_evaluation(
    agent: HSVL,
    env: Any,
    protocol: dict[str, Any],
    work_dir: Path,
) -> list[dict[str, Any]]:
    path = work_dir / "policy_episodes.jsonl"
    schedule = protocol["policy_schedule"]
    rows: list[dict[str, Any]] = []
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) > len(schedule):
        raise ValueError("policy episode ledger exceeds frozen schedule")
    for index, expected in enumerate(schedule[: len(rows)]):
        for key in ("episode_index", "task_id", "task_name", "reset_seed", "actor_seed"):
            if rows[index].get(key) != expected[key]:
                raise ValueError("policy episode ledger differs from frozen schedule")
    max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("environment lacks a finite episode step limit")
    for expected in schedule[len(rows) :]:
        reset_seed = expected["reset_seed"]
        np.random.seed(reset_seed)
        env.action_space.seed(reset_seed)
        observation, info = env.reset(
            seed=reset_seed, options={"task_id": expected["task_id"], "render_goal": False}
        )
        goal = info["goal"]
        actor_key = jax.random.PRNGKey(expected["actor_seed"])
        reward_total = 0.0
        success = 0.0
        terminal_info: dict[str, Any] = {}
        length = 0
        for step in range(1, max_steps + 1):
            actor_key, action_key = jax.random.split(actor_key)
            action = agent.sample_actions(
                observations=jnp.asarray(observation)[None, :],
                goals=jnp.asarray(goal)[None, :],
                seed=action_key,
                temperature=0.0,
            )
            action_np = np.asarray(action)[0]
            observation, reward, terminated, truncated, terminal_info = env.step(action_np)
            reward_total += float(reward)
            success = max(success, float(terminal_info.get("success", 0.0)))
            length = step
            if terminated or truncated:
                break
        else:
            raise RuntimeError("episode exceeded environment step limit without termination")
        row = {
            **expected,
            "reward": reward_total,
            "length": length,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_info": jsonable(terminal_info),
        }
        rows.append(row)
        atomic_text(path, "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    task_rows = []
    for task_id in sorted({row["task_id"] for row in rows}):
        subset = [row for row in rows if row["task_id"] == task_id]
        task_rows.append(
            {
                "task_id": task_id,
                "task_name": subset[0]["task_name"],
                "episodes": len(subset),
                "success_mean": float(np.mean([row["success"] for row in subset])),
                "reward_mean": float(np.mean([row["reward"] for row in subset])),
                "length_mean": float(np.mean([row["length"] for row in subset])),
            }
        )
    atomic_json(
        work_dir / "policy_summary.json",
        {
            "schema_version": 1,
            "scope": "operational_C1_context_not_C3_benchmark_claim",
            "episodes": len(rows),
            "temperature": 0.0,
            "gaussian_noise": None,
            "success_mean": float(np.mean([row["success"] for row in rows])),
            "reward_mean": float(np.mean([row["reward"] for row in rows])),
            "tasks": task_rows,
        },
    )
    return rows


def heldout_batches(
    dataset: FrozenDict,
    config: FrozenDict,
    total: int,
    chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    keys = (
        "observations",
        "value_goals",
        "value_goal_idxs",
        "surv_is_event",
        "surv_time",
        "surv_censor_time",
        "surv_valid",
    )
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    observation_indices: list[np.ndarray] = []
    rng = jax.random.PRNGKey(seed)
    pool = dataset_size_jax(dataset)
    remaining = total
    while remaining:
        current = min(chunk_size, remaining)
        starting_rng = rng
        batch, rng = HGCDataset_sample(starting_rng, dataset, current, pool, config)
        indices, _ = Dataset_sample_idxs(starting_rng, dataset, current, pool)
        jax.block_until_ready((batch, indices))
        if not np.array_equal(
            np.asarray(batch["observations"]),
            np.asarray(dataset["observations"][indices]),
        ):
            raise ValueError("held-out observation indices do not reproduce sampled observations")
        observation_indices.append(np.asarray(indices))
        for key in keys:
            pieces[key].append(np.asarray(batch[key]))
        remaining -= current
    result = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    result["observation_indices"] = np.concatenate(observation_indices)
    return result


def predict_twins(agent: HSVL, observations: np.ndarray, goals: np.ndarray, chunk: int) -> dict[str, np.ndarray]:
    @jax.jit
    def apply(obs: jax.Array, goal: jax.Array):
        return agent.network.select("value")(obs, goal)

    output: dict[str, list[np.ndarray]] = {
        "head1_logit0": [],
        "head1_logits": [],
        "head2_logit0": [],
        "head2_logits": [],
    }
    for start in range(0, len(observations), chunk):
        first, second = apply(
            jnp.asarray(observations[start : start + chunk]),
            jnp.asarray(goals[start : start + chunk]),
        )
        jax.block_until_ready((first, second))
        output["head1_logit0"].append(np.asarray(first[0]))
        output["head1_logits"].append(np.asarray(first[1]))
        output["head2_logit0"].append(np.asarray(second[0]))
        output["head2_logits"].append(np.asarray(second[1]))
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def heldout_evaluation(
    agent: HSVL,
    validation_dataset: FrozenDict,
    config: FrozenDict,
    contract: dict[str, Any],
    work_dir: Path,
) -> None:
    raw_path = work_dir / "twin_survival_raw.npz"
    if not raw_path.is_file():
        spec = contract["exact_protocol"]["heldout_twin_survival"]
        targets = heldout_batches(
            validation_dataset,
            config,
            spec["tuple_count"],
            spec["chunk_size"],
            spec["tuple_sampler_seed"],
        )
        predictions = predict_twins(
            agent, targets["observations"], targets["value_goals"], spec["chunk_size"]
        )
        atomic_npz(
            work_dir / "heldout_tuple_indices.npz",
            observation_indices=targets["observation_indices"],
            value_goal_indices=targets["value_goal_idxs"],
            tuple_sampler_seed=np.asarray(spec["tuple_sampler_seed"], dtype=np.int64),
            tuple_count=np.asarray(spec["tuple_count"], dtype=np.int64),
        )
        atomic_npz(
            raw_path,
            bins=np.asarray(agent._bins),
            observation_indices=targets["observation_indices"],
            value_goal_indices=targets["value_goal_idxs"],
            observations=targets["observations"],
            value_goals=targets["value_goals"],
            surv_is_event=targets["surv_is_event"],
            surv_time=targets["surv_time"],
            surv_censor_time=targets["surv_censor_time"],
            surv_valid=targets["surv_valid"],
            **predictions,
        )
    subprocess.run(
        [sys.executable, str(AUDITOR_PATH), "--results-dir", str(work_dir)],
        check=True,
    )


def directory_inventory(root: Path, excluded: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = excluded or set()
    output = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.endswith(".tmp") or relative.endswith(".tmp.npz"):
            continue
        output[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return output


def seal_return(work_dir: Path, completion: dict[str, Any]) -> None:
    atomic_json(
        work_dir / "bucket_inventory_after.json",
        {
            "schema_version": 1,
            "capture": "after_scientific_outputs_before_terminal_seal",
            "files": directory_inventory(
                work_dir,
                {".execution.lock", "bucket_inventory_after.json", "completion.json", "SHA256SUMS"},
            ),
        },
    )
    atomic_json(work_dir / "completion.json", completion)
    inventory = directory_inventory(work_dir, {".execution.lock", "SHA256SUMS"})
    lines = [f"{record['sha256']}  {name}\n" for name, record in sorted(inventory.items())]
    atomic_text(work_dir / "SHA256SUMS", "".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.work_dir.is_absolute():
        parser.error("--work-dir must be an absolute mounted persistent path")
    return args


def main() -> None:
    args = parse_args()
    contract, source_manifest, returned_protocol = load_and_verify_contract()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"exact-scale continuation requires GPU, got {jax.default_backend()}")
    if package_versions() != returned_protocol["package_versions"]:
        raise RuntimeError("runtime package versions differ from the returned update-100 protocol")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.work_dir.parent / f".{args.work_dir.name}.qio-c1-exact.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError("another exact qIO continuation owns the persistent directory") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    try:
        before_path = args.work_dir / "bucket_inventory_before.json"
        if not before_path.exists():
            atomic_json(
                before_path,
                {
                    "schema_version": 1,
                    "capture": "before_first_continuation_mutation",
                    "files": directory_inventory(args.work_dir, {".execution.lock"}),
                },
            )
        returned_source_manifest = args.work_dir / "source_manifest.json"
        if (
            returned_source_manifest.exists()
            and returned_source_manifest.read_bytes() != SOURCE_MANIFEST_PATH.read_bytes()
        ):
            raise ValueError("persistent source manifest differs from the frozen manifest")
        if not returned_source_manifest.exists():
            atomic_bytes(returned_source_manifest, SOURCE_MANIFEST_PATH.read_bytes())
        train_path = resolve(contract["source_and_data"]["train_data"])
        env, train_host, validation_host = ogbench.make_env_and_datasets(
            DATASET_NAME,
            dataset_path=str(train_path),
            compact_dataset=True,
        )
        tasks = task_inventory(env)
        protocol = execution_protocol(
            contract, source_manifest, returned_protocol, tasks, LOG_INTERVAL
        )
        protocol_path = args.work_dir / "execution_protocol.json"
        if protocol_path.is_file() and strict_json(protocol_path) != protocol:
            raise ValueError("persistent execution protocol differs from frozen protocol")
        atomic_json(protocol_path, protocol)
        initialize_persistent_state(args.work_dir, contract)
        config = runtime_agent_config(returned_protocol)
        train_dataset = prepare_hgc_dataset_for_jax(train_host, do_terminals=True)
        validation_dataset = prepare_hgc_dataset_for_jax(validation_host, do_terminals=True)
        template, train_size, module_counts = build_template(train_dataset, config, 0)
        agent, manifest = restore_checkpoint(template, args.work_dir)
        if manifest["completed_updates"] == START_UPDATE and manifest["state_sha256"] != EXPECTED_INPUT_STATE_SHA256:
            raise ValueError("update-100 persistent state differs from exact returned state")

        sessions_path = args.work_dir / "sessions.json"
        sessions = strict_json(sessions_path)
        if not isinstance(sessions, list):
            raise ValueError("sessions ledger must be a list")
        for previous in sessions:
            if previous.get("status") == "running":
                previous.update(
                    {
                        "status": "interrupted_recovered",
                        "recovered_unix": time.time(),
                        "recovered_at_completed_updates": manifest["completed_updates"],
                        "recovered_checkpoint_sha256": manifest["checkpoint_sha256"],
                    }
                )
        session_id = f"qio-c1-{uuid.uuid4().hex}"
        session = {
            "schema_version": 1,
            "session_id": session_id,
            "status": "running",
            "started_unix": time.time(),
            "starting_completed_updates": manifest["completed_updates"],
            "starting_checkpoint_sha256": manifest["checkpoint_sha256"],
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "platform": platform.platform(),
        }
        sessions.append(session)
        atomic_json(sessions_path, sessions)
        agent, manifest = train_to_target(
            agent,
            manifest,
            train_dataset,
            train_size,
            args.work_dir,
            int(config["batch_size"]),
            LOG_INTERVAL,
            contract["exact_protocol"]["checkpoint_interval_updates"],
            session_id,
        )
        agent, final_manifest = restore_checkpoint(template, args.work_dir)
        final_state = state_digest(flax.serialization.to_state_dict(agent))
        if not (
            final_manifest["completed_updates"] == TARGET_UPDATE
            and final_manifest["active_checkpoint"] == "checkpoint-01000000.pkl"
            and final_state == final_manifest["state_sha256"]
        ):
            raise RuntimeError("terminal checkpoint exact-reload gate failed")
        deterministic_policy_evaluation(agent, env, protocol, args.work_dir)
        heldout_evaluation(agent, validation_dataset, config, contract, args.work_dir)
        audit = strict_json(args.work_dir / "independent_audit.json")
        sessions = strict_json(sessions_path)
        sessions[-1].update(
            {
                "status": "complete",
                "ended_unix": time.time(),
                "finished_completed_updates": TARGET_UPDATE,
                "ending_checkpoint_sha256": final_manifest["checkpoint_sha256"],
                "ending_state_sha256": final_manifest["state_sha256"],
            }
        )
        atomic_json(sessions_path, sessions)
        completion = {
            "schema_version": 1,
            "status": "COMPLETED",
            "scope": "C1_only_not_C3_not_score_guaranteed",
            "completed_updates": TARGET_UPDATE,
            "target_updates": TARGET_UPDATE,
            "seed": 0,
            "checkpoint_reload_exact": True,
            "checkpoint_sha256": final_manifest["checkpoint_sha256"],
            "checkpoint_bytes": final_manifest["checkpoint_bytes"],
            "state_sha256": final_manifest["state_sha256"],
            "restored_state_sha256": final_state,
            "returned_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "execution_protocol_sha256": sha256_file(protocol_path),
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
            "training_trace_sha256": sha256_file(args.work_dir / "training_trace.json"),
            "sessions_sha256": sha256_file(sessions_path),
            "policy_episodes_sha256": sha256_file(args.work_dir / "policy_episodes.jsonl"),
            "policy_summary_sha256": sha256_file(args.work_dir / "policy_summary.json"),
            "heldout_indices_sha256": sha256_file(args.work_dir / "heldout_tuple_indices.npz"),
            "twin_survival_raw_sha256": sha256_file(args.work_dir / "twin_survival_raw.npz"),
            "twin_survival_metrics_sha256": sha256_file(args.work_dir / "twin_survival_metrics.json"),
            "negative_controls_sha256": sha256_file(args.work_dir / "negative_controls.json"),
            "independent_audit_sha256": sha256_file(args.work_dir / "independent_audit.json"),
            "policy_episode_count": 50,
            "heldout_tuple_count": 32_768,
            "parameter_count": sum(module_counts.values()),
            "module_parameter_counts": module_counts,
            "c1_support_candidate": audit["c1_support_candidate"],
            "scientific_interpretation": audit["scientific_interpretation"],
        }
        atomic_json(args.work_dir / "PROGRESS.json", completion)
        seal_return(args.work_dir, completion)
        print(json.dumps(completion, indent=2, sort_keys=True))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


if __name__ == "__main__":
    main()
