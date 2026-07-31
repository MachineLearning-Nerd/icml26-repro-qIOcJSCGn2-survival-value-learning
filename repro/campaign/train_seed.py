#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "repro" / "campaign" / "run_spec.json"
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "training"
ALLOWED = {
    ("hsvl", "antmaze-giant-navigate-v0"),
    ("hsvl", "humanoidmaze-giant-navigate-v0"),
    ("visual_hsvl", "visual-antmaze-giant-navigate-v0"),
    ("hiql", "antmaze-giant-navigate-v0"),
    ("hiql", "humanoidmaze-giant-navigate-v0"),
    ("hiql", "visual-antmaze-giant-navigate-v0"),
    ("flat_svl", "humanoidmaze-giant-navigate-v0"),
    ("crl", "humanoidmaze-giant-navigate-v0"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(tree) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def json_value(value):
    if hasattr(value, "shape") and np.size(value) == 1:
        value = np.asarray(value).item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def load_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    pair = (spec["method"], spec["dataset"])
    if pair not in ALLOWED:
        raise ValueError(f"Unsupported method/dataset pair: {pair}")
    if spec["kind"] not in {"integration", "paper_scale"}:
        raise ValueError(f"Unsupported run kind: {spec['kind']}")
    expected_steps = 500_000 if spec["dataset"].startswith("visual-") else 1_000_000
    expected_episodes = 50
    if spec["kind"] == "paper_scale":
        if spec["train_steps"] != expected_steps:
            raise ValueError(f"Paper-scale run requires {expected_steps} updates")
        if spec["evaluation_episodes_per_task"] != expected_episodes:
            raise ValueError("Paper-scale run requires 50 evaluation episodes per task")
        if spec["seed"] not in {0, 1, 2, 3}:
            raise ValueError("Paper-scale seed must be one of 0, 1, 2, 3")
    for key in ("train_steps", "log_interval", "checkpoint_interval"):
        if int(spec[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("evaluation_episodes_per_task", "negative_control_episodes_per_task"):
        if int(spec[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not spec.get("orx_experiment_id"):
        raise ValueError("orx_experiment_id is required")
    return spec


def load_raw_dataset(dataset_name: str):
    import ogbench

    return ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)


def create_hsvl(spec: dict):
    sys.path.insert(0, str(ROOT / "upstream"))
    import jax
    import jax.numpy as jnp
    import yaml
    from hsvl.agent import HSVL
    from hsvl.utils.datasets import dataset_size_jax, prepare_hgc_dataset_for_jax

    env, train_dataset, validation_dataset = load_raw_dataset(spec["dataset"])
    config = yaml.safe_load((ROOT / "upstream/hsvl/config/agent/hsvl.yaml").read_text())
    config.update(frame_stack=None, num_log_bins=500)
    if spec["method"] == "visual_hsvl":
        sys.path.insert(0, str(ROOT / "repro/campaign"))
        from visual_hsvl import configure_visual_hsvl

        config = configure_visual_hsvl(config)
    if spec["dataset"].startswith("humanoidmaze-giant"):
        config.update(discount=0.999, subgoal_steps=100)
    dataset = prepare_hgc_dataset_for_jax(train_dataset, device=jax.devices("cpu")[0])
    size = dataset_size_jax(dataset)
    agent = HSVL.create(
        spec["seed"],
        jnp.asarray(train_dataset["observations"][:1]),
        jnp.asarray(train_dataset["actions"][:1]),
        config,
    )

    def update(current):
        return current.sample_and_update(dataset, config["batch_size"], size)

    metadata = {
        "actor_hidden_dims": config["actor_hidden_dims"],
        "actor_num_blocks": config["actor_num_blocks"],
        "actor_residual": config["residual_actor"],
        "batch_size": config["batch_size"],
        "dataset_size": size,
        "discount": config["discount"],
        "encoder": config.get("encoder"),
        "num_log_bins": config["num_log_bins"],
        "subgoal_steps": config["subgoal_steps"],
        "survival_horizon": config["surv_horizon"],
        "validation_dataset_size": len(validation_dataset["observations"]),
    }
    return env, agent, update, metadata


def create_flat_svl(spec: dict):
    sys.path.insert(0, str(ROOT / "upstream"))
    sys.path.insert(0, str(ROOT / "repro/campaign"))
    import jax
    import jax.numpy as jnp
    import yaml
    from flat_svl import FlatSVL
    from hsvl.utils.datasets import dataset_size_jax, prepare_hgc_dataset_for_jax

    env, train_dataset, validation_dataset = load_raw_dataset(spec["dataset"])
    config = yaml.safe_load((ROOT / "upstream/hsvl/config/agent/hsvl.yaml").read_text())
    config.update(
        frame_stack=None,
        actor_alpha=0.1,
        discount=0.999,
        num_log_bins=500,
        subgoal_steps=100,
    )
    dataset = prepare_hgc_dataset_for_jax(train_dataset, device=jax.devices("cpu")[0])
    size = dataset_size_jax(dataset)
    agent = FlatSVL.create(
        spec["seed"],
        jnp.asarray(train_dataset["observations"][:1]),
        jnp.asarray(train_dataset["actions"][:1]),
        config,
    )

    def update(current):
        return current.sample_and_update(dataset, config["batch_size"], size)

    metadata = {
        "actor_alpha": config["actor_alpha"],
        "actor_depth": 6,
        "batch_size": config["batch_size"],
        "dataset_size": size,
        "discount": config["discount"],
        "num_log_bins": config["num_log_bins"],
        "survival_horizon": config["surv_horizon"],
        "validation_dataset_size": len(validation_dataset["observations"]),
    }
    return env, agent, update, metadata


def create_ogbench(spec: dict):
    impls = ROOT / "upstream" / "ogbench-hiql-v1.2.1" / "impls"
    sys.path.insert(0, str(impls))
    from agents.crl import CRLAgent, get_config as crl_config
    from agents.hiql import HIQLAgent, get_config as hiql_config
    from utils.datasets import Dataset, GCDataset, HGCDataset

    env, train_dataset, validation_dataset = load_raw_dataset(spec["dataset"])
    visual = spec["dataset"].startswith("visual-")
    if spec["method"] == "hiql":
        config = hiql_config()
        config.encoder = "impala_small" if visual else None
        config.batch_size = 256 if visual else 1024
        config.discount = 0.995
        config.low_alpha = 3.0
        config.high_alpha = 3.0
        config.low_actor_rep_grad = visual
        config.p_aug = 0.5 if visual else 0.0
        config.subgoal_steps = 100 if "humanoidmaze" in spec["dataset"] else 25
        dataset = HGCDataset(Dataset.create(**train_dataset), config)
        agent_class = HIQLAgent
    else:
        config = crl_config()
        config.encoder = None
        config.batch_size = 1024
        config.discount = 0.999
        config.alpha = 0.1
        config.actor_hidden_dims = (512, 512, 512, 512, 512, 512)
        dataset = GCDataset(Dataset.create(**train_dataset), config)
        agent_class = CRLAgent
    config.frame_stack = None
    example = dataset.sample(config.batch_size)
    agent = agent_class.create(
        spec["seed"],
        example["observations"][:1],
        example["actions"][:1],
        config,
    )

    def update(current):
        return current.update(dataset.sample(config.batch_size))

    metadata = {
        "actor_depth": len(config.actor_hidden_dims),
        "batch_size": config.batch_size,
        "dataset_size": dataset.size,
        "discount": config.discount,
        "encoder": config.encoder,
        "image_augmentation_probability": config.p_aug,
        "validation_dataset_size": len(validation_dataset["observations"]),
    }
    return env, agent, update, metadata


def create_run(spec: dict):
    if spec["method"] in {"hsvl", "visual_hsvl"}:
        return create_hsvl(spec)
    if spec["method"] == "flat_svl":
        return create_flat_svl(spec)
    return create_ogbench(spec)


def evaluate(agent, env, seed: int, episodes_per_task: int) -> dict:
    import jax

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    rng = jax.random.PRNGKey(seed + 1_000_003)
    episodes = []
    task_successes = {}
    for task_id, task_info in enumerate(task_infos, start=1):
        successes = []
        for episode_id in range(episodes_per_task):
            observation, info = env.reset(options={"task_id": task_id})
            goal = info.get("goal")
            done = False
            episode_return = 0.0
            length = 0
            while not done:
                rng, action_rng = jax.random.split(rng)
                action = agent.sample_actions(
                    observations=observation,
                    goals=goal,
                    seed=action_rng,
                    temperature=0.0,
                )
                observation, reward, terminated, truncated, info = env.step(
                    np.clip(np.asarray(action), -1, 1)
                )
                done = bool(terminated or truncated)
                episode_return += float(reward)
                length += 1
            success = float(info.get("success", 0.0))
            successes.append(success)
            episodes.append(
                {
                    "episode": episode_id,
                    "length": length,
                    "return": episode_return,
                    "success": success,
                    "task_id": task_id,
                    "task_name": task_info["task_name"],
                }
            )
        task_successes[task_info["task_name"]] = float(np.mean(successes))
    return {
        "episodes": episodes,
        "episodes_per_task": episodes_per_task,
        "overall_success": float(np.mean(list(task_successes.values()))),
        "task_successes": task_successes,
    }


def evaluate_random(env, seed: int, episodes_per_task: int) -> dict:
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    rng = np.random.default_rng(seed + 2_000_003)
    episodes = []
    task_successes = {}
    for task_id, task_info in enumerate(task_infos, start=1):
        successes = []
        for episode_id in range(episodes_per_task):
            _, info = env.reset(options={"task_id": task_id})
            done = False
            length = 0
            while not done:
                action = rng.uniform(env.action_space.low, env.action_space.high)
                _, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                length += 1
            success = float(info.get("success", 0.0))
            successes.append(success)
            episodes.append(
                {
                    "episode": episode_id,
                    "length": length,
                    "success": success,
                    "task_id": task_id,
                    "task_name": task_info["task_name"],
                }
            )
        task_successes[task_info["task_name"]] = float(np.mean(successes))
    return {
        "episodes": episodes,
        "episodes_per_task": episodes_per_task,
        "overall_success": float(np.mean(list(task_successes.values()))),
        "task_successes": task_successes,
    }


def save_checkpoint(
    agent,
    path: Path,
    step: int,
    trace: list[dict],
    training_elapsed_seconds: float,
) -> dict:
    import flax

    payload = {
        "agent": flax.serialization.to_state_dict(agent),
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "run_spec_sha256": sha256(SPEC_PATH),
        "step": step,
        "trace": trace,
        "training_elapsed_seconds": training_elapsed_seconds,
    }
    with path.open("wb") as checkpoint:
        pickle.dump(payload, checkpoint)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size": path.stat().st_size, "step": step}


def restore_checkpoint(agent, path: Path):
    import flax

    with path.open("rb") as checkpoint:
        payload = pickle.load(checkpoint)
    if payload["run_spec_sha256"] != sha256(SPEC_PATH):
        raise RuntimeError(f"Checkpoint run spec does not match {SPEC_PATH}")
    restored = flax.serialization.from_state_dict(agent, payload["agent"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    return (
        restored,
        int(payload["step"]),
        list(payload["trace"]),
        float(payload["training_elapsed_seconds"]),
    )


def checkpoint_roundtrip(agent, path: Path) -> bool:
    import flax
    import jax

    restored, _, _, _ = restore_checkpoint(agent, path)
    original_leaves = jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent))
    restored_leaves = jax.tree_util.tree_leaves(flax.serialization.to_state_dict(restored))
    return len(original_leaves) == len(restored_leaves) and all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(original_leaves, restored_leaves)
    )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def emit(path: Path) -> None:
    relative = path.relative_to(ROOT)
    print(f"ORX_ARTIFACT_BEGIN {relative} {sha256(path)} {path.stat().st_size}")
    print(path.read_text(), end="")
    print(f"ORX_ARTIFACT_END {relative}")


def checkpoint_record(path: Path) -> dict:
    step = int(path.stem.rsplit("_", 1)[1])
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "step": step,
    }


def main() -> int:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("JAX_PLATFORMS must be cpu")
    import jax

    devices = [f"{device.platform}:{device.device_kind}" for device in jax.devices()]
    if not devices or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"Expected CPU-only JAX devices, got {devices}")
    spec = load_spec()
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    output_dir = ARTIFACTS / f"{spec['method']}_{spec['dataset']}_seed{spec['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env, agent, update, method_config = create_run(spec)
    trace = []
    checkpoint_paths = sorted(
        output_dir.glob("checkpoint_*.pkl"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    checkpoints = [checkpoint_record(path) for path in checkpoint_paths]
    resume_step = 0
    prior_training_seconds = 0.0
    if checkpoint_paths:
        agent, resume_step, trace, prior_training_seconds = restore_checkpoint(
            agent, checkpoint_paths[-1]
        )
        print(f"TRAIN_RESUME {checkpoint_paths[-1]} step={resume_step}", flush=True)
    started = time.monotonic()
    last_log = started
    last_step = resume_step
    for step in range(resume_step + 1, spec["train_steps"] + 1):
        agent, info = update(agent)
        if step == 1 or step % spec["log_interval"] == 0 or step == spec["train_steps"]:
            block((agent, info))
            now = time.monotonic()
            metrics = {key: json_value(value) for key, value in info.items()}
            numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
            if any(not math.isfinite(value) for value in numeric):
                raise RuntimeError(f"Non-finite training metric at step {step}: {metrics}")
            record = {
                "elapsed_seconds": round(prior_training_seconds + now - started, 3),
                "interval_steps_per_second": round((step - last_step) / (now - last_log), 6),
                "metrics": metrics,
                "step": step,
            }
            trace.append(record)
            print("TRAIN_PROGRESS " + json.dumps(record, sort_keys=True), flush=True)
            last_log = now
            last_step = step
        if step % spec["checkpoint_interval"] == 0 or step == spec["train_steps"]:
            checkpoint_path = output_dir / f"checkpoint_{step}.pkl"
            checkpoints.append(
                save_checkpoint(
                    agent,
                    checkpoint_path,
                    step,
                    trace,
                    prior_training_seconds + time.monotonic() - started,
                )
            )

    final_checkpoint = output_dir / f"checkpoint_{spec['train_steps']}.pkl"
    roundtrip_passed = checkpoint_roundtrip(agent, final_checkpoint)
    if not roundtrip_passed:
        raise RuntimeError("Checkpoint round-trip comparison failed")
    random_control = evaluate_random(
        env,
        spec["seed"],
        spec["negative_control_episodes_per_task"],
    )
    evaluation = evaluate(agent, env, spec["seed"], spec["evaluation_episodes_per_task"])
    finished = time.monotonic()
    run_summary = {
        "campaign_contract_sha256": sha256(ROOT / "repro/campaign/campaign_contract.json"),
        "checkpoint_roundtrip_passed": roundtrip_passed,
        "cpu_count": os.cpu_count(),
        "dataset": spec["dataset"],
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "jax_devices": devices,
        "kind": spec["kind"],
        "method": spec["method"],
        "method_config": method_config,
        "orx_experiment_id": spec["orx_experiment_id"],
        "overall_success": evaluation["overall_success"],
        "platform": platform.platform(),
        "random_control_overall_success": random_control["overall_success"],
        "resumed_from_step": resume_step,
        "runtime_seconds": round(prior_training_seconds + finished - started, 3),
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "upstream/SOURCE_REVISION",
                ROOT / "upstream/hsvl/agent.py",
                ROOT / "repro/campaign/flat_svl.py",
                ROOT / "repro/campaign/visual_hsvl.py",
                ROOT / "repro/campaign/train_seed.py",
                SPEC_PATH,
            )
        },
        "seed": spec["seed"],
        "train_steps": spec["train_steps"],
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
    }
    write_json(output_dir / "run_spec.json", spec)
    write_json(output_dir / "training_trace.json", trace)
    write_json(output_dir / "checkpoint_manifest.json", checkpoints)
    write_json(output_dir / "negative_control.json", random_control)
    write_json(output_dir / "evaluation.json", evaluation)
    write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    for path in sorted(output_dir.glob("*.json")):
        emit(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
