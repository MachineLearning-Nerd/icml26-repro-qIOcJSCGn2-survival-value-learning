#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def block(tree) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure_updates(agent, update, timed_updates: int):
    started = time.monotonic()
    agent, info = update(agent)
    block((agent, info))
    compile_seconds = time.monotonic() - started

    steady_seconds = None
    if timed_updates:
        started = time.monotonic()
        for _ in range(timed_updates):
            agent, info = update(agent)
        block((agent, info))
        steady_seconds = (time.monotonic() - started) / timed_updates

    return agent, info, {
        "compile_update_seconds": round(compile_seconds, 3),
        "steady_update_seconds": None if steady_seconds is None else round(steady_seconds, 6),
        "timed_updates": timed_updates,
        "update_seconds": round(compile_seconds, 3),
    }


def load_dataset(dataset_name: str):
    import ogbench

    env, train_dataset, _ = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    observation, info = env.reset(seed=0)
    return env, train_dataset, observation, info


def smoke_hsvl(dataset_name: str, visual: bool = False, timed_updates: int = 0) -> dict:
    sys.path.insert(0, str(ROOT / "upstream"))
    import jax
    import jax.numpy as jnp
    import yaml
    from hsvl.agent import HSVL
    from hsvl.utils.datasets import dataset_size_jax, prepare_hgc_dataset_for_jax

    _, train_dataset, _, _ = load_dataset(dataset_name)
    config = yaml.safe_load((ROOT / "upstream/hsvl/config/agent/hsvl.yaml").read_text())
    config.update(
        frame_stack=None,
        num_log_bins=500,
    )
    if visual:
        sys.path.insert(0, str(ROOT / "repro/campaign"))
        from visual_hsvl import configure_visual_hsvl

        config = configure_visual_hsvl(config)
    if dataset_name.startswith("humanoidmaze-giant"):
        config["discount"] = 0.999
        config["subgoal_steps"] = 100
    dataset = prepare_hgc_dataset_for_jax(train_dataset, device=jax.devices("cpu")[0])
    size = dataset_size_jax(dataset)
    example_observations = jnp.asarray(train_dataset["observations"][:1])
    example_actions = jnp.asarray(train_dataset["actions"][:1])
    agent = HSVL.create(0, example_observations, example_actions, config)
    agent, info, timing = measure_updates(
        agent,
        lambda current: current.sample_and_update(dataset, config["batch_size"], size),
        timed_updates,
    )
    return timing | {
        "actor_num_blocks": config["actor_num_blocks"],
        "actor_residual": config["residual_actor"],
        "batch_size": config["batch_size"],
        "dataset_size": size,
        "discount": config["discount"],
        "encoder": config.get("encoder"),
        "num_log_bins": config["num_log_bins"],
        "subgoal_steps": config["subgoal_steps"],
    }


def smoke_flat_svl(dataset_name: str, timed_updates: int = 0) -> dict:
    sys.path.insert(0, str(ROOT / "upstream"))
    import jax
    import jax.numpy as jnp
    import yaml
    from hsvl.utils.datasets import dataset_size_jax, prepare_hgc_dataset_for_jax

    sys.path.insert(0, str(ROOT / "repro/campaign"))
    from flat_svl import FlatSVL

    _, train_dataset, _, _ = load_dataset(dataset_name)
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
        0,
        jnp.asarray(train_dataset["observations"][:1]),
        jnp.asarray(train_dataset["actions"][:1]),
        config,
    )
    agent, info, timing = measure_updates(
        agent,
        lambda current: current.sample_and_update(dataset, config["batch_size"], size),
        timed_updates,
    )
    return timing | {
        "batch_size": config["batch_size"],
        "dataset_size": size,
        "discount": config["discount"],
        "actor_depth": 6,
        "num_log_bins": config["num_log_bins"],
    }


def smoke_ogbench(method: str, dataset_name: str, timed_updates: int = 0) -> dict:
    impls = ROOT / "upstream/ogbench-hiql-v1.2.1/impls"
    sys.path.insert(0, str(impls))
    import jax
    from agents.crl import CRLAgent, get_config as crl_config
    from agents.hiql import HIQLAgent, get_config as hiql_config
    from utils.datasets import Dataset, GCDataset, HGCDataset

    _, train_dataset, _, _ = load_dataset(dataset_name)
    visual = dataset_name.startswith("visual-")
    if method == "hiql":
        config = hiql_config()
        config.encoder = "impala_small" if visual else None
        config.batch_size = 256 if visual else 1024
        config.discount = 0.995
        config.low_alpha = 3.0
        config.high_alpha = 3.0
        config.low_actor_rep_grad = visual
        config.p_aug = 0.5 if visual else 0.0
        config.subgoal_steps = 100 if "humanoidmaze" in dataset_name else 25
        dataset = HGCDataset(Dataset.create(**train_dataset), config)
        agent_class = HIQLAgent
    elif method == "crl":
        config = crl_config()
        config.encoder = "impala_small" if visual else None
        config.batch_size = 256 if visual else 1024
        config.discount = 0.999 if "humanoidmaze-giant" in dataset_name else 0.995
        config.alpha = 0.1
        config.actor_hidden_dims = (512, 512, 512, 512, 512, 512)
        dataset = GCDataset(Dataset.create(**train_dataset), config)
        agent_class = CRLAgent
    else:
        raise ValueError(method)
    config.frame_stack = None
    example_batch = dataset.sample(config.batch_size)
    agent = agent_class.create(
        0,
        example_batch["observations"][:1],
        example_batch["actions"][:1],
        config,
    )
    agent, info, timing = measure_updates(
        agent,
        lambda current: current.update(dataset.sample(config.batch_size)),
        timed_updates,
    )
    return timing | {
        "actor_depth": len(config.actor_hidden_dims),
        "batch_size": config.batch_size,
        "dataset_size": dataset.size,
        "discount": config.discount,
        "encoder": config.encoder,
    }


def main() -> int:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("JAX_PLATFORMS must be cpu")
    method, dataset_name = sys.argv[1:3]
    timed_updates = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    output_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else None
    if timed_updates < 0:
        raise ValueError("timed_updates must be non-negative")
    np.random.seed(0)
    if method == "hsvl":
        result = smoke_hsvl(dataset_name, timed_updates=timed_updates)
    elif method == "visual_hsvl":
        result = smoke_hsvl(dataset_name, visual=True, timed_updates=timed_updates)
    elif method == "flat_svl":
        result = smoke_flat_svl(dataset_name, timed_updates=timed_updates)
    else:
        result = smoke_ogbench(method, dataset_name, timed_updates=timed_updates)
    result.update({"method": method, "dataset": dataset_name})
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
