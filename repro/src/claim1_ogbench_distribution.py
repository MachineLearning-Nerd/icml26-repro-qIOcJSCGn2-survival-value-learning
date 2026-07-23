#!/usr/bin/env python3
"""Claim 1: learn a time-to-goal distribution on real OGBench navigation data.

This is deliberately narrower than the paper's policy-performance experiments.  It
uses the released HSVL goal sampler, PCS censored likelihood, goal encoder, and
SurvivalHazardHead, but omits both actors and reduces the critic size.  The matched
baseline has the same conditioning path and backbone depth, but emits only one
goal-conditioned scalar hazard repeated at every time bin (a geometric model).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import flax.linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax

from hsvl.utils.common import create_log_bins
from hsvl.utils.critics import GCSurvivalValue
from hsvl.utils.datasets import (
    HGCDataset_sample,
    dataset_size_jax,
    prepare_hgc_dataset_for_jax,
)
from hsvl.utils.encoders import GCEncoder
from hsvl.utils.mlp import Identity, LengthNormalize, MLP
from hsvl.utils.survival import create_perbin_nll_fn


DATASET_NAME = "pointmaze-large-navigate-v0"
DATASET_URL_ROOT = "https://rail.eecs.berkeley.edu/datasets/ogbench"
UPSTREAM_HSVL_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"
UPSTREAM_OGBENCH_REVISION = "1d4140997f60c52c6fb0702ec100dc988b18c548"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    packages = ["hsvl", "ogbench", "jax", "jaxlib", "flax", "optax", "numpy", "scipy"]
    return {name: importlib.metadata.version(name) for name in packages}


def make_sampler_config(horizon: int) -> FrozenDict:
    """Released HSVL defaults relevant to tuple construction (frame_stack=1)."""
    return FrozenDict(
        dict(
            frame_stack=1,
            value_p_curgoal=0.08,
            value_p_trajgoal=0.60,
            value_p_randomgoal=0.32,
            value_geom_sample=True,
            discount=0.995,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            subgoal_steps=25,
            surv_horizon=horizon,
        )
    )


def make_official_pcs_module(
    num_bins: int,
    hidden_dim: int,
    rep_dim: int,
    k_basis: int,
    num_basis_sets: int,
) -> GCSurvivalValue:
    """Construct the released value path, with explicitly reduced dimensions."""
    goal_rep = nn.Sequential(
        [
            MLP(
                hidden_dims=(hidden_dim, hidden_dim, hidden_dim, rep_dim),
                activate_final=False,
                layer_norm=True,
            ),
            LengthNormalize(),
        ]
    )
    encoder = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep)
    return GCSurvivalValue(
        hidden_dims=(hidden_dim, hidden_dim, hidden_dim),
        num_bins=num_bins,
        k_basis=k_basis,
        num_basis_sets=num_basis_sets,
        layer_norm=True,
        gc_encoder=encoder,
        activation=nn.gelu,
        ensemble=False,
        use_residual=False,
        num_blocks=2,
    )


class GeometricHazardBaseline(nn.Module):
    """Comparable goal-conditioned backbone with one repeated hazard parameter."""

    num_bins: int
    hidden_dim: int
    rep_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        goal_rep = nn.Sequential(
            [
                MLP(
                    hidden_dims=(
                        self.hidden_dim,
                        self.hidden_dim,
                        self.hidden_dim,
                        self.rep_dim,
                    ),
                    activate_final=False,
                    layer_norm=True,
                    name="goal_rep_mlp",
                ),
                LengthNormalize(name="goal_rep_norm"),
            ],
            name="goal_rep",
        )(jnp.concatenate([observations, goals], axis=-1))
        z = jnp.concatenate([observations, goal_rep], axis=-1)
        h = MLP(
            hidden_dims=(self.hidden_dim, self.hidden_dim, self.hidden_dim),
            activate_final=True,
            layer_norm=True,
            name="backbone",
        )(z)
        scalar_logit = nn.Dense(1, name="scalar_hazard")(h).squeeze(-1)
        return scalar_logit, jnp.repeat(scalar_logit[:, None], self.num_bins, axis=1)


def tree_parameter_count(params: Any) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def sample_validation_tuples(
    dataset: FrozenDict,
    sampler_cfg: FrozenDict,
    total_size: int,
    chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    keys = [
        "observations",
        "value_goals",
        "surv_is_event",
        "surv_time",
        "surv_censor_time",
        "surv_valid",
    ]
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    rng = jax.random.PRNGKey(seed)
    remaining = total_size
    pool_size = dataset_size_jax(dataset)
    while remaining:
        current = min(chunk_size, remaining)
        batch, rng = HGCDataset_sample(rng, dataset, current, pool_size, sampler_cfg)
        jax.block_until_ready(batch)
        for key in keys:
            chunks[key].append(np.asarray(batch[key]))
        remaining -= current
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def make_states(
    model: nn.Module,
    rng: jax.Array,
    observations: jax.Array,
    goals: jax.Array,
    learning_rate: float,
) -> train_state.TrainState:
    params = model.init(rng, observations, goals)["params"]
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def make_update(nll_fn):
    @jax.jit
    def update(state: train_state.TrainState, batch: dict[str, jax.Array]):
        def loss_fn(params):
            output = state.apply_fn(
                {"params": params}, batch["observations"], batch["value_goals"]
            )
            # Released GCSurvivalValue returns an identical two-head tuple when ensemble=False.
            if len(output) == 2 and isinstance(output[0], tuple):
                logit0, logits = output[0]
            else:
                logit0, logits = output
            per_item = nll_fn(
                logit0,
                logits,
                batch["surv_is_event"],
                batch["surv_time"],
                batch["surv_censor_time"],
            )
            valid = batch["surv_valid"]
            return jnp.sum(valid * per_item) / jnp.maximum(jnp.sum(valid), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return update


def predict(model_state, observations: np.ndarray, goals: np.ndarray, batch_size: int):
    @jax.jit
    def apply(params, obs, goal):
        output = model_state.apply_fn({"params": params}, obs, goal)
        if len(output) == 2 and isinstance(output[0], tuple):
            return output[0]
        return output

    all_logit0, all_logits = [], []
    for start in range(0, len(observations), batch_size):
        pair = apply(
            model_state.params,
            jnp.asarray(observations[start : start + batch_size]),
            jnp.asarray(goals[start : start + batch_size]),
        )
        jax.block_until_ready(pair)
        all_logit0.append(np.asarray(pair[0]))
        all_logits.append(np.asarray(pair[1]))
    return np.concatenate(all_logit0), np.concatenate(all_logits)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x))


def reconstruct_distribution_numpy(logit0: np.ndarray, logits: np.ndarray):
    """Independent NumPy reconstruction, not the released JAX value helper."""
    p0 = sigmoid(logit0.astype(np.float64))
    hazards = sigmoid(logits.astype(np.float64))
    one_minus_h = 1.0 - hazards
    before = np.concatenate(
        [
            np.ones((len(logits), 1), dtype=np.float64),
            np.cumprod(one_minus_h, axis=1)[:, :-1],
        ],
        axis=1,
    )
    survival_start = (1.0 - p0)[:, None] * before
    interval_mass = survival_start * hazards
    survival_end = (1.0 - p0)[:, None] * np.cumprod(one_minus_h, axis=1)
    tail_mass = survival_end[:, -1]
    probabilities = np.concatenate(
        [p0[:, None], interval_mass, tail_mass[:, None]], axis=1
    )
    return probabilities, hazards, survival_start, survival_end


def nll_numpy(
    bins: np.ndarray,
    logit0: np.ndarray,
    logits: np.ndarray,
    is_event: np.ndarray,
    tau: np.ndarray,
    censor: np.ndarray,
) -> np.ndarray:
    """Independent NumPy implementation of the released grouped PCS likelihood."""
    logp0 = -np.logaddexp(0.0, -logit0.astype(np.float64))
    log1mp0 = -np.logaddexp(0.0, logit0.astype(np.float64))
    logh = -np.logaddexp(0.0, -logits.astype(np.float64))
    log1mh = -np.logaddexp(0.0, logits.astype(np.float64))
    prefix = np.cumsum(log1mh, axis=1)
    prefix_before = np.concatenate([np.zeros((len(logits), 1)), prefix[:, :-1]], axis=1)
    rows = np.arange(len(logits))
    tau_bin = np.clip(
        np.searchsorted(bins, tau, side="right") - 1, 0, logits.shape[1] - 1
    )
    ll_event = np.where(
        tau == 0, logp0, log1mp0 + prefix_before[rows, tau_bin] + logh[rows, tau_bin]
    )
    censor_edge = np.clip(
        np.searchsorted(bins, censor, side="left"), 0, logits.shape[1]
    )
    survival_terms = np.zeros(len(logits), dtype=np.float64)
    positive = censor_edge > 0
    survival_terms[positive] = prefix[rows[positive], censor_edge[positive] - 1]
    ll_censor = log1mp0 + survival_terms
    return -np.where(is_event > 0.5, ll_event, ll_censor)


def distribution_metrics(
    bins: np.ndarray,
    logit0: np.ndarray,
    logits: np.ndarray,
    targets: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities, hazards, _survival_start, survival_end = (
        reconstruct_distribution_numpy(logit0, logits)
    )
    event = targets["surv_is_event"] > 0.5
    tau = targets["surv_time"].astype(np.int64)
    censor = targets["surv_censor_time"].astype(np.int64)
    nll = nll_numpy(bins, logit0, logits, event, tau, censor)

    edges = bins[1:]
    # Grouped PCS diagnostic: an event assigned to [b_k, b_{k+1}) has survived
    # preceding boundaries, hence tau >= edge.  This is intentionally not called
    # an exact per-step S(t)=P(T>t) identity check.
    known = event[:, None] | (edges[None, :] <= censor[:, None])
    observed_survival = np.where(event[:, None], tau[:, None] >= edges[None, :], 1.0)
    squared_error = np.square(survival_end - observed_survival)
    absolute_error = np.abs(survival_end - observed_survival)
    brier = float(squared_error[known].mean())
    survival_mae = float(absolute_error[known].mean())

    eps = np.finfo(np.float64).tiny
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, eps)), axis=1)
    effective_support = np.exp(entropy)
    cdf = np.cumsum(probabilities[:, :-1], axis=1)
    median_idx = np.argmax(cdf >= 0.5, axis=1)
    no_median = ~np.any(cdf >= 0.5, axis=1)
    median_time = np.where(
        median_idx == 0, 0, bins[np.clip(median_idx, 1, len(bins) - 1)]
    )
    median_time = median_time.astype(np.float64)
    median_time[no_median] = float(bins[-1] + 1)

    metrics: dict[str, Any] = {
        "nll_mean": float(nll.mean()),
        "nll_se": float(nll.std(ddof=1) / math.sqrt(len(nll))),
        "brier_known_times": brier,
        "survival_mae_known_times": survival_mae,
        "normalization_max_abs_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
        "minimum_probability": float(probabilities.min()),
        "monotonicity_max_increase": float(
            np.maximum(np.diff(survival_end, axis=1), 0.0).max(initial=0.0)
        ),
        "mean_tail_mass": float(probabilities[:, -1].mean()),
        "mean_entropy_nats": float(entropy.mean()),
        "mean_effective_support": float(effective_support.mean()),
        "median_effective_support": float(np.median(effective_support)),
        "fraction_effective_support_gt_2": float(np.mean(effective_support > 2.0)),
        "mean_temporal_hazard_std": float(np.std(hazards, axis=1).mean()),
        "std_predicted_median_time": float(np.std(median_time)),
        "fraction_predicted_median_beyond_horizon": float(no_median.mean()),
    }
    return metrics, nll


def distance_strata(
    bins: np.ndarray,
    logit0: np.ndarray,
    logits: np.ndarray,
    targets: dict[str, np.ndarray],
    num_strata: int = 5,
) -> list[dict[str, Any]]:
    distances = np.linalg.norm(targets["observations"] - targets["value_goals"], axis=1)
    quantiles = np.quantile(distances, np.linspace(0.0, 1.0, num_strata + 1))
    # Quantile ties are rare for continuous PointMaze coordinates but are made safe.
    quantiles = np.maximum.accumulate(quantiles)
    result = []
    for idx in range(num_strata):
        if idx + 1 == num_strata:
            mask = (distances >= quantiles[idx]) & (distances <= quantiles[idx + 1])
        else:
            mask = (distances >= quantiles[idx]) & (distances < quantiles[idx + 1])
        subset = {key: value[mask] for key, value in targets.items()}
        metrics, _ = distribution_metrics(bins, logit0[mask], logits[mask], subset)
        event = subset["surv_is_event"] > 0.5
        result.append(
            {
                "stratum": idx + 1,
                "distance_min": float(quantiles[idx]),
                "distance_max": float(quantiles[idx + 1]),
                "n": int(mask.sum()),
                "event_rate": float(event.mean()),
                "event_tau_median": float(np.median(subset["surv_time"][event]))
                if event.any()
                else None,
                "nll_mean": metrics["nll_mean"],
                "brier_known_times": metrics["brier_known_times"],
                "mean_effective_support": metrics["mean_effective_support"],
            }
        )
    return result


def jax_nll_values(bins, logit0, logits, targets):
    nll_fn = create_perbin_nll_fn(jnp.asarray(bins))
    output = nll_fn(
        jnp.asarray(logit0),
        jnp.asarray(logits),
        jnp.asarray(targets["surv_is_event"]),
        jnp.asarray(targets["surv_time"]),
        jnp.asarray(targets["surv_censor_time"]),
    )
    return np.asarray(output)


def train_seed(
    seed: int,
    train_dataset: FrozenDict,
    val_targets: dict[str, np.ndarray],
    sampler_cfg: FrozenDict,
    bins: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    num_bins = len(bins) - 1
    pcs_model = make_official_pcs_module(
        num_bins, args.hidden_dim, args.rep_dim, args.k_basis, args.num_basis_sets
    )
    geometric_model = GeometricHazardBaseline(num_bins, args.hidden_dim, args.rep_dim)
    example_obs = jnp.asarray(val_targets["observations"][: args.batch_size])
    example_goal = jnp.asarray(val_targets["value_goals"][: args.batch_size])
    init_rng = jax.random.PRNGKey(seed)
    pcs_rng, geometric_rng, sampler_rng = jax.random.split(init_rng, 3)
    pcs_state = make_states(
        pcs_model, pcs_rng, example_obs, example_goal, args.learning_rate
    )
    geometric_state = make_states(
        geometric_model, geometric_rng, example_obs, example_goal, args.learning_rate
    )
    nll_fn = create_perbin_nll_fn(jnp.asarray(bins))
    pcs_update = make_update(nll_fn)
    geometric_update = make_update(nll_fn)
    pool_size = dataset_size_jax(train_dataset)
    trace = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch, sampler_rng = HGCDataset_sample(
            sampler_rng,
            train_dataset,
            args.batch_size,
            pool_size,
            sampler_cfg,
        )
        pcs_state, pcs_loss = pcs_update(pcs_state, batch)
        geometric_state, geometric_loss = geometric_update(geometric_state, batch)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            jax.block_until_ready((pcs_loss, geometric_loss))
            row = {
                "step": step,
                "pcs_train_nll": float(pcs_loss),
                "geometric_train_nll": float(geometric_loss),
                "elapsed_seconds": time.perf_counter() - start,
            }
            trace.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)

    training_seconds = time.perf_counter() - start
    pcs_logit0, pcs_logits = predict(
        pcs_state,
        val_targets["observations"],
        val_targets["value_goals"],
        args.predict_batch_size,
    )
    geometric_logit0, geometric_logits = predict(
        geometric_state,
        val_targets["observations"],
        val_targets["value_goals"],
        args.predict_batch_size,
    )
    pcs_metrics, pcs_nll = distribution_metrics(
        bins, pcs_logit0, pcs_logits, val_targets
    )
    geometric_metrics, geometric_nll = distribution_metrics(
        bins, geometric_logit0, geometric_logits, val_targets
    )
    pcs_jax_nll = jax_nll_values(bins, pcs_logit0, pcs_logits, val_targets)
    geometric_jax_nll = jax_nll_values(
        bins, geometric_logit0, geometric_logits, val_targets
    )

    shuffle_rng = np.random.default_rng(10_000 + seed)
    permutation = shuffle_rng.permutation(len(val_targets["surv_time"]))
    shuffled_targets = dict(val_targets)
    for key in ["surv_is_event", "surv_time", "surv_censor_time"]:
        shuffled_targets[key] = val_targets[key][permutation]
    shuffled_nll = nll_numpy(
        bins,
        pcs_logit0,
        pcs_logits,
        shuffled_targets["surv_is_event"],
        shuffled_targets["surv_time"],
        shuffled_targets["surv_censor_time"],
    )

    raw_path = output_dir / f"claim1_seed{seed}_predictions.npz"
    np.savez_compressed(
        raw_path,
        bins=bins,
        observations=val_targets["observations"],
        value_goals=val_targets["value_goals"],
        surv_is_event=val_targets["surv_is_event"],
        surv_time=val_targets["surv_time"],
        surv_censor_time=val_targets["surv_censor_time"],
        pcs_logit0=pcs_logit0,
        pcs_logits=pcs_logits,
        geometric_logit0=geometric_logit0,
        geometric_logits=geometric_logits,
    )
    trace_path = output_dir / f"claim1_seed{seed}_training_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    result = {
        "seed": seed,
        "training_seconds": training_seconds,
        "pcs_parameter_count": tree_parameter_count(pcs_state.params),
        "geometric_parameter_count": tree_parameter_count(geometric_state.params),
        "pcs": pcs_metrics,
        "geometric": geometric_metrics,
        "paired_nll_improvement_geometric_minus_pcs": float(
            np.mean(geometric_nll - pcs_nll)
        ),
        "paired_nll_improvement_se": float(
            np.std(geometric_nll - pcs_nll, ddof=1) / math.sqrt(len(pcs_nll))
        ),
        "numpy_vs_jax_nll_max_abs_error": {
            "pcs": float(np.max(np.abs(pcs_nll - pcs_jax_nll))),
            "geometric": float(np.max(np.abs(geometric_nll - geometric_jax_nll))),
        },
        "shuffled_target_negative_control": {
            "shuffled_nll_mean": float(shuffled_nll.mean()),
            "original_nll_mean": float(pcs_nll.mean()),
            "shuffled_minus_original": float(shuffled_nll.mean() - pcs_nll.mean()),
        },
        "distance_strata": distance_strata(bins, pcs_logit0, pcs_logits, val_targets),
        "raw_predictions": str(raw_path),
        "raw_predictions_sha256": sha256_file(raw_path),
        "training_trace": str(trace_path),
        "training_trace_sha256": sha256_file(trace_path),
    }
    return result


def aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        ("pcs_nll", lambda item: item["pcs"]["nll_mean"]),
        ("geometric_nll", lambda item: item["geometric"]["nll_mean"]),
        (
            "nll_improvement",
            lambda item: item["paired_nll_improvement_geometric_minus_pcs"],
        ),
        ("pcs_brier", lambda item: item["pcs"]["brier_known_times"]),
        ("geometric_brier", lambda item: item["geometric"]["brier_known_times"]),
        ("pcs_effective_support", lambda item: item["pcs"]["mean_effective_support"]),
        (
            "shuffled_nll_penalty",
            lambda item: item["shuffled_target_negative_control"][
                "shuffled_minus_original"
            ],
        ),
    ]
    output = {}
    for name, getter in fields:
        values = np.asarray([getter(item) for item in seed_results], dtype=np.float64)
        output[name] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/ogbench"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/claim1_ogbench")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=32768)
    parser.add_argument("--eval-chunk-size", type=int, default=8192)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--num-log-bins", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rep-dim", type=int, default=32)
    parser.add_argument("--k-basis", type=int, default=32)
    parser.add_argument("--num-basis-sets", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / f"{DATASET_NAME}.npz"
    val_path = args.data_dir / f"{DATASET_NAME}-val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Download {DATASET_URL_ROOT}/{DATASET_NAME}.npz and its -val split into {args.data_dir}"
        )

    print("Loading official OGBench train/validation files...", flush=True)
    train_host, val_host = ogbench.make_env_and_datasets(
        DATASET_NAME,
        dataset_path=str(train_path),
        compact_dataset=True,
        dataset_only=True,
    )
    train_dataset = prepare_hgc_dataset_for_jax(train_host, do_terminals=True)
    val_dataset = prepare_hgc_dataset_for_jax(val_host, do_terminals=True)
    sampler_cfg = make_sampler_config(args.horizon)
    bins_jax, _ = create_log_bins(args.horizon, args.num_log_bins)
    bins = np.asarray(bins_jax, dtype=np.int32)
    val_targets = sample_validation_tuples(
        val_dataset,
        sampler_cfg,
        args.eval_size,
        args.eval_chunk_size,
        seed=20260719,
    )
    event = val_targets["surv_is_event"] > 0.5

    seed_results = []
    total_start = time.perf_counter()
    for seed in args.seeds:
        seed_results.append(
            train_seed(
                seed,
                train_dataset,
                val_targets,
                sampler_cfg,
                bins,
                args,
                args.output_dir,
            )
        )
    total_seconds = time.perf_counter() - total_start

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    summary = {
        "claim": 1,
        "claim_text": "SVL reframes goal-conditioned RL as survival learning by modeling time-to-goal as a probability distribution.",
        "assessment_scope": "Direct critic/distribution evidence only; no actor training and no C3 policy-performance claim.",
        "command": " ".join(sys.argv),
        "environment_variables": {
            key: os.environ.get(key)
            for key in [
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "physical_ram_gb": 16,
            "max_rss_raw_platform_units": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "cost_usd": 0.0,
        "total_runtime_seconds": total_seconds,
        "package_versions": package_versions(),
        "revisions": {
            "reproduction_git_head_before_changes": git_head,
            "hsvl_official": UPSTREAM_HSVL_REVISION,
            "ogbench_source_audited": UPSTREAM_OGBENCH_REVISION,
        },
        "dataset": {
            "name": DATASET_NAME,
            "train_url": f"{DATASET_URL_ROOT}/{DATASET_NAME}.npz",
            "validation_url": f"{DATASET_URL_ROOT}/{DATASET_NAME}-val.npz",
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(val_path),
            "train_file_bytes": train_path.stat().st_size,
            "validation_file_bytes": val_path.stat().st_size,
            "train_rows": int(len(train_host["observations"])),
            "train_valid_pool": dataset_size_jax(train_dataset),
            "validation_rows": int(len(val_host["observations"])),
            "validation_valid_pool": dataset_size_jax(val_dataset),
            "validation_tuples_sampled": args.eval_size,
            "validation_event_rate": float(event.mean()),
            "validation_censor_rate": float((~event).mean()),
            "event_tau_quantiles": np.quantile(
                val_targets["surv_time"][event], [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
            ).tolist(),
            "censor_time_quantiles": np.quantile(
                val_targets["surv_censor_time"][~event],
                [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
            ).tolist(),
            "sampler": dict(sampler_cfg),
        },
        "architecture_and_scale": {
            "actual_bins": bins.tolist(),
            "actual_num_bins": len(bins) - 1,
            "requested_num_log_bins": args.num_log_bins,
            "horizon": args.horizon,
            "hidden_dims": [args.hidden_dim] * 3,
            "rep_dim": args.rep_dim,
            "k_basis": args.k_basis,
            "num_basis_sets": args.num_basis_sets,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "released_paper_defaults_for_disclosure": {
                "maze_steps": 1_000_000,
                "maze_batch_size": 1024,
                "hidden_dims": [512, 512, 512],
                "rep_dim": 256,
                "k_basis": 256,
                "num_basis_sets": 16,
                "paper_table_num_bins": 500,
                "released_config_num_log_bins": 800,
                "horizon": 10_000,
                "ensemble": True,
            },
            "deviations": [
                "PointMaze-large is a genuine OGBench navigation dataset but is not a Table 1 HSVL headline task.",
                "Actors and policy evaluation are omitted because Claim 1 concerns distribution modeling, not C3 performance.",
                "One survival head replaces the released twin ensemble.",
                "Critic width, representation, basis library, bin count, horizon, batch, and optimization steps are reduced as recorded above.",
                "Validation uses a fixed 32,768-tuple sample from the full disjoint 100,000-valid-transition pool by default.",
            ],
        },
        "seed_results": seed_results,
        "aggregate": aggregate_seed_metrics(seed_results),
    }
    summary_path = args.output_dir / "claim1_ogbench_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(summary_path), "sha256": sha256_file(summary_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
