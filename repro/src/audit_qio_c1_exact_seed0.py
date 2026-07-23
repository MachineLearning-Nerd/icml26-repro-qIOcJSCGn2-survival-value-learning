#!/usr/bin/env python3
"""Independent NumPy audit for the qIO exact-scale C1 raw survival return.

This module deliberately imports neither JAX nor HSVL. Scientific controls can
fail without making the artifact invalid; such outcomes are reported as
negative or inconclusive C1 evidence and are never discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RAW_NAME = "twin_survival_raw.npz"
METRICS_NAME = "twin_survival_metrics.json"
CONTROLS_NAME = "negative_controls.json"
AUDIT_NAME = "independent_audit.json"
EXPECTED_TUPLES = 32_768
SHUFFLE_SEED = 20_260_721
NORMALIZATION_LIMIT = 1e-10
MONOTONICITY_LIMIT = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def reconstruct(logit0: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
    p0 = sigmoid(logit0)
    hazards = sigmoid(logits)
    one_minus = 1.0 - hazards
    survival_before_hazard = np.concatenate(
        [
            np.ones((len(logits), 1), dtype=np.float64),
            np.cumprod(one_minus, axis=1)[:, :-1],
        ],
        axis=1,
    )
    survival_start = (1.0 - p0)[:, None] * survival_before_hazard
    interval_mass = survival_start * hazards
    survival_end = (1.0 - p0)[:, None] * np.cumprod(one_minus, axis=1)
    tail_mass = survival_end[:, -1]
    probabilities = np.concatenate(
        [p0[:, None], interval_mass, tail_mass[:, None]], axis=1
    )
    return {
        "p0": p0,
        "hazards": hazards,
        "interval_mass": interval_mass,
        "survival_end": survival_end,
        "probabilities": probabilities,
    }


def mean_distribution(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    probabilities = 0.5 * (first["probabilities"] + second["probabilities"])
    interval_mass = probabilities[:, 1:-1]
    survival_end = np.flip(
        np.cumsum(np.flip(probabilities[:, 2:], axis=1), axis=1), axis=1
    )
    return {
        "p0": probabilities[:, 0],
        "interval_mass": interval_mass,
        "survival_end": survival_end,
        "probabilities": probabilities,
    }


def nll_from_distribution(
    bins: np.ndarray,
    distribution: dict[str, np.ndarray],
    is_event: np.ndarray,
    tau: np.ndarray,
    censor: np.ndarray,
) -> np.ndarray:
    p0 = distribution["p0"]
    interval_mass = distribution["interval_mass"]
    survival_end = distribution["survival_end"]
    rows = np.arange(len(p0))
    tau_bin = np.clip(
        np.searchsorted(bins, tau, side="right") - 1,
        0,
        interval_mass.shape[1] - 1,
    )
    event_probability = np.where(tau == 0, p0, interval_mass[rows, tau_bin])
    censor_edge = np.clip(
        np.searchsorted(bins, censor, side="left"), 0, interval_mass.shape[1]
    )
    censor_probability = 1.0 - p0
    positive = censor_edge > 0
    censor_probability = censor_probability.copy()
    censor_probability[positive] = survival_end[
        rows[positive], censor_edge[positive] - 1
    ]
    probability = np.where(is_event > 0.5, event_probability, censor_probability)
    return -np.log(np.maximum(probability, np.finfo(np.float64).tiny))


def calibration_curve(
    bins: np.ndarray,
    survival_end: np.ndarray,
    is_event: np.ndarray,
    tau: np.ndarray,
    censor: np.ndarray,
    points: int = 20,
) -> list[dict[str, Any]]:
    indices = np.unique(
        np.rint(np.linspace(0, survival_end.shape[1] - 1, points)).astype(np.int64)
    )
    rows = []
    for index in indices:
        edge = int(bins[index + 1])
        known = (is_event > 0.5) | (censor >= edge)
        observed = np.where(is_event > 0.5, tau >= edge, True)
        rows.append(
            {
                "bin_index": int(index),
                "time_edge": edge,
                "known_count": int(known.sum()),
                "predicted_survival_mean": float(survival_end[known, index].mean()),
                "observed_survival_mean": float(observed[known].mean()),
                "absolute_gap": float(
                    abs(survival_end[known, index].mean() - observed[known].mean())
                ),
            }
        )
    return rows


def distribution_metrics(
    bins: np.ndarray,
    distribution: dict[str, np.ndarray],
    is_event: np.ndarray,
    tau: np.ndarray,
    censor: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = distribution["probabilities"]
    survival_end = distribution["survival_end"]
    nll = nll_from_distribution(bins, distribution, is_event, tau, censor)
    edges = bins[1:]
    known = (is_event[:, None] > 0.5) | (edges[None, :] <= censor[:, None])
    observed = np.where(is_event[:, None] > 0.5, tau[:, None] >= edges[None, :], True)
    error = survival_end - observed
    metrics = {
        "censored_negative_log_likelihood": float(nll.mean()),
        "censored_negative_log_likelihood_se": float(
            nll.std(ddof=1) / math.sqrt(len(nll))
        ),
        "known_time_brier_score": float(np.square(error)[known].mean()),
        "survival_mae": float(np.abs(error)[known].mean()),
        "calibration_curve": calibration_curve(
            bins, survival_end, is_event, tau, censor
        ),
        "distribution_normalization_max_abs_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
        "survival_monotonicity_max_increase": float(
            np.maximum(np.diff(survival_end, axis=1), 0.0).max(initial=0.0)
        ),
        "tail_mass_mean": float(probabilities[:, -1].mean()),
        "tail_mass_max": float(probabilities[:, -1].max()),
        "minimum_probability": float(probabilities.min()),
    }
    return metrics, nll


def require_raw(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "bins",
        "observation_indices",
        "value_goal_indices",
        "observations",
        "value_goals",
        "surv_is_event",
        "surv_time",
        "surv_censor_time",
        "surv_valid",
        "head1_logit0",
        "head1_logits",
        "head2_logit0",
        "head2_logits",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(
                f"raw twin-survival keys differ: {sorted(set(archive.files) ^ expected)}"
            )
        raw = {key: archive[key] for key in expected}
    n = len(raw["surv_valid"])
    if n != EXPECTED_TUPLES:
        raise ValueError(f"expected {EXPECTED_TUPLES} raw tuples, found {n}")
    bins = raw["bins"]
    if bins.ndim != 1 or len(bins) < 2 or not np.all(np.diff(bins) > 0):
        raise ValueError("survival bins must be a strictly increasing vector")
    num_bins = len(bins) - 1
    vector_keys = (
        "observation_indices",
        "value_goal_indices",
        "surv_is_event",
        "surv_time",
        "surv_censor_time",
        "surv_valid",
        "head1_logit0",
        "head2_logit0",
    )
    if any(raw[key].shape != (n,) for key in vector_keys):
        raise ValueError("one or more raw vector shapes are invalid")
    if raw["observations"].shape[0] != n or raw["value_goals"].shape != raw["observations"].shape:
        raise ValueError("raw observation/goal shapes are invalid")
    if raw["head1_logits"].shape != (n, num_bins) or raw["head2_logits"].shape != (n, num_bins):
        raise ValueError("raw twin logit shapes do not match bins")
    if not all(np.isfinite(value).all() for value in raw.values()):
        raise ValueError("raw twin-survival artifact contains non-finite values")
    return raw


def audit_raw(raw_path: Path) -> dict[str, Any]:
    raw = require_raw(raw_path)
    valid = raw["surv_valid"] > 0.5
    if not valid.any():
        raise ValueError("held-out artifact has no valid survival tuples")
    bins = raw["bins"].astype(np.int64)
    event = raw["surv_is_event"][valid].astype(np.float64)
    tau = raw["surv_time"][valid].astype(np.int64)
    censor = raw["surv_censor_time"][valid].astype(np.int64)
    first = reconstruct(raw["head1_logit0"][valid], raw["head1_logits"][valid])
    second = reconstruct(raw["head2_logit0"][valid], raw["head2_logits"][valid])
    twin = mean_distribution(first, second)

    metrics: dict[str, Any] = {}
    per_nll: dict[str, np.ndarray] = {}
    for name, distribution in (("head1", first), ("head2", second), ("twin_mean", twin)):
        metrics[name], per_nll[name] = distribution_metrics(
            bins, distribution, event, tau, censor
        )
    metrics["twin_head_disagreement"] = {
        "mean_total_variation": float(
            (0.5 * np.abs(first["probabilities"] - second["probabilities"]).sum(axis=1)).mean()
        ),
        "mean_absolute_survival_difference": float(
            np.abs(first["survival_end"] - second["survival_end"]).mean()
        ),
    }

    rng = np.random.default_rng(SHUFFLE_SEED)
    permutation = rng.permutation(len(event))
    shuffled = nll_from_distribution(
        bins, twin, event[permutation], tau[permutation], censor[permutation]
    )
    reversed_first = reconstruct(
        raw["head1_logit0"][valid], raw["head1_logits"][valid, ::-1]
    )
    reversed_second = reconstruct(
        raw["head2_logit0"][valid], raw["head2_logits"][valid, ::-1]
    )
    reversed_twin = mean_distribution(reversed_first, reversed_second)
    reversed_nll = nll_from_distribution(bins, reversed_twin, event, tau, censor)
    original = per_nll["twin_mean"]
    controls = {
        "joint_target_shuffle": {
            "seed": SHUFFLE_SEED,
            "original_nll_mean": float(original.mean()),
            "control_nll_mean": float(shuffled.mean()),
            "penalty": float(shuffled.mean() - original.mean()),
            "required_positive": True,
            "passes": bool(shuffled.mean() > original.mean()),
        },
        "temporal_hazard_reversal": {
            "original_nll_mean": float(original.mean()),
            "control_nll_mean": float(reversed_nll.mean()),
            "penalty": float(reversed_nll.mean() - original.mean()),
            "required_positive": True,
            "passes": bool(reversed_nll.mean() > original.mean()),
        },
    }
    finite_metrics = all(
        math.isfinite(float(metrics[name][key]))
        for name in ("head1", "head2", "twin_mean")
        for key in (
            "censored_negative_log_likelihood",
            "known_time_brier_score",
            "survival_mae",
            "distribution_normalization_max_abs_error",
            "survival_monotonicity_max_increase",
            "tail_mass_mean",
        )
    )
    numeric_pass = all(
        metrics[name]["distribution_normalization_max_abs_error"] <= NORMALIZATION_LIMIT
        and metrics[name]["survival_monotonicity_max_increase"] <= MONOTONICITY_LIMIT
        for name in ("head1", "head2", "twin_mean")
    )
    support_candidate = bool(
        finite_metrics
        and numeric_pass
        and controls["joint_target_shuffle"]["passes"]
        and controls["temporal_hazard_reversal"]["passes"]
    )
    metrics_document = {
        "schema_version": 1,
        "scope": "C1_only_not_C3",
        "raw_sha256": sha256_file(raw_path),
        "tuple_count": int(len(valid)),
        "valid_tuple_count": int(valid.sum()),
        "invalid_tuple_count": int((~valid).sum()),
        "twin_mean_definition": "arithmetic_mean_of_head_probability_distributions",
        "per_head_and_twin_mean": metrics,
        "integrity_thresholds": {
            "all_values_finite": True,
            "distribution_normalization_max_abs_error_lte": NORMALIZATION_LIMIT,
            "survival_monotonicity_max_increase_lte": MONOTONICITY_LIMIT,
        },
        "finite_metrics": finite_metrics,
        "numeric_integrity_pass": numeric_pass,
        "calibration_claim_threshold": None,
    }
    controls_document = {
        "schema_version": 1,
        "scope": "C1_only_not_C3",
        "raw_sha256": sha256_file(raw_path),
        **controls,
        "report_regardless_of_outcome": True,
    }
    audit_document = {
        "schema_version": 1,
        "method": "independent_numpy_only_no_jax_no_hsvl_import",
        "scope": "C1_only_not_C3_not_score_guaranteed",
        "raw_sha256": sha256_file(raw_path),
        "tuple_count": int(len(valid)),
        "valid_tuple_count": int(valid.sum()),
        "numeric_integrity_pass": numeric_pass,
        "negative_controls_pass": bool(
            controls["joint_target_shuffle"]["passes"]
            and controls["temporal_hazard_reversal"]["passes"]
        ),
        "c1_support_candidate": support_candidate,
        "scientific_interpretation": (
            "C1_support_candidate"
            if support_candidate
            else "inconclusive_or_negative_C1_evidence_not_integrity_failure"
        ),
    }
    return {
        "metrics": metrics_document,
        "controls": controls_document,
        "audit": audit_document,
    }


def compare_json(expected: object, observed: object, label: str) -> None:
    if expected != observed:
        raise ValueError(f"independent recomputation differs from saved {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    raw_path = args.results_dir / RAW_NAME
    documents = audit_raw(raw_path)
    paths = {
        "metrics": args.results_dir / METRICS_NAME,
        "controls": args.results_dir / CONTROLS_NAME,
        "audit": args.results_dir / AUDIT_NAME,
    }
    if args.verify_only:
        for label, path in paths.items():
            compare_json(
                documents[label], json.loads(path.read_text(encoding="utf-8")), label
            )
    else:
        for label, path in paths.items():
            atomic_json(path, documents[label])
    print(
        json.dumps(
            {
                "status": "verified" if args.verify_only else "written",
                "raw_sha256": sha256_file(raw_path),
                "c1_support_candidate": documents["audit"]["c1_support_candidate"],
                "outputs": {
                    label: {"path": str(path), "sha256": sha256_file(path)}
                    for label, path in paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
