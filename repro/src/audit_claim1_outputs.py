#!/usr/bin/env python3
"""Independent NumPy-only audit of saved Claim 1 prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_predictions(path: Path) -> dict[str, float | str]:
    raw = np.load(path)
    bins = raw["bins"]
    logit0 = raw["pcs_logit0"].astype(np.float64)
    logits = raw["pcs_logits"].astype(np.float64)
    event = raw["surv_is_event"] > 0.5
    tau = raw["surv_time"].astype(np.int64)
    censor = raw["surv_censor_time"].astype(np.int64)

    p0 = np.exp(-np.logaddexp(0.0, -logit0))
    hazards = np.exp(-np.logaddexp(0.0, -logits))
    survival_end = (1.0 - p0)[:, None] * np.cumprod(1.0 - hazards, axis=1)
    survival_before = np.concatenate(
        [np.ones((len(logits), 1)), np.cumprod(1.0 - hazards, axis=1)[:, :-1]],
        axis=1,
    )
    interval_mass = (1.0 - p0)[:, None] * survival_before * hazards
    probability_sum = p0 + interval_mass.sum(axis=1) + survival_end[:, -1]

    logp0 = -np.logaddexp(0.0, -logit0)
    log1mp0 = -np.logaddexp(0.0, logit0)
    logh = -np.logaddexp(0.0, -logits)
    log1mh = -np.logaddexp(0.0, logits)
    prefix = np.cumsum(log1mh, axis=1)
    prefix_before = np.concatenate([np.zeros((len(logits), 1)), prefix[:, :-1]], axis=1)
    rows = np.arange(len(logits))
    event_bin = np.clip(
        np.searchsorted(bins, tau, side="right") - 1, 0, logits.shape[1] - 1
    )
    event_ll = np.where(
        tau == 0,
        logp0,
        log1mp0 + prefix_before[rows, event_bin] + logh[rows, event_bin],
    )
    censor_edge = np.clip(
        np.searchsorted(bins, censor, side="left"), 0, logits.shape[1]
    )
    censor_sum = np.zeros(len(logits))
    positive = censor_edge > 0
    censor_sum[positive] = prefix[rows[positive], censor_edge[positive] - 1]
    censor_ll = log1mp0 + censor_sum
    nll = -np.where(event, event_ll, censor_ll)

    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "n": len(logits),
        "pcs_nll_mean": float(nll.mean()),
        "probability_sum_max_abs_error": float(np.max(np.abs(probability_sum - 1.0))),
        "survival_monotonicity_max_increase": float(
            np.maximum(np.diff(survival_end, axis=1), 0.0).max(initial=0.0)
        ),
        "mean_temporal_hazard_std": float(np.std(hazards, axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("outputs/claim1_ogbench")
    )
    args = parser.parse_args()
    summary_path = args.results_dir / "claim1_ogbench_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audits = []
    for seed_result in summary["seed_results"]:
        audit = audit_predictions(
            args.results_dir / Path(seed_result["raw_predictions"]).name
        )
        audit["seed"] = seed_result["seed"]
        audit["summary_nll_abs_error"] = abs(
            audit["pcs_nll_mean"] - seed_result["pcs"]["nll_mean"]
        )
        audit["summary_sha_match"] = (
            audit["sha256"] == seed_result["raw_predictions_sha256"]
        )
        audits.append(audit)
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(args.results_dir.iterdir())
        if path.is_file() and path.name != "independent_audit.json"
    }
    repo_root = Path.cwd()
    source_paths = [
        Path("repro/src/claim1_ogbench_distribution.py"),
        Path("repro/src/audit_claim1_outputs.py"),
        Path("repro/tests/test_claim1_ogbench_distribution.py"),
        Path("upstream/SOURCE_REVISION"),
        Path("upstream/hsvl/utils/critics.py"),
        Path("upstream/hsvl/utils/datasets.py"),
        Path("upstream/hsvl/utils/survival.py"),
        Path("data/ogbench/pointmaze-large-navigate-v0.npz"),
        Path("data/ogbench/pointmaze-large-navigate-v0-val.npz"),
    ]
    source_manifest = {
        str(path): sha256_file(repo_root / path) for path in source_paths
    }
    output = {
        "method": "standalone NumPy-only audit; no imports from the experiment or released HSVL code",
        "summary_sha256": sha256_file(summary_path),
        "audits": audits,
        "all_checks_pass": all(
            item["summary_sha_match"]
            and item["summary_nll_abs_error"] < 1e-12
            and item["probability_sum_max_abs_error"] < 1e-12
            and item["survival_monotonicity_max_increase"] == 0.0
            for item in audits
        ),
        "artifact_manifest": manifest,
        "source_data_test_manifest": source_manifest,
    }
    output_path = args.results_dir / "independent_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output_path), "sha256": sha256_file(output_path), **output},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
