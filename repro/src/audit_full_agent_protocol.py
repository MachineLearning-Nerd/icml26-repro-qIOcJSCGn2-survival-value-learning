#!/usr/bin/env python3
"""Audit the preregistered full-agent gate without importing JAX.

This is a provenance and protocol audit, not a test suite.  It binds every
executed vendored source/configuration file to the official Git commit and
checks the official OGBench inputs plus the declared execution-gate scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PAPER_ROOT / "upstream"
DEFAULT_MANIFEST = PAPER_ROOT / "repro" / "configs" / "upstream_source_manifest.json"
DEFAULT_OUTPUT = PAPER_ROOT / "outputs" / "full_agent_protocol_audit.json"
EXPECTED_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"
EXPECTED_DATASET_HASHES = {
    "pointmaze-large-navigate-v0.npz": (
        "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61"
    ),
    "pointmaze-large-navigate-v0-val.npz": (
        "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4"
    ),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    """Return the Git SHA-1 blob object ID without invoking Git."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def vendored_file_set() -> set[str]:
    files: set[str] = set()
    for path in UPSTREAM_ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(UPSTREAM_ROOT).as_posix()
        if relative != "SOURCE_REVISION":
            files.add(relative)
    return files


def verify_upstream_source(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported upstream manifest schema")
    if manifest.get("commit") != EXPECTED_REVISION:
        raise ValueError("upstream manifest commit is not the preregistered revision")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("upstream manifest contains no file identities")

    revision_text = (UPSTREAM_ROOT / "SOURCE_REVISION").read_text()
    if f"Commit: {EXPECTED_REVISION}" not in revision_text:
        raise ValueError("SOURCE_REVISION does not name the preregistered commit")

    observed_files = vendored_file_set()
    expected_set = set(expected_files)
    if observed_files != expected_set:
        raise ValueError(
            "vendored source inventory mismatch: "
            f"missing={sorted(expected_set - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_set)}"
        )

    observed_blobs = {
        relative: git_blob_oid(UPSTREAM_ROOT / relative)
        for relative in sorted(expected_files)
    }
    mismatches = {
        relative: {"expected": expected_files[relative], "observed": observed}
        for relative, observed in observed_blobs.items()
        if observed != expected_files[relative]
    }
    if mismatches:
        raise ValueError(f"vendored source blob mismatch: {mismatches}")

    return {
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "tree": manifest["tree"],
        "file_count": len(observed_blobs),
        "git_blob_manifest_sha256": canonical_digest(observed_blobs),
        "manifest_sha256": sha256_file(manifest_path),
        "all_git_blobs_match": True,
        "intentionally_omitted_remote_files": manifest.get(
            "intentionally_omitted_remote_files", {}
        ),
    }


def verify_datasets(data_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, expected in EXPECTED_DATASET_HASHES.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"official dataset hash mismatch for {name}")
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": observed,
            "matches_official_download": True,
        }
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--data-dir", type=Path, default=PAPER_ROOT / "data" / "ogbench"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = verify_upstream_source(args.manifest)
    datasets = verify_datasets(args.data_dir)
    result = {
        "schema_version": 1,
        "status": "complete",
        "all_checks_pass": True,
        "scope": "preregistered_execution_gate_provenance_not_claim_evidence",
        "paper": {
            "title": "SVL: Goal-Conditioned Reinforcement Learning as Survival Learning",
            "arxiv": "2604.17551v2",
            "paper_protocol_anchors": {
                "maze_train_updates": 1_000_000,
                "maze_batch_size": 1024,
                "reported_seeds": 4,
                "reported_bins": 500,
                "source": "Section 5 and Appendix Table 3",
            },
            "released_evaluation_config": {
                "evaluation_episodes_per_task": 50,
                "source": "upstream/hsvl/config/main.yaml",
            },
        },
        "official_source": source,
        "official_datasets": datasets,
        "queued_gate": {
            "environment": "pointmaze-large-navigate-v0",
            "updates": 100,
            "seeds": 1,
            "policy_evaluation": False,
            "preserved_components": [
                "width-512 twin PCS critic",
                "width-512 low-level actor",
                "width-512 high-level actor",
                "256-dimensional goal representation",
                "256-by-16 temporal basis library",
                "requested 800 bins and horizon 10000",
                "batch size 1024",
            ],
            "interpretation": (
                "Execution, checkpoint, and exact-reload gate only; it cannot verify "
                "Claim 3 performance."
            ),
            "declared_protocol_difference": (
                "The released YAML requests 800 geometric edges (499 unique bins after "
                "integer de-duplication), while Appendix Table 3 reports K=500. The gate "
                "follows the released source exactly and records this distinction."
            ),
        },
        "remaining_claim3_gate": {
            "required": [
                "one Table 1 OGBench task",
                "1,000,000 training updates",
                "four independent seeds",
                "50 evaluation episodes per task",
                "a matched hierarchical TD baseline or an exact reported-result target",
            ],
            "status": "unexecuted",
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
