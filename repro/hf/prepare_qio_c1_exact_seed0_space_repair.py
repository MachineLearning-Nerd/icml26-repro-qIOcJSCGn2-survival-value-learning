#!/usr/bin/env python3
"""Prepare, and only with explicit approval publish, a C1-only Space overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
SPACE = "DineshAI/qIOcJSCGn2"
PARENT = "757e63d6377d6a3789657d75e354306c95d6e43f"
PAGE_PATHS = (
    "pages/claim-1-time-to-goal-distribution/page.md",
    "pages/executive-summary/page.md",
    "pages/conclusion/page.md",
)
C2_PATH = "pages/claim-2-survival-identity/page.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not ({".cache", ".git"} & set(path.relative_to(root).parts))
    }


def append_cell(source: Path, destination: Path, title: str, body: str) -> None:
    original = source.read_text()
    cell_id = "cell_qio_c1_exact_seed0_terminal_v1_" + destination.parent.name.replace("-", "_")
    cell = f"\n\n---\n<!-- trackio-cell\n{{\"type\": \"markdown\", \"id\": \"{cell_id}\", \"title\": {json.dumps(title)}}}\n-->\n{body.rstrip()}\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(original.rstrip() + cell)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True, help="download of the pinned existing Space parent")
    parser.add_argument("--verified-import", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--override-contract-publication-block", action="store_true")
    args = parser.parse_args()
    parent = args.parent_dir.resolve()
    imported = args.verified_import.resolve()
    output = args.output_dir.resolve()
    required = [
        "completion.json", "independent_audit.json", "policy_summary.json",
        "twin_survival_metrics.json", "negative_controls.json", "terminal_import_receipt.json",
    ]
    if any(not (imported / name).is_file() for name in required):
        raise ValueError("verified import is partial")
    if any(not (parent / path).is_file() for path in (*PAGE_PATHS, C2_PATH)):
        raise ValueError("parent directory is not the expected existing Space layout")
    completion = json.loads((imported / "completion.json").read_text())
    audit = json.loads((imported / "independent_audit.json").read_text())
    import_receipt = json.loads((imported / "terminal_import_receipt.json").read_text())
    if completion.get("status") != "COMPLETED" or completion.get("completed_updates") != 1_000_000:
        raise ValueError("terminal exact-update completion gate failed")
    if import_receipt.get("space_parent_sha") != PARENT:
        raise ValueError("terminal import is not bound to the pinned Space parent")
    outcome = "supports the preregistered C1 integrity gates" if completion.get("c1_support_candidate") is True else "does not support every preregistered C1 integrity gate"
    interpretation = completion.get("scientific_interpretation", audit.get("scientific_interpretation"))
    evidence = {
        "scope": "C1_only_single_seed_exact_scale_not_C3_not_official_score",
        "space_parent_sha": PARENT,
        "completion_sha256": sha256(imported / "completion.json"),
        "terminal_import_receipt_sha256": sha256(imported / "terminal_import_receipt.json"),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "c1_support_candidate": completion["c1_support_candidate"],
        "scientific_interpretation": interpretation,
        "policy_episode_count": completion["policy_episode_count"],
        "heldout_tuple_count": completion["heldout_tuple_count"],
        "audit_sha256": sha256(imported / "independent_audit.json"),
        "policy_summary_sha256": sha256(imported / "policy_summary.json"),
        "twin_survival_metrics_sha256": sha256(imported / "twin_survival_metrics.json"),
        "negative_controls_sha256": sha256(imported / "negative_controls.json"),
    }
    body = (
        "## Exact-scale seed-0 terminal addendum\n\n"
        f"The frozen update-100 state was resumed to exactly update 1,000,000 and reloaded before evaluation. "
        f"The returned single-seed evidence {outcome}. The independent interpretation is: `{interpretation}`.\n\n"
        f"This adds 50 deterministic policy episodes and 32,768 held-out twin-survival tuples. It is **C1 only**, "
        "is not a C3/Table-1 benchmark, does not erase the earlier bounded three-seed result, and does not assert a new official score. "
        f"Terminal completion SHA-256: `{evidence['completion_sha256']}`."
    )
    if output.exists():
        raise FileExistsError("repair output already exists; use a new immutable directory")
    for relative in PAGE_PATHS:
        append_cell(parent / relative, output / relative, "Exact-scale seed-0 terminal addendum", body)
    evidence_dir = output / "evidence/qio-c1-exact-seed0"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "terminal-summary.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    shutil.copy2(imported / "terminal_import_receipt.json", evidence_dir / "terminal-import-receipt.json")
    manifest = {
        "schema_version": 1,
        "repo_id": SPACE,
        "expected_parent_sha": PARENT,
        "existing_space_only": True,
        "c2_parent_sha256": sha256(parent / C2_PATH),
        "c2_uploaded": False,
        "allowlisted_paths": sorted(inventory(output)),
        "overlay_inventory": inventory(output),
    }
    (output / "repair-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.publish:
        print(json.dumps({"status": "PREPARED_NOT_PUBLISHED", "output": str(output), "manifest": manifest}, indent=2, sort_keys=True))
        return
    if not args.override_contract_publication_block:
        raise PermissionError("the frozen execution contract blocks publication; explicit override is required")
    if os.environ.get("QIO_C1_SPACE_REPAIR_APPROVED") != "publish-existing-space-only":
        raise PermissionError("QIO_C1_SPACE_REPAIR_APPROVED is absent or incorrect")
    if os.environ.get("QIO_C1_SPACE_PARENT_SHA") != PARENT:
        raise PermissionError("QIO_C1_SPACE_PARENT_SHA does not bind the approved parent")
    api = HfApi()
    if api.space_info(SPACE).sha != PARENT:
        raise ValueError("Space parent drifted at publish boundary")
    with tempfile.TemporaryDirectory(prefix="qio-space-parent-") as temporary:
        canonical_parent = Path(snapshot_download(repo_id=SPACE, repo_type="space", revision=PARENT, local_dir=temporary))
        if inventory(canonical_parent) != inventory(parent):
            raise ValueError("supplied parent directory is not the pinned existing Space parent")
    commit = api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=str(output), commit_message="Add verified qIO C1 exact seed-0 terminal evidence")
    new_sha = commit.oid
    with tempfile.TemporaryDirectory(prefix="qio-space-readback-") as temporary:
        current = Path(snapshot_download(repo_id=SPACE, repo_type="space", revision=new_sha, local_dir=temporary))
        parent_inventory = inventory(parent)
        current_inventory = inventory(current)
        allowed = set(inventory(output))
        for relative, digest in parent_inventory.items():
            if relative not in allowed and current_inventory.get(relative) != digest:
                raise RuntimeError(f"protected parent file drifted after publication: {relative}")
        if current_inventory.get(C2_PATH) != sha256(parent / C2_PATH):
            raise RuntimeError("C2 was not preserved byte-for-byte")
        for relative, digest in inventory(output).items():
            if current_inventory.get(relative) != digest:
                raise RuntimeError(f"published overlay readback mismatch: {relative}")
    print(json.dumps({"status": "PUBLISHED_AND_READ_BACK", "parent_sha": PARENT, "new_sha": new_sha, "c2_sha256": sha256(parent / C2_PATH)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
