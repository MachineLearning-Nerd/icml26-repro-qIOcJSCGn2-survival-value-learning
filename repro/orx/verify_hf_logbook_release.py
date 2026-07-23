#!/usr/bin/env python3
"""Fail-closed verifier for the human-approved two-file HF logbook release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "repro" / "configs" / "hf_logbook_release.json"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_files(candidate_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(candidate_root))
        for path in candidate_root.rglob("*")
        if path.is_file()
    )


def validate(
    config: dict[str, object],
    contract: dict[str, object],
    *,
    candidate_logbook: dict[str, object] | None = None,
    allowlist: list[str] | None = None,
    protected: dict[str, str] | None = None,
) -> dict[str, object]:
    candidate_root = ROOT / config["candidate_root"]
    logbook = candidate_logbook or json.loads(
        (candidate_root / "logbook.json").read_text()
    )
    files = candidate_files(candidate_root)
    expected_allowlist = allowlist or config["upload_allowlist"]
    protected_hashes = protected or config["protected_sha256"]
    metadata = config["base_logbook_metadata"]
    old_children = config["base_root_children"]
    children = logbook["root"]["children"]
    new_node = config["new_node"]
    page = (candidate_root / new_node["file"]).read_text()

    checks = {
        "schema_version": logbook["schema_version"] == metadata["schema_version"],
        "title": logbook["title"] == metadata["title"],
        "emoji": logbook["emoji"] == metadata["emoji"],
        "space_id": logbook["space_id"] == metadata["space_id"],
        "paper": logbook["paper"] == metadata["paper"],
        "tags": logbook["tags"] == metadata["tags"],
        "root_slug": logbook["root"]["slug"] == metadata["root_slug"],
        "root_title": logbook["root"]["title"] == metadata["root_title"],
        "root_file": logbook["root"]["file"] == metadata["root_file"],
        "existing_nodes_preserved": children[: len(old_children)] == old_children,
        "only_one_node_added": len(children) == len(old_children) + 1,
        "new_node_exact": children[-1] == new_node,
        "candidate_files_equal_allowlist": files == expected_allowlist,
        "configured_allowlist_exact": sorted(config["upload_allowlist"])
        == sorted(["logbook.json", "pages/three-interpretation-retry/page.md"]),
        "no_deletes": config["delete_allowlist"] == [],
        "base_head_prechecked": config["head_equal_to_base_before_candidate"] is True,
        "base_revision_matches_contract": config["base_revision"]
        == contract["live_verdict"]["judged_sha"],
        "base_manifest_has_all_existing_nodes": all(
            child["file"] in config["judged_file_sha256"] for child in old_children
        ),
        "base_file_set_nonempty": len(config["judged_file_sha256"]) == 22,
        "protected_c2_page": protected_hashes[
            "pages/claim-2-survival-identity/page.md"
        ]
        == contract["protected_hf_files"]["pages/claim-2-survival-identity/page.md"],
        "protected_c2_verifier": protected_hashes["repro/src/verify_survival.py"]
        == contract["protected_hf_files"]["repro/src/verify_survival.py"],
        "protected_base_logbook": protected_hashes["logbook.json"]
        == contract["protected_hf_files"]["logbook.json"],
        "page_live_score_only": "live judged state remains **4/12**" in page,
        "page_disclaims_unjudged_score": "do not estimate an unjudged score" in page,
        "page_preserves_c2": "**C2 remains verified.**" in page,
        "page_does_not_upgrade_c4_c6": all(
            marker in page
            for marker in (
                "**C4 remains inconclusive.**",
                "**C5 remains inconclusive.**",
                "**C6 remains inconclusive.**",
            )
        ),
        "page_records_blocked_scope": "faithful execution requires forbidden GPU compute"
        in page,
        "candidate_score_is_null": config["candidate_score"] is None,
        "user_publication_authorization": config["publication_authorized_by_user"]
        is True,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "candidate_files": files,
        "effective_candidate_file_count": len(config["judged_file_sha256"]) + 1,
        "existing_file_count": len(config["judged_file_sha256"]),
        "new_file_count": 1,
    }


def verify_hf_logbook_release(contract: dict[str, object], c2_pass: bool) -> int:
    config = json.loads(CONFIG.read_text())
    if config["schema_version"] != 1:
        raise RuntimeError("HF release config schema must be 1")

    from evaluate_table_arithmetic import evaluate_table_arithmetic

    table_returncode = evaluate_table_arithmetic(contract, c2_pass)
    primary = validate(config, contract)

    removed_child = json.loads(
        (ROOT / config["candidate_root"] / "logbook.json").read_text()
    )
    removed_child["root"]["children"] = removed_child["root"]["children"][1:]
    removed_child_rejected = not validate(
        config, contract, candidate_logbook=removed_child
    )["pass"]

    expanded_allowlist_rejected = not validate(
        config,
        contract,
        allowlist=[*config["upload_allowlist"], "README.md"],
    )["pass"]

    tampered_protected = dict(config["protected_sha256"])
    old = tampered_protected["pages/claim-2-survival-identity/page.md"]
    tampered_protected["pages/claim-2-survival-identity/page.md"] = (
        ("0" if old[0] != "0" else "1") + old[1:]
    )
    tampered_c2_rejected = not validate(
        config, contract, protected=tampered_protected
    )["pass"]

    negative_controls = {
        "removed_existing_root_child_rejected": removed_child_rejected,
        "expanded_upload_allowlist_rejected": expanded_allowlist_rejected,
        "tampered_protected_c2_hash_rejected": tampered_c2_rejected,
    }
    release_pass = (
        c2_pass
        and table_returncode == 0
        and primary["pass"]
        and all(negative_controls.values())
    )
    candidate_root = ROOT / config["candidate_root"]
    evidence = {
        "schema_version": 1,
        "release_kind": config["release_kind"],
        "space_id": config["space_id"],
        "base_revision": config["base_revision"],
        "upload_allowlist": config["upload_allowlist"],
        "delete_allowlist": config["delete_allowlist"],
        "candidate_file_sha256": {
            relative: sha256(candidate_root / relative)
            for relative in config["upload_allowlist"]
        },
        "judged_file_sha256": config["judged_file_sha256"],
        "protected_c2_regression_pass": c2_pass,
        "table_primary_independent_controls_pass": table_returncode == 0,
        "subset_and_claim_tree_check": primary,
        "negative_controls": negative_controls,
        "claim_recommendations": config["claim_recommendations"],
        "candidate_score": None,
        "release_candidate_pass": release_pass,
        "hf_upload_performed_by_this_run": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACTS / "hf_logbook_release_candidate.json"
    text_path = ARTIFACTS / "hf_logbook_release_candidate.txt"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    text_lines = [
        f"release_candidate: {'PASS' if release_pass else 'FAIL'}",
        f"space_id: {config['space_id']}",
        f"base_revision: {config['base_revision']}",
        f"protected_c2_regression: {'PASS' if c2_pass else 'FAIL'}",
        f"table_primary_independent_controls: {'PASS' if table_returncode == 0 else 'FAIL'}",
        f"judged_subset_and_claim_tree: {'PASS' if primary['pass'] else 'FAIL'}",
        f"negative_controls: {'PASS' if all(negative_controls.values()) else 'FAIL'}",
        "candidate_score: not estimated",
        "upload_allowlist:",
        *[f"  - {path}" for path in config["upload_allowlist"]],
        "delete_allowlist: empty",
        "hf_upload_performed_by_this_run: false",
    ]
    text_path.write_text("\n".join(text_lines) + "\n")

    eval_lines = [
        "# HF logbook text-only release candidate",
        "",
        f"- Release gate: **{'PASS' if release_pass else 'FAIL'}**",
        f"- Base/current Space revision: `{config['base_revision']}`",
        f"- Protected C2 regression: **{'PASS' if c2_pass else 'FAIL'}**",
        f"- Table primary/auditor/controls: **{'PASS' if table_returncode == 0 else 'FAIL'}**",
        f"- Judged file/claim-tree subset: **{'PASS' if primary['pass'] else 'FAIL'}**",
        f"- Negative controls: **{'PASS' if all(negative_controls.values()) else 'FAIL'}**",
        "- Candidate score: **not estimated**",
        "- Upload performed by this run: **no**",
        "",
        "## Exact upload allowlist",
        "",
        *[f"- `{path}`" for path in config["upload_allowlist"]],
        "",
        "Delete allowlist is empty. Every existing root child is preserved verbatim,",
        "the protected C2 page and verifier hashes remain pinned, and the only new",
        "claim-tree node is the text-only three-interpretation retry page.",
        "",
        "## Evidence",
        "",
        "- `.openresearch/artifacts/hf_logbook_release_candidate.json`",
        "- `.openresearch/artifacts/hf_logbook_release_candidate.txt`",
        "- `.openresearch/artifacts/table_arithmetic_manifest.json`",
        "",
    ]
    (ROOT / "EVAL.md").write_text("\n".join(eval_lines))
    print(text_path.read_text(), end="")
    print(json_path.read_text(), end="")
    print(f"HF_LOGBOOK_RELEASE_CANDIDATE={'PASS' if release_pass else 'FAIL'}")
    print("CANDIDATE_SCORE=NOT_ESTIMATED")
    return 0 if release_pass else 1
