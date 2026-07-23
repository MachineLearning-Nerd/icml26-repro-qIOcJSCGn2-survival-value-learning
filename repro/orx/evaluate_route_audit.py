#!/usr/bin/env python3
"""Fail-closed source/protocol route audit; never scientific claim evidence."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "repro" / "configs" / "route_audit.json"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "scipy", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def evaluate_route_audit(contract: dict[str, object], c2_pass: bool) -> int:
    config = json.loads(CONFIG.read_text())
    if config.get("schema_version") != 1:
        raise RuntimeError("route audit schema_version must be 1")
    if config.get("scientific_evidence") is not False:
        raise RuntimeError("route audit must explicitly disclaim scientific evidence")

    claims_by_id = {row["id"]: row for row in contract["claims"]}
    claims = config["claims"]
    if not claims or any(claim not in claims_by_id for claim in claims):
        raise RuntimeError("route audit claims are not present in judge contract")
    expected_verdicts = config["current_live_verdicts"]
    verdicts_match = all(
        claims_by_id[claim]["current_verdict"] == expected_verdicts[claim]
        for claim in claims
    )

    source = config["paper_source"]
    source_matches_contract = (
        source["url"] == contract["source"]["primary_url"]
        and source["retrieved_on"] == contract["source"]["retrieved_on"]
        and source["sha256"] == contract["source"]["sha256"]
    )

    hash_checks = []
    for relative, expected in sorted(config["required_sha256"].items()):
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        hash_checks.append(
            {"path": relative, "expected": expected, "actual": actual, "pass": actual == expected}
        )

    pattern_checks = []
    for spec in config["pattern_checks"]:
        path = ROOT / spec["path"]
        text = path.read_text(errors="replace") if path.is_file() else ""
        observed = re.search(spec["pattern"], text, flags=re.MULTILINE) is not None
        passed = observed is bool(spec["must_exist"])
        pattern_checks.append({**spec, "observed": observed, "pass": passed})

    first_hash = next(iter(sorted(config["required_sha256"].items())))
    tampered_expected = ("0" if first_hash[1][0] != "0" else "1") + first_hash[1][1:]
    tampered_manifest_rejected = sha256(ROOT / first_hash[0]) != tampered_expected

    integrity_pass = (
        c2_pass
        and verdicts_match
        and source_matches_contract
        and all(row["pass"] for row in hash_checks)
        and all(row["pass"] for row in pattern_checks)
        and tampered_manifest_rejected
    )
    if not integrity_pass:
        route_status = "AUDIT_FAILED"
    elif config["cpu_reachable"] and config["score_relevant_if_completed"]:
        route_status = "PROMOTE_TO_BOUNDED_PREFLIGHT"
    elif not config["cpu_reachable"]:
        route_status = "BLOCKED"
    else:
        route_status = "DO_NOT_PROMOTE"

    report = {
        "schema_version": 1,
        "route_id": config["route_id"],
        "claims": claims,
        "audit_kind": "source_and_protocol_feasibility_only",
        "scientific_evidence": False,
        "live_judged_score": contract["live_verdict"]["score"],
        "live_judged_maximum": contract["live_verdict"]["maximum"],
        "candidate_score": None,
        "current_live_verdicts": expected_verdicts,
        "paper_source": source,
        "exact_scope": config["exact_scope"],
        "environment": resolved_environment(),
        "protected_c2_regression_pass": c2_pass,
        "verdicts_match_contract": verdicts_match,
        "source_matches_contract": source_matches_contract,
        "hash_checks": hash_checks,
        "pattern_checks": pattern_checks,
        "negative_control": {
            "name": "tampered_required_hash",
            "target": first_hash[0],
            "tampered_manifest_rejected": tampered_manifest_rejected,
        },
        "integrity_pass": integrity_pass,
        "cpu_reachable": config["cpu_reachable"],
        "score_relevant_if_completed": config["score_relevant_if_completed"],
        "route_status": route_status,
        "decision_basis": config["decision_basis"],
        "blockers": config["blockers"],
        "next_step": config["next_step"],
        "hf_job_launched": False,
        "hf_space_modified": False,
        "publication_authorized": False,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stem = f"route_audit_{config['route_id']}"
    json_path = ARTIFACTS / f"{stem}.json"
    text_path = ARTIFACTS / f"{stem}.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    text_lines = [
        f"route_id: {report['route_id']}",
        f"claims: {', '.join(claims)}",
        f"route_status: {route_status}",
        "scientific_evidence: false",
        f"live_judged_score: {report['live_judged_score']}/{report['live_judged_maximum']}",
        "candidate_score: not estimated",
        f"protected_c2_regression: {'PASS' if c2_pass else 'FAIL'}",
        f"integrity_checks: {'PASS' if integrity_pass else 'FAIL'}",
        f"cpu_reachable: {str(config['cpu_reachable']).lower()}",
        f"score_relevant_if_completed: {str(config['score_relevant_if_completed']).lower()}",
        f"paper_source: {source['url']}",
        f"paper_anchor: {source['anchor']}",
        f"paper_retrieved_on: {source['retrieved_on']}",
        f"paper_sha256: {source['sha256']}",
        "exact_scope:",
        f"  {config['exact_scope']}",
        "decision_basis:",
        *[f"  - {item}" for item in config["decision_basis"]],
        "blockers:",
        *([f"  - {item}" for item in config["blockers"]] or ["  - none"]),
        f"next_step: {config['next_step']}",
        "hf_job_launched: false",
        "hf_space_modified: false",
    ]
    text_path.write_text("\n".join(text_lines) + "\n")

    eval_lines = [
        f"# Route audit: {config['route_id']}",
        "",
        "This run is source/protocol feasibility evidence only. It does not verify or falsify a",
        "scientific claim and it does not estimate an unjudged score.",
        "",
        f"- Claims: **{', '.join(claims)}**",
        f"- Route status: **{route_status}**",
        f"- Live judged score (unchanged): **{report['live_judged_score']}/{report['live_judged_maximum']}**",
        "- Candidate score: **not estimated**",
        f"- Protected C2 regression: **{'PASS' if c2_pass else 'FAIL'}**",
        f"- Audit integrity: **{'PASS' if integrity_pass else 'FAIL'}**",
        f"- Exact scope CPU-reachable: **{'yes' if config['cpu_reachable'] else 'no'}**",
        "- HF job launched: **no**",
        "- Judged HF Space modified: **no**",
        "",
        "## Exact audited scope",
        "",
        config["exact_scope"],
        "",
        "## Decision basis",
        "",
        *[f"- {item}" for item in config["decision_basis"]],
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in config["blockers"]] or ["- None at route-audit level."]),
        "",
        "## Next step",
        "",
        config["next_step"],
        "",
        "## Evidence files",
        "",
        f"- `.openresearch/artifacts/{json_path.name}`",
        f"- `.openresearch/artifacts/{text_path.name}`",
        "",
    ]
    (ROOT / "EVAL.md").write_text("\n".join(eval_lines))

    print(text_path.read_text(), end="")
    print(json_path.read_text(), end="")
    print(f"ROUTE_AUDIT_STATUS={route_status}")
    print("CANDIDATE_SCORE=NOT_ESTIMATED")
    print(f"PROTECTED_C2_REGRESSION={'PASS' if c2_pass else 'FAIL'}")
    return 0 if integrity_pass else 1
