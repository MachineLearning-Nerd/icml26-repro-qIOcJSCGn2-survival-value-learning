#!/usr/bin/env python3
"""Primary exhaustive arithmetic audit for the four-seed paper tables."""

from __future__ import annotations

from fractions import Fraction
import hashlib
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "repro" / "configs" / "table_arithmetic_audit.json"
AUDITOR = ROOT / "repro" / "orx" / "audit_table_witnesses.py"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(center: int) -> tuple[Fraction, Fraction]:
    return max(Fraction(0), Fraction(2 * center - 1, 2)), Fraction(2 * center + 1, 2)


def variance_for_counts(counts: tuple[int, int, int, int], ddof: int) -> Fraction:
    rates = [Fraction(2 * count, 5) for count in counts]
    mean = sum(rates, start=Fraction(0)) / 4
    return sum((rate - mean) ** 2 for rate in rates) / (4 - ddof)


def find_witness(reported_mean: int, reported_std: int, ddof: int) -> dict[str, object] | None:
    mean_lower, mean_upper = interval(reported_mean)
    std_lower, std_upper = interval(reported_std)
    if mean_lower > 100 or mean_upper <= 0:
        return None
    universal_variance_bound = Fraction(10000, 4 - ddof)
    if std_lower**2 > universal_variance_bound:
        return None
    for total in range(0, 1001):
        mean = Fraction(total, 10)
        if not (mean_lower <= mean < mean_upper):
            continue
        mean_count = Fraction(total, 4)
        count_radius = 5 * std_upper
        low = max(0, math.floor(mean_count - count_radius) - 1)
        high = min(250, math.ceil(mean_count + count_radius) + 1)
        for first in range(low, high + 1):
            for second in range(first, high + 1):
                for third in range(second, high + 1):
                    fourth = total - first - second - third
                    if fourth < third or fourth > high or fourth > 250:
                        continue
                    counts = (first, second, third, fourth)
                    variance = variance_for_counts(counts, ddof)
                    if std_lower**2 <= variance < std_upper**2:
                        rates = [float(Fraction(2 * count, 5)) for count in counts]
                        return {
                            "success_counts": list(counts),
                            "seed_rates_percent": rates,
                            "mean_percent": float(mean),
                            "std_percent": math.sqrt(float(variance)),
                            "variance_fraction": f"{variance.numerator}/{variance.denominator}",
                            "ddof": ddof,
                        }
    return None


def environment() -> dict[str, object]:
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


def validate_manifest(manifest: dict[str, object]) -> bool:
    return all(
        (ROOT / relative).is_file() and sha256(ROOT / relative) == expected
        for relative, expected in manifest["files"].items()
    )


def evaluate_table_arithmetic(contract: dict[str, object], c2_pass: bool) -> int:
    config = json.loads(CONFIG.read_text())
    if config["schema_version"] != 1 or config["candidate_score"] is not None:
        raise RuntimeError("invalid or score-claiming arithmetic audit configuration")
    if config["protocol"]["binary_trials_per_seed"] != 250:
        raise RuntimeError("audit must retain five tasks times 50 episodes")
    if config["protocol"]["training_seed_count"] != 4:
        raise RuntimeError("audit must retain the paper's four-seed protocol")
    if config["paper_source"]["sha256"] != contract["source"]["sha256"]:
        raise RuntimeError("paper source hash differs from the judge contract")

    target_results = []
    for target in config["targets"]:
        witnesses = {}
        for convention in config["protocol"]["std_conventions_checked"]:
            witnesses[convention["name"]] = find_witness(
                target["reported_mean_percent"],
                target["reported_std_percent"],
                convention["ddof"],
            )
        target_results.append(
            {
                **target,
                "witnesses": witnesses,
                "compatible_under_any_convention": any(
                    witness is not None for witness in witnesses.values()
                ),
            }
        )

    control_results = []
    for control in config["destructive_controls"]:
        witnesses = {}
        for convention in config["protocol"]["std_conventions_checked"]:
            witnesses[convention["name"]] = find_witness(
                control["reported_mean_percent"],
                control["reported_std_percent"],
                convention["ddof"],
            )
        control_results.append(
            {
                **control,
                "witnesses": witnesses,
                "compatible_under_any_convention": any(
                    witness is not None for witness in witnesses.values()
                ),
            }
        )

    all_targets_compatible = all(
        row["compatible_under_any_convention"] for row in target_results
    )
    destructive_controls_rejected = all(
        not row["compatible_under_any_convention"] for row in control_results
    )
    primary = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "inputs": {
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "paper_url": config["paper_source"]["url"],
            "paper_retrieved_on": config["paper_source"]["retrieved_on"],
            "paper_sha256": config["paper_source"]["sha256"],
            "evaluator_source_sha256": sha256(Path(__file__)),
            "upstream_evaluation_source": "upstream/hsvl/utils/evaluation.py",
            "upstream_evaluation_source_sha256": sha256(
                ROOT / "upstream" / "hsvl" / "utils" / "evaluation.py"
            ),
        },
        "protocol": config["protocol"],
        "environment": environment(),
        "targets": target_results,
        "destructive_controls": control_results,
        "all_targets_compatible": all_targets_compatible,
        "destructive_controls_rejected": destructive_controls_rejected,
        "primary_conclusion": (
            "compatible_not_verified"
            if all_targets_compatible
            else "at_least_one_table_entry_falsified_arithmetically"
        ),
        "candidate_score": None,
        "hf_job_launched": False,
        "hf_space_modified": False,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    primary_path = ARTIFACTS / "table_arithmetic_primary.json"
    audit_path = ARTIFACTS / "table_arithmetic_independent_audit.json"
    text_path = ARTIFACTS / "table_arithmetic_summary.txt"
    manifest_path = ARTIFACTS / "table_arithmetic_manifest.json"
    primary_path.write_text(json.dumps(primary, indent=2, sort_keys=True) + "\n")

    audited = subprocess.run(
        [sys.executable, str(AUDITOR), str(CONFIG), str(primary_path), str(audit_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {"pass": False}
    primary_pass = destructive_controls_rejected
    integrity_pass = c2_pass and primary_pass and audited.returncode == 0 and audit["pass"]

    summary_lines = [
        f"audit_id: {config['audit_id']}",
        f"primary_conclusion: {primary['primary_conclusion']}",
        f"all_targets_compatible: {str(all_targets_compatible).lower()}",
        f"destructive_controls_rejected: {str(destructive_controls_rejected).lower()}",
        f"independent_audit: {'PASS' if audit['pass'] else 'FAIL'}",
        f"protected_c2_regression: {'PASS' if c2_pass else 'FAIL'}",
        "candidate_score: not estimated",
        "C1: not_applicable_to_table_arithmetic",
    ]
    for row in target_results:
        summary_lines.append(
            f"{row['id']}: {'COMPATIBLE' if row['compatible_under_any_convention'] else 'FALSIFIED'}"
        )
    text_path.write_text("\n".join(summary_lines) + "\n")

    manifest = {
        "schema_version": 1,
        "fail_closed": True,
        "files": {
            str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            str(AUDITOR.relative_to(ROOT)): sha256(AUDITOR),
            str(primary_path.relative_to(ROOT)): sha256(primary_path),
            str(audit_path.relative_to(ROOT)): sha256(audit_path),
            str(text_path.relative_to(ROOT)): sha256(text_path),
            ".openresearch/judge_contract.json": sha256(
                ROOT / ".openresearch" / "judge_contract.json"
            ),
            "repro/src/verify_survival.py": sha256(
                ROOT / "repro" / "src" / "verify_survival.py"
            ),
            "repro/tests/test_svl.py": sha256(
                ROOT / "repro" / "tests" / "test_svl.py"
            ),
            "upstream/hsvl/utils/evaluation.py": sha256(
                ROOT / "upstream" / "hsvl" / "utils" / "evaluation.py"
            ),
        },
    }
    manifest_valid = validate_manifest(manifest)
    tampered = json.loads(json.dumps(manifest))
    first_path = sorted(tampered["files"])[0]
    old_hash = tampered["files"][first_path]
    tampered["files"][first_path] = ("0" if old_hash[0] != "0" else "1") + old_hash[1:]
    tampered_manifest_rejected = not validate_manifest(tampered)
    manifest["self_check_pass"] = manifest_valid
    manifest["negative_control"] = {
        "name": "tampered_file_hash",
        "target": first_path,
        "rejected": tampered_manifest_rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    integrity_pass = integrity_pass and manifest_valid and tampered_manifest_rejected

    claim_compatibility = {
        claim_id: all(
            row["compatible_under_any_convention"]
            for row in target_results
            if row["claim_id"] == claim_id
        )
        for claim_id in ("C4", "C5", "C6")
    }
    claim_recommendations = {
        "C1": "toy",
        "C4": "inconclusive" if claim_compatibility["C4"] else "falsified",
        "C5": "inconclusive" if claim_compatibility["C5"] else "falsified",
        "C6": "inconclusive" if claim_compatibility["C6"] else "falsified",
    }
    eval_lines = [
        "# Interpretation C: table arithmetic falsification",
        "",
        "This CPU audit asks whether the reported integer mean ± std entries could arise",
        "from four seed rates at the released evaluator's five-task × 50-episode",
        "granularity. Compatibility is not policy-performance verification.",
        "",
        f"- Primary conclusion: **{primary['primary_conclusion']}**",
        f"- Protected C2 regression: **{'PASS' if c2_pass else 'FAIL'}**",
        f"- Independent auditor: **{'PASS' if audit['pass'] else 'FAIL'}**",
        f"- Destructive controls: **{'PASS' if destructive_controls_rejected else 'FAIL'}**",
        f"- Fail-closed manifest: **{'PASS' if manifest_valid and tampered_manifest_rejected else 'FAIL'}**",
        "- Candidate score: **not estimated**",
        "- HF job launched: **no**",
        "- Judged HF Space modified: **no**",
        "",
        "| Claim | Arithmetic recommendation | Boundary |",
        "|---|---|---|",
        "| C1 | toy | Not a four-seed table-summary claim. |",
        f"| C4 | {claim_recommendations['C4']} | Arithmetic compatibility cannot reproduce performance. |",
        f"| C5 | {claim_recommendations['C5']} | Both task conjuncts were checked. |",
        f"| C6 | {claim_recommendations['C6']} | Actor identity and performance remain unmeasured. |",
        "",
        "## Target results",
        "",
    ]
    for row in target_results:
        eval_lines.append(
            f"- `{row['id']}`: "
            f"{'compatible witness found' if row['compatible_under_any_convention'] else 'no witness under either convention'}."
        )
    eval_lines.extend(
        [
            "",
            "## Evidence",
            "",
            "- `.openresearch/artifacts/table_arithmetic_primary.json`",
            "- `.openresearch/artifacts/table_arithmetic_independent_audit.json`",
            "- `.openresearch/artifacts/table_arithmetic_summary.txt`",
            "- `.openresearch/artifacts/table_arithmetic_manifest.json`",
            "",
        ]
    )
    (ROOT / "EVAL.md").write_text("\n".join(eval_lines))

    print(primary_path.read_text(), end="")
    sys.stdout.write(audited.stdout)
    sys.stderr.write(audited.stderr)
    print(text_path.read_text(), end="")
    print(manifest_path.read_text(), end="")
    print(f"TABLE_ARITHMETIC_STATUS={primary['primary_conclusion']}")
    print("CANDIDATE_SCORE=NOT_ESTIMATED")
    print(f"PROTECTED_C2_REGRESSION={'PASS' if c2_pass else 'FAIL'}")
    return 0 if integrity_pass else 1
