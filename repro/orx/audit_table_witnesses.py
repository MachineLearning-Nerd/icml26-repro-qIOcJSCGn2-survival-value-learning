#!/usr/bin/env python3
"""Independent reconstruction of table witnesses; does not import the primary search."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(center: int) -> tuple[Fraction, Fraction]:
    return max(Fraction(0), Fraction(2 * center - 1, 2)), Fraction(2 * center + 1, 2)


def reconstruct(counts: list[int], ddof: int) -> tuple[Fraction, Fraction]:
    rates = [Fraction(2 * count, 5) for count in counts]
    mean = sum(rates, start=Fraction(0)) / 4
    variance = sum((rate - mean) ** 2 for rate in rates) / (4 - ddof)
    return mean, variance


def witness_passes(
    counts: list[int], reported_mean: int, reported_std: int, ddof: int
) -> tuple[bool, dict[str, object]]:
    integer_counts = (
        len(counts) == 4
        and counts == sorted(counts)
        and all(type(count) is int and 0 <= count <= 250 for count in counts)
    )
    if not integer_counts:
        return False, {"integer_counts": False}
    mean, variance = reconstruct(counts, ddof)
    mean_lower, mean_upper = interval(reported_mean)
    std_lower, std_upper = interval(reported_std)
    passed = (
        mean_lower <= mean < mean_upper
        and std_lower**2 <= variance < std_upper**2
    )
    return passed, {
        "integer_counts": True,
        "mean_fraction": f"{mean.numerator}/{mean.denominator}",
        "mean_percent": float(mean),
        "variance_fraction": f"{variance.numerator}/{variance.denominator}",
        "std_percent": math.sqrt(float(variance)),
        "mean_interval_pass": mean_lower <= mean < mean_upper,
        "std_interval_pass": std_lower**2 <= variance < std_upper**2,
    }


def independent_find(
    reported_mean: int, reported_std: int, ddof: int
) -> list[int] | None:
    """Exhaustive ordered-count search implemented independently from the primary."""
    mean_lower, mean_upper = interval(reported_mean)
    std_lower, std_upper = interval(reported_std)
    if mean_lower > 100 or mean_upper <= 0:
        return None
    if std_lower**2 > Fraction(10000, 4 - ddof):
        return None
    for total in range(1001):
        mean = Fraction(total, 10)
        if not (mean_lower <= mean < mean_upper):
            continue
        for first in range(0, min(250, total // 4) + 1):
            max_second = min(250, (total - first) // 3)
            for second in range(first, max_second + 1):
                remaining = total - first - second
                low_third = max(second, remaining - 250)
                high_third = min(250, remaining // 2)
                for third in range(low_third, high_third + 1):
                    fourth = remaining - third
                    counts = [first, second, third, fourth]
                    _, variance = reconstruct(counts, ddof)
                    if std_lower**2 <= variance < std_upper**2:
                        return counts
    return None


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: audit_table_witnesses.py CONFIG PRIMARY OUTPUT")
    config_path, primary_path, output_path = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_text())
    primary = json.loads(primary_path.read_text())

    targets = {row["id"]: row for row in config["targets"]}
    target_checks = []
    for result in primary["targets"]:
        target = targets[result["id"]]
        convention_checks = []
        for convention in config["protocol"]["std_conventions_checked"]:
            name = convention["name"]
            witness = result["witnesses"][name]
            if witness is None:
                independent_witness = independent_find(
                    target["reported_mean_percent"],
                    target["reported_std_percent"],
                    convention["ddof"],
                )
                convention_checks.append(
                    {
                        "convention": name,
                        "witness_present": False,
                        "independent_witness": independent_witness,
                        "pass": independent_witness is None,
                    }
                )
                continue
            passed, details = witness_passes(
                witness["success_counts"],
                target["reported_mean_percent"],
                target["reported_std_percent"],
                convention["ddof"],
            )
            convention_checks.append(
                {
                    "convention": name,
                    "witness_present": True,
                    "pass": passed,
                    "reconstruction": details,
                }
            )
        at_least_one_valid = any(
            row["witness_present"] and row["pass"] for row in convention_checks
        )
        target_checks.append(
            {
                "id": result["id"],
                "conventions": convention_checks,
                "primary_compatible": result["compatible_under_any_convention"],
                "independent_compatible": at_least_one_valid,
                "pass": (
                    result["compatible_under_any_convention"] == at_least_one_valid
                    and all(row["pass"] for row in convention_checks)
                ),
            }
        )

    control_checks = []
    for control in config["destructive_controls"]:
        primary_control = next(
            row for row in primary["destructive_controls"] if row["id"] == control["id"]
        )
        if control["id"] == "impossible_bounded_std":
            lower_std = Fraction(2 * control["reported_std_percent"] - 1, 2)
            universal_sample_variance_bound = Fraction(10000, 3)
            independent_rejection = lower_std**2 > universal_sample_variance_bound
            proof = (
                "Popoviciu/Bhatia-Davis bound with n/(n-1) sample correction: "
                "sample variance <= 10000/3 < 59.5^2."
            )
        elif control["id"] == "impossible_mean_above_range":
            lower_mean = Fraction(2 * control["reported_mean_percent"] - 1, 2)
            independent_rejection = lower_mean > 100
            proof = "Every seed rate is at most 100, so their mean is at most 100."
        else:
            independent_rejection = False
            proof = "Unknown destructive control."
        control_checks.append(
            {
                "id": control["id"],
                "primary_rejected": not primary_control["compatible_under_any_convention"],
                "independent_rejected": independent_rejection,
                "proof": proof,
                "pass": (
                    not primary_control["compatible_under_any_convention"]
                    and independent_rejection
                ),
            }
        )

    input_hashes_match = (
        primary["inputs"]["config_sha256"] == sha256(config_path)
        and primary["inputs"]["paper_sha256"] == config["paper_source"]["sha256"]
    )
    ids_match = set(targets) == {row["id"] for row in primary["targets"]}
    passed = (
        input_hashes_match
        and ids_match
        and all(row["pass"] for row in target_checks)
        and all(row["pass"] for row in control_checks)
    )
    report = {
        "schema_version": 1,
        "auditor": "independent_fraction_reconstruction",
        "imports_primary_computation": False,
        "input_hashes_match": input_hashes_match,
        "target_ids_match": ids_match,
        "target_checks": target_checks,
        "destructive_control_checks": control_checks,
        "pass": passed,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
