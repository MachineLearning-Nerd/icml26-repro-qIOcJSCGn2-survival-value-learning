#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXPECTED_NAME = "icml26-survival-value-learning"
EXPECTED_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
OLD_SLUG = "icml26-repro-qIOcJSCGn2-survival-value-learning"
EXPECTED_BRANCHES = {
    "audit/claim-1-exact-scale-readiness",
    "audit/claim-3-mle-erm-theorem",
    "audit/claim-6-flat-svl-crl",
    "audit/claims-4-5-benchmark-readiness",
    "audit/interpretation-literal-gate-fix",
    "audit/interpretation-literal-protocol",
    "audit/interpretation-official-artifact-witness",
    "audit/interpretation-table-arithmetic",
    "audit/round-1-route-feasibility",
    "audit/three-interpretation-retry",
    "baseline/frozen-four-of-twelve",
    "baseline/six-claim-seed-20260723",
    "experiment/hf-cpu-dependency-repair",
    "experiment/hf-cpu-full-scale-preflight",
    "experiment/hf-cpu-hashable-config",
    "experiment/hf-cpu-integration-crl-humanoid",
    "experiment/hf-cpu-integration-flat-svl-humanoid",
    "experiment/hf-cpu-integration-hiql-ant",
    "experiment/hf-cpu-integration-hiql-humanoid",
    "experiment/hf-cpu-integration-hsvl-humanoid",
    "experiment/hf-cpu-integration-visual-hiql-ant",
    "experiment/hf-cpu-integration-visual-hsvl-ant",
    "experiment/hf-cpu-method-import",
    "experiment/hf-cpu-opengl-repair",
    "experiment/hf-cpu-resumable-runner",
    "experiment/hf-cpu-throughput-benchmark",
    "experiment/paper-scale-crl-humanoid-seed-0",
    "experiment/paper-scale-crl-humanoid-seed-1",
    "experiment/paper-scale-crl-humanoid-seed-2",
    "experiment/paper-scale-crl-humanoid-seed-3",
    "experiment/paper-scale-flat-svl-humanoid-seed-0",
    "experiment/paper-scale-flat-svl-humanoid-seed-1",
    "experiment/paper-scale-flat-svl-humanoid-seed-2",
    "experiment/paper-scale-flat-svl-humanoid-seed-3",
    "experiment/paper-scale-hiql-ant-seed-0",
    "experiment/paper-scale-hiql-ant-seed-1",
    "experiment/paper-scale-hiql-ant-seed-2",
    "experiment/paper-scale-hiql-ant-seed-3",
    "experiment/paper-scale-hiql-humanoid-seed-0",
    "experiment/paper-scale-hiql-humanoid-seed-1",
    "experiment/paper-scale-hiql-humanoid-seed-2",
    "experiment/paper-scale-hiql-humanoid-seed-3",
    "experiment/paper-scale-hsvl-ant-seed-0",
    "experiment/paper-scale-hsvl-ant-seed-1",
    "experiment/paper-scale-hsvl-ant-seed-2",
    "experiment/paper-scale-hsvl-ant-seed-3",
    "experiment/paper-scale-hsvl-humanoid-seed-0",
    "experiment/paper-scale-hsvl-humanoid-seed-1",
    "experiment/paper-scale-hsvl-humanoid-seed-2",
    "experiment/paper-scale-hsvl-humanoid-seed-3",
    "experiment/paper-scale-visual-hiql-ant-seed-0",
    "experiment/paper-scale-visual-hiql-ant-seed-1",
    "experiment/paper-scale-visual-hiql-ant-seed-2",
    "experiment/paper-scale-visual-hiql-ant-seed-3",
    "experiment/paper-scale-visual-hsvl-ant-seed-0",
    "experiment/paper-scale-visual-hsvl-ant-seed-1",
    "experiment/paper-scale-visual-hsvl-ant-seed-2",
    "experiment/paper-scale-visual-hsvl-ant-seed-3",
    "main",
    "release/hf-logbook-text-only",
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = bool(condition)
        if detail:
            details[name] = detail

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "evidence/claim_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "outputs/svl_summary.json").read_text(encoding="utf-8"))
    required_docs = [
        "docs/CLAIM_EVIDENCE.md",
        "docs/BRANCH_AUDIT.md",
        "docs/SOURCE_AUDIT.md",
        "docs/PUBLICATION_GATE.md",
        "docs/research_log.md",
    ]

    check("project_name", f'name = "{EXPECTED_NAME}"' in pyproject)
    check("paper_identity", sources["paper"]["openreview_id"] == "qIOcJSCGn2"
          and sources["paper"]["arxiv_id"] == "2604.17551"
          and "SVL" in readme)
    check("official_source_pin", sources["official_implementation"]["commit"] == "5f13cf22d397be42a87b7d35336db7662879d6db")
    check("citation_and_thanks", "## Citation" in readme and "## Thank you" in readme
          and all(name in readme for name in sources["paper"]["authors"]))
    check("required_docs", all((root / path).is_file() for path in required_docs), ", ".join(required_docs))
    check("six_claim_summary", set(evidence["claims"]) == {"C1", "C2", "C3", "C4", "C5", "C6"})
    check("claim_statuses", all(key in status for key in ("C1", "C2", "C3", "C4", "C5", "C6")))
    check("c2_identity", summary["identity"]["ok"] is True
          and summary["identity"]["max_diff"] < 1e-10)
    check("c2_negative_controls", summary["neg_control_nonsparse"]["ok"] is True
          and summary["neg_control_nonabsorbing"]["ok"] is True)
    check("c2_evidence_record", evidence["claims"]["C2"]["status"] == "VERIFIED"
          and evidence["claims"]["C2"]["promoted"] is True)

    tracked = git(root, "ls-files").splitlines()
    check("no_stale_state", not any(path == ".trackio" or path.startswith(".trackio/") for path in tracked)
          and "logbook.json" not in tracked)
    branch_output = git(root, "branch", "-a")
    final_refs: set[str] = set()
    for line in branch_output.splitlines():
        ref = line.strip().removeprefix("*").strip()
        if " -> " in ref:
            ref = ref.rsplit(" -> ", 1)[1]
        if ref.startswith("remotes/origin/"):
            ref = ref.removeprefix("remotes/origin/")
        elif ref.startswith("origin/"):
            ref = ref.removeprefix("origin/")
        final_refs.add(ref)
    check("branch_surface", "master" not in branch_output and "orx/" not in branch_output
          and EXPECTED_BRANCHES <= final_refs, str(sorted(final_refs)))
    check("branch_audit_rows", sum(line.startswith("| `") for line in (root / "docs/BRANCH_AUDIT.md").read_text(encoding="utf-8").splitlines()) == len(EXPECTED_BRANCHES))
    check("canonical_branch", git(root, "branch", "--show-current") == "main")
    remote = git(root, "remote", "get-url", "origin")
    check("final_remote", EXPECTED_NAME in remote, remote)

    identities = git(root, "log", "--all", "--format=%an <%ae>%n%cn <%ce>").splitlines()
    unexpected = sorted({item for item in identities if item != EXPECTED_IDENTITY})
    check("commit_identity", bool(identities) and not unexpected, ", ".join(unexpected))
    private_path_marker = "/Users/" + "dineshjinjala/"
    content = "\n".join((root / path).read_text(errors="ignore") for path in tracked if (root / path).is_file())
    check("no_private_workspace_path", "dinesh.jinjala@mareana.com" not in git(root, "log", "--all", "--format=%B")
          and private_path_marker not in content)
    documentation = readme + status + "\n" + "\n".join((root / path).read_text(encoding="utf-8") for path in required_docs)
    check("active_links_clean", f"https://github.com/MachineLearning-Nerd/{OLD_SLUG}" not in documentation)

    result = {
        "repository": EXPECTED_NAME,
        "paper": "qIOcJSCGn2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": details,
    }
    if not args.no_write:
        (root / "outputs/verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
