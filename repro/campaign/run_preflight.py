#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2604.17551"
PAPER_SHA256 = "69f0db8b7819cac262f4c91e945fc7a6d4b011e7d603fbfe3f0844fa5acecec0"
JUDGED_REVISION = "612e99f40c171c7391db6529fe6ac4d9aa6619fa"
ORX_EXPERIMENT_ID = "d698ce45-4b6d-45e9-8f2e-b33edf043b3e"
LEMMA42_URL = (
    "https://huggingface.co/spaces/DineshAI/qIOcJSCGn2/resolve/"
    f"{JUDGED_REVISION}/repro/src/verify_lemma42.py"
)
LEMMA42_SHA256 = "9991d8e29a2bfa1dd0fb8a712a25634840ee6916d31a9c7bebc7476564ed7e2f"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SVL-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def run(name: str, command: list[str]) -> dict:
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log = result.stdout + result.stderr
    (ARTIFACTS / f"{name}.log").write_text(log)
    if result.returncode:
        print(f"===== {name} failed =====", file=sys.stderr)
        print(log, file=sys.stderr)
    return {
        "command": command,
        "duration_seconds": round(time.monotonic() - started, 3),
        "returncode": result.returncode,
        "log_sha256": sha256(log.encode()),
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = {
        "execution_attestation": {
            "orx_experiment_id": ORX_EXPERIMENT_ID,
            "note": "HF job ID and cpu-upgrade flavor are verified from external ORX/HF job metadata.",
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_platform_request": os.environ.get("JAX_PLATFORMS"),
        "checks": {},
    }
    contract = ROOT / "repro/campaign/campaign_contract.json"
    report["campaign_contract_sha256"] = sha256(contract.read_bytes())
    report["source_files"] = {
        str(path.relative_to(ROOT)): sha256(path.read_bytes())
        for path in (
            ROOT / "upstream/SOURCE_REVISION",
            ROOT / "upstream/hsvl/agent.py",
            ROOT / "repro/campaign/flat_svl.py",
            ROOT / "repro/campaign/visual_hsvl.py",
        )
    }

    import jax

    devices = [f"{device.platform}:{device.device_kind}" for device in jax.devices()]
    report["jax_devices"] = devices
    report["cpu_only"] = bool(devices) and all(device.platform == "cpu" for device in jax.devices())

    paper = download(PAPER_URL)
    paper_hash = sha256(paper)
    anchors = [b"S4.Thmtheorem1", b"S4.Thmtheorem2", b"S5.T1", b"S5.T2", b"A1.T3", b"A1.T5"]
    report["paper_source"] = {
        "url": PAPER_URL,
        "sha256": paper_hash,
        "expected_sha256": PAPER_SHA256,
        "anchors_present": {anchor.decode(): anchor in paper for anchor in anchors},
    }

    report["checks"]["claim_2"] = run("claim_2", [sys.executable, "repro/src/verify_survival.py"])

    lemma42 = download(LEMMA42_URL)
    lemma42_hash = sha256(lemma42)
    lemma_path = ARTIFACTS / "verify_lemma42.py"
    lemma_path.write_bytes(lemma42)
    report["lemma42_source"] = {
        "url": LEMMA42_URL,
        "sha256": lemma42_hash,
        "expected_sha256": LEMMA42_SHA256,
    }
    report["checks"]["claim_3"] = run("claim_3", [sys.executable, str(lemma_path)])

    smoke_cases = [
        ("hsvl_antmaze", "hsvl", "antmaze-giant-navigate-v0"),
        ("hsvl_humanoid", "hsvl", "humanoidmaze-giant-navigate-v0"),
        ("hsvl_visual_antmaze", "visual_hsvl", "visual-antmaze-giant-navigate-v0"),
        ("hiql_humanoid", "hiql", "humanoidmaze-giant-navigate-v0"),
        ("hiql_visual_antmaze", "hiql", "visual-antmaze-giant-navigate-v0"),
        ("flat_svl_humanoid", "flat_svl", "humanoidmaze-giant-navigate-v0"),
        ("crl_humanoid", "crl", "humanoidmaze-giant-navigate-v0"),
    ]
    for name, method, dataset in smoke_cases:
        report["checks"][name] = run(
            name,
            [sys.executable, "repro/campaign/smoke_method.py", method, dataset],
        )

    required_implementations = {
        "flat_svl": ROOT / "repro" / "campaign" / "flat_svl.py",
        "visual_hsvl": ROOT / "repro" / "campaign" / "visual_hsvl.py",
    }
    report["required_implementations"] = {
        name: path.is_file() for name, path in required_implementations.items()
    }

    successful_checks = all(check["returncode"] == 0 for check in report["checks"].values())
    source_ok = (
        paper_hash == PAPER_SHA256
        and all(report["paper_source"]["anchors_present"].values())
        and lemma42_hash == LEMMA42_SHA256
    )
    report["preflight_passed"] = (
        report["cpu_only"]
        and source_ok
        and successful_checks
        and all(report["required_implementations"].values())
    )
    report_path = ARTIFACTS / "preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = [
        "# HF CPU campaign preflight",
        "",
        f"- ORX experiment: `{ORX_EXPERIMENT_ID}` (HF job metadata verified externally)",
        f"- CPU-only JAX: {'PASS' if report['cpu_only'] else 'FAIL'} ({', '.join(devices)})",
        f"- Paper source: {'PASS' if source_ok else 'FAIL'} (`{paper_hash}`)",
        f"- Overall: {'PASS' if report['preflight_passed'] else 'FAIL'}",
        "",
        "| Check | Exit | Seconds | Log SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, check in report["checks"].items():
        rows.append(
            f"| {name} | {check['returncode']} | {check['duration_seconds']} | `{check['log_sha256']}` |"
        )
    rows.extend([
        "",
        "Flat SVL and visual HSVL are explicitly recorded as clean-room reconstructions in ",
        "`repro/campaign/campaign_contract.json`; they are not represented as author-released code.",
        "",
    ])
    (ROOT / "EVAL.md").write_text("\n".join(rows))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
