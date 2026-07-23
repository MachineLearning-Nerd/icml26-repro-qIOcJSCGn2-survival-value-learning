#!/usr/bin/env python3
"""Run the full-agent PointMaze gate after an exact CPU queue releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

from audit_full_agent_protocol import verify_upstream_source


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS = ROOT / "outputs" / "full_agent_probe_queue.json"
PROCESS_QUERY_UNAVAILABLE = "<process-query-unavailable>"
RUNNER = ROOT / "repro" / "src" / "run_full_agent_probe.py"
AUDITOR = ROOT / "repro" / "src" / "audit_full_agent_protocol.py"
EXPECTED_DATASET_HASHES = {
    "train": "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    "validation": "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def command_for_pid(pid: int) -> str | None:
    try:
        return " ".join(psutil.Process(pid).cmdline()) or None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except (psutil.AccessDenied, OSError):
        return PROCESS_QUERY_UNAVAILABLE


def blocker_matches(pid: int, marker: str) -> bool:
    command = command_for_pid(pid)
    return command == PROCESS_QUERY_UNAVAILABLE or (
        command is not None and marker in command
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def safe_artifact(path_text: object) -> Path | None:
    if not isinstance(path_text, str) or not path_text:
        return None
    path = (ROOT / path_text).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return None
    return path if path.is_file() else None


def artifact_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        payload.get("status") == "complete"
        and payload.get("scope") == "execution_gate_not_paper_scale"
        and payload.get("target_updates") == 100
        and payload.get("completed_updates") == 100
        and payload.get("checkpoint_reload_exact") is True
        and payload.get("dataset_hashes") == EXPECTED_DATASET_HASHES
        and payload.get("requested_bins") == 800
        and payload.get("actual_bins") == 499
        and payload.get("includes_twin_critic") is True
        and payload.get("includes_low_actor") is True
        and payload.get("includes_high_actor") is True
        and payload.get("state_sha256") == payload.get("restored_state_sha256")
    ):
        return False
    protocol_path = safe_artifact(payload.get("protocol_artifact"))
    manifest_path = safe_artifact(payload.get("checkpoint_manifest"))
    if protocol_path is None or manifest_path is None:
        return False
    if sha256_file(protocol_path) != payload.get("protocol_artifact_sha256"):
        return False
    if sha256_file(manifest_path) != payload.get("checkpoint_manifest_sha256"):
        return False
    try:
        protocol_artifact = json.loads(protocol_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        current_upstream_source = verify_upstream_source()
    except (OSError, json.JSONDecodeError):
        return False
    except (ValueError, FileNotFoundError):
        return False
    protocol_sha256 = protocol_artifact.pop("sha256", None)
    if (
        protocol_sha256 != payload.get("protocol_sha256")
        or canonical_digest(protocol_artifact) != protocol_sha256
        or protocol_artifact.get("upstream_source") != payload.get("upstream_source")
        or protocol_artifact.get("upstream_source") != current_upstream_source
        or protocol_artifact.get("dataset_hashes") != EXPECTED_DATASET_HASHES
        or protocol_artifact.get("backend") != "cpu"
        or protocol_artifact.get("execution_wrapper_sha256")
        != sha256_file(RUNNER)
        or protocol_artifact.get("provenance_auditor_sha256")
        != sha256_file(AUDITOR)
    ):
        return False
    checkpoint_name = manifest.get("active_checkpoint")
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        return False
    checkpoint = manifest_path.with_name(checkpoint_name)
    return (
        checkpoint.is_file()
        and manifest.get("completed_updates") == 100
        and manifest.get("protocol_sha256") == protocol_sha256
        and manifest.get("checkpoint_sha256") == payload.get("checkpoint_sha256")
        and sha256_file(checkpoint) == payload.get("checkpoint_sha256")
        and manifest.get("state_sha256") == payload.get("state_sha256")
    )


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_attempt(
    command: list[str], log_path: Path, timeout_s: float
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_s), "exited"
        except subprocess.TimeoutExpired:
            terminate(process)
            return 124, "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-command", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    args = parser.parse_args()
    if args.wait_pid <= 0 or args.poll_seconds <= 0.0:
        raise ValueError("wait PID and poll interval must be positive")
    if args.max_attempts <= 0 or args.timeout_hours <= 0.0:
        raise ValueError("attempts and timeout must be positive")

    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"repository Python is missing: {python}")
    artifact = ROOT / "outputs" / "full_agent_probe" / "COMPLETE.json"
    log_path = ROOT / "outputs" / "full_agent_probe.log"
    command = [
        str(python),
        "repro/src/run_full_agent_probe.py",
        "--updates",
        "100",
        "--log-interval",
        "10",
        "--checkpoint-interval",
        "10",
    ]
    progress = {
        "paper": "qIOcJSCGn2",
        "status": "waiting",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "blocker": {
            "pid": args.wait_pid,
            "command_marker": args.wait_command,
            "status": "matching",
        },
        "policy": {
            "max_attempts": args.max_attempts,
            "timeout_hours": args.timeout_hours,
            "exact_repository_python": str(python),
            "checkpoint_resume": "atomic immutable generations plus active pointer",
        },
        "command": command,
        "artifact": str(artifact),
        "log": str(log_path),
        "attempts": [],
    }
    atomic_json(args.progress, progress)
    while blocker_matches(args.wait_pid, args.wait_command):
        progress["blocker"]["last_matching_command"] = command_for_pid(
            args.wait_pid
        )
        progress["blocker"]["last_checked_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        atomic_json(args.progress, progress)
        time.sleep(args.poll_seconds)

    progress["blocker"]["status"] = "released_or_identity_changed"
    progress["blocker"]["released_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if artifact_complete(artifact):
        progress["status"] = "complete"
        progress["completion_source"] = "preexisting_verified_artifact"
        atomic_json(args.progress, progress)
        return

    progress["status"] = "running"
    atomic_json(args.progress, progress)
    for number in range(1, args.max_attempts + 1):
        attempt = {
            "number": number,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        progress["attempts"].append(attempt)
        atomic_json(args.progress, progress)
        returncode, outcome = run_attempt(
            command, log_path, args.timeout_hours * 3600.0
        )
        attempt.update(
            {
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "returncode": returncode,
                "outcome": outcome,
                "artifact_complete": artifact_complete(artifact),
            }
        )
        atomic_json(args.progress, progress)
        if returncode == 0 and attempt["artifact_complete"]:
            progress["status"] = "complete"
            break
    else:
        progress["status"] = "failed_after_retries"
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(args.progress, progress)
    print(json.dumps({"status": progress["status"]}, indent=2))


if __name__ == "__main__":
    main()
