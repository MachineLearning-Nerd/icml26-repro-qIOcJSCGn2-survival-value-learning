"""Checks for the delayed full-agent execution gate."""

import json

import run_full_agent_probe_queue as queue


def test_blocker_identity_is_exact_and_query_failure_waits(monkeypatch):
    monkeypatch.setattr(
        queue, "command_for_pid", lambda pid: "python run_full_matrix_queue.py"
    )
    assert queue.blocker_matches(62748, "run_full_matrix_queue.py")
    assert not queue.blocker_matches(62748, "another.py")
    monkeypatch.setattr(
        queue, "command_for_pid", lambda pid: queue.PROCESS_QUERY_UNAVAILABLE
    )
    assert queue.blocker_matches(62748, "run_full_matrix_queue.py")


def test_completion_requires_100_updates_and_exact_reload(tmp_path, monkeypatch):
    runner = tmp_path / "run_full_agent_probe.py"
    auditor = tmp_path / "audit_full_agent_protocol.py"
    runner.write_text("# runner\n")
    auditor.write_text("# auditor\n")
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "RUNNER", runner)
    monkeypatch.setattr(queue, "AUDITOR", auditor)

    upstream_source = {"manifest_sha256": "source"}
    monkeypatch.setattr(queue, "verify_upstream_source", lambda: upstream_source)

    protocol = {
        "upstream_source": upstream_source,
        "dataset_hashes": queue.EXPECTED_DATASET_HASHES,
        "backend": "cpu",
        "execution_wrapper_sha256": queue.sha256_file(runner),
        "provenance_auditor_sha256": queue.sha256_file(auditor),
    }
    protocol_sha256 = queue.canonical_digest(protocol)
    protocol["sha256"] = protocol_sha256
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))

    checkpoint = tmp_path / "checkpoint-00000100.pkl"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = queue.sha256_file(checkpoint)
    manifest = {
        "active_checkpoint": checkpoint.name,
        "completed_updates": 100,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "state_sha256": "state",
    }
    manifest_path = tmp_path / "checkpoint.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    path = tmp_path / "COMPLETE.json"
    payload = {
        "status": "complete",
        "scope": "execution_gate_not_paper_scale",
        "target_updates": 100,
        "completed_updates": 100,
        "checkpoint_reload_exact": True,
        "dataset_hashes": queue.EXPECTED_DATASET_HASHES,
        "requested_bins": 800,
        "actual_bins": 499,
        "includes_twin_critic": True,
        "includes_low_actor": True,
        "includes_high_actor": True,
        "state_sha256": "state",
        "restored_state_sha256": "state",
        "protocol_artifact": protocol_path.name,
        "protocol_artifact_sha256": queue.sha256_file(protocol_path),
        "protocol_sha256": protocol_sha256,
        "checkpoint_manifest": manifest_path.name,
        "checkpoint_manifest_sha256": queue.sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_sha256,
        "upstream_source": upstream_source,
    }
    path.write_text(json.dumps(payload))
    assert queue.artifact_complete(path)
    payload["completed_updates"] = 90
    path.write_text(json.dumps(payload))
    assert not queue.artifact_complete(path)
