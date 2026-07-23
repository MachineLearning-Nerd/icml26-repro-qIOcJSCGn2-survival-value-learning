import json
from pathlib import Path

import numpy as np

import run_full_agent_probe as probe


def test_released_full_agent_config_is_pinned():
    path = (
        probe.UPSTREAM_ROOT / "hsvl" / "config" / "agent" / "hsvl.yaml"
    )
    config = probe.load_released_config(path)
    assert config["ensemble"] is True
    assert config["batch_size"] == 1024
    assert config["value_hidden_dims"] == [512, 512, 512]
    assert config["actor_hidden_dims"] == [512, 512, 512]
    assert config["num_log_bins"] == 800


def test_protocol_identity_is_deterministic_and_extension_safe():
    config = {"ensemble": True, "batch_size": 1024}
    hashes = {"train": "a", "validation": "b"}
    upstream_source = {"manifest_sha256": "source"}
    first = probe.protocol_payload(
        config, probe.DATASET_NAME, hashes, seed=0, upstream_source=upstream_source
    )
    second = probe.protocol_payload(
        dict(reversed(list(config.items()))),
        probe.DATASET_NAME,
        hashes,
        seed=0,
        upstream_source=upstream_source,
    )
    assert probe.canonical_digest(first) == probe.canonical_digest(second)
    assert "updates" not in json.dumps(first)
    assert probe.canonical_digest(first) != probe.canonical_digest(
        probe.protocol_payload(
            config,
            probe.DATASET_NAME,
            hashes,
            seed=1,
            upstream_source=upstream_source,
        )
    )


def test_checkpoint_reuse_fails_closed_on_protocol_or_bytes(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "schema_version": 1,
        "protocol_sha256": "protocol-a",
        "completed_updates": 10,
        "checkpoint_sha256": probe.sha256_file(checkpoint),
    }
    assert probe.checkpoint_reusable(manifest, "protocol-a", checkpoint)
    assert not probe.checkpoint_reusable(manifest, "protocol-b", checkpoint)
    checkpoint.write_bytes(b"changed")
    assert not probe.checkpoint_reusable(manifest, "protocol-a", checkpoint)


def test_state_digest_covers_shape_dtype_and_values():
    baseline = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    assert probe.state_digest(baseline) == probe.state_digest(baseline)
    assert probe.state_digest(baseline) != probe.state_digest(
        {"x": np.array([1.0, 2.0], dtype=np.float32)}
    )
    assert probe.state_digest(baseline) != probe.state_digest(
        {"x": np.array([[1.0, 2.0]], dtype=np.float64)}
    )
    assert probe.state_digest(baseline) != probe.state_digest(
        {"x": np.array([[1.0, 3.0]], dtype=np.float32)}
    )


def test_atomic_checkpoint_round_trip_is_exact(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pkl"
    state = {"network": {"weight": np.arange(6, dtype=np.float32).reshape(2, 3)}}
    manifest = probe.save_checkpoint(state, 7, "protocol", checkpoint)
    restored, restored_manifest = probe.restore_checkpoint(
        {"network": {"weight": np.zeros((2, 3), dtype=np.float32)}},
        "protocol",
        checkpoint,
    )
    np.testing.assert_array_equal(restored["network"]["weight"], state["network"]["weight"])
    assert restored_manifest == manifest
    assert restored_manifest["completed_updates"] == 7
    assert restored_manifest["active_checkpoint"] == "checkpoint-00000007.pkl"
    assert not checkpoint.exists()
    assert (tmp_path / "checkpoint-00000007.pkl").is_file()
    assert (tmp_path / "checkpoint-00000007.manifest.json").is_file()


def test_active_pointer_survives_unpublished_new_generation(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pkl"
    first = {"network": {"weight": np.array([1.0], dtype=np.float32)}}
    probe.save_checkpoint(first, 10, "protocol", checkpoint)
    unpublished = tmp_path / "checkpoint-00000020.pkl"
    unpublished.write_bytes(b"interrupted-generation")
    restored, manifest = probe.restore_checkpoint(
        {"network": {"weight": np.array([0.0], dtype=np.float32)}},
        "protocol",
        checkpoint,
    )
    np.testing.assert_array_equal(restored["network"]["weight"], [1.0])
    assert manifest["completed_updates"] == 10
