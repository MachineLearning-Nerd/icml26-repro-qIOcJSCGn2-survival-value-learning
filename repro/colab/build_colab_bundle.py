#!/usr/bin/env python3
"""Build the deterministic, self-verifying qIO Colab upload bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


PAPER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PAPER_ROOT / "outputs" / "colab_handoff" / "qio-full-agent-colab-input-v1.tar.gz"
EXPECTED_REVISION = "5f13cf22d397be42a87b7d35336db7662879d6db"
EXPECTED_DATASETS = {
    "data/ogbench/pointmaze-large-navigate-v0.npz": "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    "data/ogbench/pointmaze-large-navigate-v0-val.npz": "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    upstream = PAPER_ROOT / "upstream"
    for path in sorted(upstream.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(PAPER_ROOT).as_posix()
        selected[relative] = path
    for relative in EXPECTED_DATASETS:
        selected[relative] = PAPER_ROOT / relative
    for relative in (
        "repro/configs/upstream_source_manifest.json",
        "repro/src/audit_full_agent_protocol.py",
        "repro/src/run_full_agent_probe.py",
        "repro/colab/run_full_agent_probe_colab.py",
        "repro/colab/package_colab_return.py",
    ):
        selected[relative] = PAPER_ROOT / relative
    missing = [relative for relative, path in selected.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Colab bundle input(s) missing: {missing}")
    observed = {relative: sha256_file(PAPER_ROOT / relative) for relative in EXPECTED_DATASETS}
    if observed != EXPECTED_DATASETS:
        raise ValueError(f"official dataset identities changed: {observed}")
    return dict(sorted(selected.items()))


def tar_info(name: str, size: int, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    selected = input_files()
    records = {
        relative: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in selected.items()
    }
    manifest = {
        "schema_version": 1,
        "purpose": "qio_full_agent_colab_cuda_input",
        "official_source": {
            "repository": "https://github.com/Simple-Robotics/hierarchical-survival-value-learning",
            "commit": EXPECTED_REVISION,
        },
        "official_dataset_hashes": EXPECTED_DATASETS,
        "members": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = PAPER_ROOT / "BUNDLE_MANIFEST.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_bytes(manifest_bytes)
    temporary_manifest.replace(manifest_path)

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = args.archive.with_suffix(args.archive.suffix + ".tmp")
    with temporary_archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                archive.addfile(
                    tar_info("BUNDLE_MANIFEST.json", len(manifest_bytes)),
                    io.BytesIO(manifest_bytes),
                )
                for relative, path in selected.items():
                    with path.open("rb") as handle:
                        archive.addfile(
                            tar_info(relative, path.stat().st_size, path.suffix == ".py"),
                            handle,
                        )
    temporary_archive.replace(args.archive)
    integrity = {
        "schema_version": 1,
        "archive": args.archive.name,
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "member_count": len(selected),
    }
    integrity_path = args.archive.with_name(args.archive.name + ".integrity.json")
    temporary_integrity = integrity_path.with_suffix(integrity_path.suffix + ".tmp")
    temporary_integrity.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    temporary_integrity.replace(integrity_path)
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
