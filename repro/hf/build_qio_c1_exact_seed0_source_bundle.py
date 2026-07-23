#!/usr/bin/env python3
"""Build the deterministic 31-file source archive used by the HF route."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "repro/hf/qio_c1_exact_seed0_source_manifest.json"
CONTRACT = ROOT / "repro/hf/qio_c1_exact_seed0_contract.json"
OUT = ROOT / "hf_jobs/qio-c1-exact-seed0-input"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = json.loads(MANIFEST.read_text())
    files = source["files"]
    if len(files) != 31:
        raise ValueError("expected the frozen 31-file manifest")
    members = dict(files)
    members["repro/hf/qio_c1_exact_seed0_source_manifest.json"] = sha256(MANIFEST)
    members["repro/hf/qio_c1_exact_seed0_contract.json"] = sha256(CONTRACT)
    for relative, digest in members.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source drift: {relative}")
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "qio-c1-exact-seed0-source-v1.tar.gz"
    temporary = archive.with_suffix(".tar.gz.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for relative in sorted(members):
                    path = ROOT / relative
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    temporary.replace(archive)
    integrity = {
        "schema_version": 1,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "payload_file_count": len(members),
        "frozen_manifest_file_count": len(files),
        "frozen_source_manifest_sha256": sha256(MANIFEST),
        "execution_contract_sha256": sha256(CONTRACT),
        "members": members,
    }
    path = OUT / "qio-c1-exact-seed0-source-v1.integrity.json"
    path.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
