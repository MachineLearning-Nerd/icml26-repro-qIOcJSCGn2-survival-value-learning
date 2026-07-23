# qIO full-agent Colab CUDA handoff

## Decision

Use Colab for the released full-agent execution gate. The local queue remains preserved and still
waits on its exact CPU blocker; this handoff neither stops nor replaces that process. If FluxNet is
using the same Colab runtime, finish and save the FluxNet artifacts first. Otherwise this qIO gate
can run now in a separate GPU runtime.

The handoff first runs 10 updates and then resumes the exact checkpoint to 100 updates. This makes
resume behavior observed evidence rather than a promise. It is still only an execution,
checkpoint, and exact-reload gate. It does **not** establish the paper's 1,000,000-update Claim 3.

## Frozen upload

- Archive: `outputs/colab_handoff/qio-full-agent-colab-input-v1.tar.gz`
- Bytes: `22,528,563` (about 21.5 MiB)
- SHA-256: `d7f8e07b72f828d8f39100533510a60858dab8b7cc7debcc2e2409260da634ee`
- Integrity sidecar:
  `outputs/colab_handoff/qio-full-agent-colab-input-v1.tar.gz.integrity.json`
- Bundle-manifest SHA-256:
  `d2e5706fe5c8f12d904234fd6b23b624624501da05e60954b291372c3e67fa42`
- Payload: the two exact official PointMaze files, all 18 official executable/config Git blobs,
  author lockfile, frozen source manifest, CPU reference runner/auditor, CUDA runner, and return
  packer. The archive has 26 declared payload members plus its manifest.

The archive is byte-deterministic: rebuilding it twice from the current frozen inputs produced the
same SHA-256 above.

## Colab procedure

Select a GPU runtime. T4 is sufficient for the 100-update feasibility gate; L4 or A100 should be
faster. Upload both the archive and its integrity sidecar to `/content`, then run the following
cells. Keep the output directory on Google Drive so a runtime disconnect cannot erase checkpoints.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path
import hashlib, json, tarfile

archive = Path("/content/qio-full-agent-colab-input-v1.tar.gz")
sidecar = Path(str(archive) + ".integrity.json")
integrity = json.loads(sidecar.read_text())
h = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
assert h.hexdigest() == integrity["archive_sha256"]
root = Path("/content/qio-full-agent")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "r:gz") as bundle:
    bundle.extractall(root, filter="data")
print(integrity)
```

```python
!pip -q install uv
!uv python install 3.10
!cd /content/qio-full-agent && uv sync --frozen --python 3.10 --project upstream
```

The environment comes from the authors' frozen `uv.lock`. Do not substitute the Colab system JAX
or upgrade individual dependencies after this cell.

```python
from pathlib import Path
import os, subprocess

root = Path("/content/qio-full-agent")
python = root / "upstream/.venv/bin/python"
runner = root / "repro/colab/run_full_agent_probe_colab.py"
archive = Path("/content/qio-full-agent-colab-input-v1.tar.gz")
sidecar = Path(str(archive) + ".integrity.json")
output = Path("/content/drive/MyDrive/ICMLPapers/qio-full-agent-colab/seed-0")
env = dict(os.environ)
env["JAX_PLATFORM_NAME"] = "gpu"
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

common = [
    str(python), str(runner),
    "--input-archive", str(archive),
    "--input-integrity", str(sidecar),
    "--seed", "0",
    "--log-interval", "10",
    "--output-dir", str(output),
]

# Phase 1: create a real, validated generation at update 10.
subprocess.run(
    common + ["--updates", "10", "--checkpoint-interval", "10"],
    cwd=root, env=env, check=True,
)
```

```python
# Phase 2: a separate process must restore update 10 and extend it to 100.
subprocess.run(
    common + ["--updates", "100", "--checkpoint-interval", "50"],
    cwd=root, env=env, check=True,
)
```

If Colab disconnects during phase 2, rerun only the phase-2 cell. The runner first rehashes the
input archive, every extracted input, the official source manifest, both datasets, the protocol,
and the active checkpoint. A mismatch fails before checkpoint reuse.

Package the return only after phase 2 completes:

```python
packer = root / "repro/colab/package_colab_return.py"
returned = Path(
    "/content/drive/MyDrive/ICMLPapers/qio-full-agent-colab/"
    "qio-full-agent-colab-return-v1.tar.gz"
)
subprocess.run(
    [
        str(python), str(packer),
        "--output-dir", str(output),
        "--archive", str(returned),
    ],
    cwd=root, env=env, check=True,
)
print(returned)
print(Path(str(returned) + ".integrity.json"))
```

Return both the `.tar.gz` and `.integrity.json` files. The archive contains the active checkpoint,
raw metric trace, exact command and environment ledger, GPU identity, session chain, protocol,
source/config/lock provenance, and every relevant hash. It excludes the 21 MiB datasets because
their exact input-archive and per-dataset identities are already bound.

## Independent local verification

After copying both return files to this paper directory, run the standalone verifier (this is an
artifact audit, not a test suite):

```bash
python3 repro/src/verify_colab_full_agent_return.py \
  /path/to/qio-full-agent-colab-return-v1.tar.gz \
  --integrity /path/to/qio-full-agent-colab-return-v1.tar.gz.integrity.json
```

It does not import JAX, extract files, or unpickle the checkpoint. It independently checks the
return archive hash and inventory, exact active-checkpoint bytes, source commit and Git-blob audit,
dataset identities, canonical protocol, raw finite metrics, command/GPU ledger, and the exact
cross-invocation resume chain.

## Runtime and storage expectation

- Upload: 21.5 MiB.
- Dependency installation: typically 10–20 minutes on a fresh Colab runtime; allow roughly 6–8 GiB
  of ephemeral disk for the pinned Python/CUDA environment.
- Gate: budget 20–45 minutes total on T4, including two separate JAX compilations. L4/A100 should
  be faster. This is an estimate until the returned ledger records measured throughput.
- Drive: reserve 2 GiB for update-10, update-50, and update-100 immutable checkpoint generations
  plus the returned archive. The exact checkpoint and archive sizes are recorded on completion.

Do not start a four-seed, 1,000,000-update campaign from this estimate. Use the returned steady-state
throughput and GPU-memory evidence to decide its hardware/session plan first.
