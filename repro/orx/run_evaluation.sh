#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 2
fi

apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libgl1-mesa-dri >/dev/null

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export MUJOCO_GL=egl

uv sync --frozen --python 3.10
uv run --frozen --no-sync python repro/campaign/diagnose_smoke.py
