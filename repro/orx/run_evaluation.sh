#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ -z "${HF_JOB_ID:-}" ]]; then
  echo "This campaign runner is authorized only inside a Hugging Face Job" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 2
fi

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export MUJOCO_GL=egl

uv sync --frozen --python 3.10
uv run --frozen --no-sync python repro/campaign/run_preflight.py
