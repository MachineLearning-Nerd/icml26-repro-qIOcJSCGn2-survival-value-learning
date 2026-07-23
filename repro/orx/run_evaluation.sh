#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ -n "${PYTHON:-}" ]]; then
  python_cmd="$PYTHON"
else
  python_cmd=""
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_cmd="$candidate"
      break
    fi
  done
fi

if [[ -z "$python_cmd" ]]; then
  echo "No supported Python interpreter was found" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  "$python_cmd" -m venv .venv
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi

.venv/bin/python -m pip install --disable-pip-version-check \
  numpy==1.26.4 scipy==1.12.0 pytest==8.4.2
.venv/bin/python repro/orx/evaluate_seed.py
