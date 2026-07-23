#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_cmd="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$python_cmd" -m venv .venv
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi

.venv/bin/python -m pip install --disable-pip-version-check \
  numpy==1.26.4 scipy==1.12.0 pytest==8.4.2
.venv/bin/python repro/orx/evaluate_seed.py
