#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
# First run downloads llama-server if missing (LLAMA_BACKEND=auto).
# LLAMA_SKIP_INSTALL=1 skips the download.

if [[ ! -x venv/bin/python ]]; then
  echo "Creating venv..."
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

exec python launch.py "$@"
