#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# Prefer the project venv; fall back to whatever python is on PATH.
PY_BIN="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || command -v python)"

"${PY_BIN}" -m synthefy_nori.evaluation.cli "$@"
