#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/install.py" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "${SCRIPT_DIR}/install.py" "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python "${SCRIPT_DIR}/install.py" "$@"
fi

echo "Python 3.10+ is required but was not found in PATH." >&2
exit 1
