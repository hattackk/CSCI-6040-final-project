#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command> [args...]"
  echo "Example:"
  echo "  $0 python -m fitd_repro --dataset-path data/advbench/sample_prompts.csv --backend mock --model mock-model --attack fitd --output-dir results/demo"
  exit 1
fi

VENV_DIR="${VENV_DIR:-/tmp/csci6040-final-venv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
