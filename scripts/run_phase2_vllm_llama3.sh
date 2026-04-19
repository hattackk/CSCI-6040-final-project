#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Runs the three condition matrix (standard / fitd / fitd+vigilant) against a
# vLLM server launched via `docker compose up -d vllm`.
#
# Pre-reqs:
#   1. Copy .env.vllm.example to .env and fill in HF_TOKEN, VLLM_MODEL, VLLM_API_KEY.
#   2. docker compose up -d vllm  (wait for the model to finish loading)
#   3. Set VLLM_API_KEY in your shell so the client can authenticate.

DATASET_PATH="${DATASET_PATH:-data/advbench/harmful_behaviors.csv}"
MODEL_NAME="${MODEL_NAME:-${VLLM_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-25}"

: "${VLLM_BASE_URL:=http://localhost:${VLLM_HOST_PORT:-8001}/v1}"
: "${VLLM_API_KEY:=EMPTY}"
export VLLM_BASE_URL VLLM_API_KEY

PYTHON_CMD=()
resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
  elif [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("${PYTHON}")
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_CMD=("${VIRTUAL_ENV}/bin/python")
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/Scripts/python.exe" ]]; then
    PYTHON_CMD=("${VIRTUAL_ENV}/Scripts/python.exe")
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${PROJECT_ROOT}/.venv/bin/python")
  elif [[ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON_CMD=("${PROJECT_ROOT}/.venv/Scripts/python.exe")
  elif command -v python3 >/dev/null 2>&1 ; then
    PYTHON_CMD=("python3")
  elif command -v python >/dev/null 2>&1 ; then
    PYTHON_CMD=("python")
  elif command -v py >/dev/null 2>&1 ; then
    PYTHON_CMD=("py" "-3")
  else
    echo "No Python interpreter found. Activate the project venv or set PYTHON_BIN=/path/to/python."
    exit 1
  fi
}

resolve_python

PYTHONPATH_SRC="${PROJECT_ROOT}/src"
PYTHONPATH_SEP=":"
if [[ "${PYTHON_CMD[0]}" == *.exe || "${PYTHON_CMD[0]}" == "py" ]]; then
  if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    export WSLENV="${WSLENV:+${WSLENV}:}PYTHONPATH/lp"
  else
    PYTHONPATH_SEP=";"
  fi
  if [[ -z "${WSL_DISTRO_NAME:-}" ]] && command -v cygpath >/dev/null 2>&1 ; then
    PYTHONPATH_SRC="$(cygpath -w "${PYTHONPATH_SRC}")"
  fi
fi
export PYTHONPATH="${PYTHONPATH_SRC}${PYTHONPATH:+${PYTHONPATH_SEP}${PYTHONPATH}}"

echo "Using Python: ${PYTHON_CMD[*]}"

HEALTH_URL="${VLLM_BASE_URL%/v1}/health"
if ! curl -s -f -o /dev/null "${HEALTH_URL}"; then
  echo "vLLM server not reachable at ${HEALTH_URL}"
  echo "Start it with: docker compose up -d vllm"
  echo "First load can take several minutes while weights download."
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}"
  echo "Set DATASET_PATH or place file at data/advbench/harmful_behaviors.csv"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

"${PYTHON_CMD[@]}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend vllm \
  --model "${MODEL_NAME}" \
  --attack standard \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_vllm_standard"

"${PYTHON_CMD[@]}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend vllm \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_vllm_fitd"

"${PYTHON_CMD[@]}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend vllm \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense vigilant \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_vllm_fitd_vigilant"

echo "vLLM runs complete. Outputs are under results/${STAMP}_vllm_*"
