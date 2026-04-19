#!/usr/bin/env bash
# Full experiment matrix over multiple models via the local vLLM container.
#
# For each model in the list:
#   1. Stop + restart the vllm container with that model (cached after first load).
#   2. Wait until /v1/models reports the model as ready.
#   3. Run the three experiment conditions: standard, fitd, fitd+vigilant.
#
# Models can be supplied in three ways (first wins):
#   1. Positional args:  bash scripts/run_vllm_model_matrix.sh model_a model_b
#   2. MODELS env var:   MODELS="model_a model_b" bash scripts/run_vllm_model_matrix.sh
#   3. Built-in default list (paper's open-source targets).
#
# Pre-reqs:
#   - .env populated from .env.vllm.example (HF_TOKEN, VLLM_API_KEY, ...)
#   - docker + NVIDIA runtime available
#   - VLLM_API_KEY exported in the current shell so the Python client can auth

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_MODELS=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen2-7B-Instruct"
  "Qwen/Qwen1.5-7B-Chat"
  "mistralai/Mistral-7B-Instruct-v0.2"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  read -ra MODELS <<< "${MODELS}"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

DATASET_PATH="${DATASET_PATH:-data/advbench/harmful_behaviors.csv}"
MAX_EXAMPLES="${MAX_EXAMPLES:-25}"
: "${VLLM_HOST_PORT:=8001}"
: "${VLLM_BASE_URL:=http://localhost:${VLLM_HOST_PORT}/v1}"
: "${VLLM_API_KEY:=EMPTY}"
: "${VLLM_READY_TIMEOUT:=900}"   # seconds; first-time weight downloads can take a while
export VLLM_BASE_URL VLLM_API_KEY

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1 ; then
  echo "docker CLI not found in PATH."
  exit 1
fi

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
MODELS_URL="${VLLM_BASE_URL}/models"

wait_for_vllm() {
  local target="$1"
  local deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
  echo "Waiting up to ${VLLM_READY_TIMEOUT}s for vLLM to serve ${target}..."
  while [[ $(date +%s) -lt ${deadline} ]]; do
    if curl -s -f -o /dev/null "${HEALTH_URL}" ; then
      local body
      body="$(curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "${MODELS_URL}" || true)"
      if echo "${body}" | grep -Fq "\"id\":\"${target}\"" ; then
        echo "vLLM ready with model: ${target}"
        return 0
      fi
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM to serve ${target}"
  return 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="results/${RUN_STAMP}_vllm_matrix"
mkdir -p "${SUMMARY_DIR}"
MANIFEST="${SUMMARY_DIR}/manifest.md"

{
  echo "# vLLM Matrix Run: ${RUN_STAMP}"
  echo ""
  echo "Dataset: \`${DATASET_PATH}\` (first ${MAX_EXAMPLES} examples)"
  echo ""
  echo "| Model | Condition | Output |"
  echo "|---|---|---|"
} > "${MANIFEST}"

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(echo "${MODEL}" | tr '/:' '__')"
  echo ""
  echo "=============================================="
  echo "Switching vLLM to: ${MODEL}"
  echo "=============================================="

  if ! bash scripts/vllm_switch_model.sh "${MODEL}" ; then
    echo "| \`${MODEL}\` | compose-up-failed | - |" >> "${MANIFEST}"
    continue
  fi

  if ! wait_for_vllm "${MODEL}" ; then
    echo "Skipping ${MODEL} (startup/ready check failed). See: docker compose logs vllm"
    echo "| \`${MODEL}\` | startup-failed | - |" >> "${MANIFEST}"
    continue
  fi

  for COND in standard fitd fitd_vigilant ; do
    case "${COND}" in
      standard)       ATTACK=standard; DEFENSE=none ;;
      fitd)           ATTACK=fitd;     DEFENSE=none ;;
      fitd_vigilant)  ATTACK=fitd;     DEFENSE=vigilant ;;
    esac
    OUT_DIR="results/${RUN_STAMP}_vllm_${MODEL_SLUG}_${COND}"
    echo "---- ${MODEL} / ${COND} -> ${OUT_DIR}"
    if "${PYTHON_CMD[@]}" -m fitd_repro \
        --dataset-path "${DATASET_PATH}" \
        --backend vllm \
        --model "${MODEL}" \
        --attack "${ATTACK}" \
        --defense "${DEFENSE}" \
        --max-examples "${MAX_EXAMPLES}" \
        --output-dir "${OUT_DIR}" ; then
      echo "| \`${MODEL}\` | ${COND} | \`${OUT_DIR}\` |" >> "${MANIFEST}"
    else
      echo "Run failed: ${MODEL} / ${COND}"
      echo "| \`${MODEL}\` | ${COND} | run-failed |" >> "${MANIFEST}"
    fi
  done
done

echo ""
echo "=============================================="
echo "Matrix run complete."
echo "Manifest: ${MANIFEST}"
echo "Per-run outputs: results/${RUN_STAMP}_vllm_*"
