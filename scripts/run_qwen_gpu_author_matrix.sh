#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-data/author_fitd/jailbreakbench.csv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-7B-Instruct}"
BACKEND="${BACKEND:-hf}"
AUTHOR_PROMPT_TRACK="${AUTHOR_PROMPT_TRACK:-prompts1}"
AUTHOR_TARGET_MODE="${AUTHOR_TARGET_MODE:-raw}"
AUTHOR_MAX_WARMUP_TURNS="${AUTHOR_MAX_WARMUP_TURNS:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
START_INDEX="${START_INDEX:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${DATASET_PATH}" ]]; then
  if [[ -f "${PROJECT_ROOT}/${DATASET_PATH}" ]]; then
    DATASET_PATH="${PROJECT_ROOT}/${DATASET_PATH}"
  else
    echo "Dataset not found at ${DATASET_PATH}"
    exit 1
  fi
fi

MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr '/:' '__' | tr ' ' '_')"
BASE_PREFIX="results/${STAMP}_${MODEL_SLUG}_author_jbb"

run_one() {
  local attack="$1"
  local defense="$2"
  local suffix="$3"

  local cmd=(
    "${PYTHON_BIN}" -m fitd_repro
    --dataset-path "${DATASET_PATH}"
    --backend "${BACKEND}"
    --model "${MODEL_NAME}"
    --attack "${attack}"
    --defense "${defense}"
    --max-examples "${MAX_EXAMPLES}"
    --start-index "${START_INDEX}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --output-dir "${BASE_PREFIX}_${suffix}"
  )

  if [[ "${attack}" == "fitd" ]]; then
    cmd+=(
      --fitd-variant author
      --author-prompt-track "${AUTHOR_PROMPT_TRACK}"
      --author-max-warmup-turns "${AUTHOR_MAX_WARMUP_TURNS}"
      --author-target-mode "${AUTHOR_TARGET_MODE}"
    )
  fi

  echo "[run] ${suffix}"
  "${cmd[@]}"
}

run_one standard none standard
run_one fitd none fitd_author_"${AUTHOR_TARGET_MODE}"
run_one fitd vigilant fitd_author_"${AUTHOR_TARGET_MODE}"_vigilant

echo ""
echo "Completed Qwen author-chain matrix."
echo "Results:"
echo "  ${BASE_PREFIX}_standard"
echo "  ${BASE_PREFIX}_fitd_author_${AUTHOR_TARGET_MODE}"
echo "  ${BASE_PREFIX}_fitd_author_${AUTHOR_TARGET_MODE}_vigilant"
