#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_PATH="${DATASET_PATH:-data/author_fitd/jailbreakbench.csv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
BACKEND="${BACKEND:-hf}"
AUTHOR_PROMPT_TRACK="${AUTHOR_PROMPT_TRACK:-prompts1}"
AUTHOR_PROMPT_FILE="${AUTHOR_PROMPT_FILE:-}"
AUTHOR_MAX_WARMUP_TURNS="${AUTHOR_MAX_WARMUP_TURNS:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  if [[ -f "${PROJECT_ROOT}/${DATASET_PATH}" ]]; then
    DATASET_PATH="${PROJECT_ROOT}/${DATASET_PATH}"
  else
    echo "Dataset not found at ${DATASET_PATH}"
    exit 1
  fi
fi

if [[ -n "${AUTHOR_PROMPT_FILE}" && ! -f "${AUTHOR_PROMPT_FILE}" ]]; then
  if [[ -f "${PROJECT_ROOT}/${AUTHOR_PROMPT_FILE}" ]]; then
    AUTHOR_PROMPT_FILE="${PROJECT_ROOT}/${AUTHOR_PROMPT_FILE}"
  else
    echo "Author prompt file not found at ${AUTHOR_PROMPT_FILE}"
    exit 1
  fi
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

CMD=(
  python -m fitd_repro
  --dataset-path "${DATASET_PATH}"
  --backend "${BACKEND}"
  --model "${MODEL_NAME}"
  --attack fitd
  --defense none
  --fitd-variant author
  --author-prompt-track "${AUTHOR_PROMPT_TRACK}"
  --author-max-warmup-turns "${AUTHOR_MAX_WARMUP_TURNS}"
  --max-examples "${MAX_EXAMPLES}"
  --output-dir "results/${STAMP}_author_fitd_${AUTHOR_PROMPT_TRACK}"
)

if [[ -n "${AUTHOR_PROMPT_FILE}" ]]; then
  CMD+=(--author-prompt-file "${AUTHOR_PROMPT_FILE}")
fi

"${CMD[@]}"

echo "Author prompt-technique run complete. Output: results/${STAMP}_author_fitd_${AUTHOR_PROMPT_TRACK}"
