#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/advbench/harmful_behaviors.csv}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.2}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}"
  echo "Set DATASET_PATH or place file at data/advbench/harmful_behaviors.csv"
  exit 1
fi

if [[ "${FITD_ALLOW_OPENAI:-0}" != "1" ]]; then
  echo "OpenAI runs are disabled by default."
  echo "Set FITD_ALLOW_OPENAI=1 only after project approval for policy-compliant testing."
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend openai \
  --model "${MODEL_NAME}" \
  --attack standard \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --sleep-seconds "${SLEEP_SECONDS}" \
  --output-dir "results/${STAMP}_phase1_standard"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend openai \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --sleep-seconds "${SLEEP_SECONDS}" \
  --output-dir "results/${STAMP}_phase1_fitd"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend openai \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense vigilant \
  --max-examples "${MAX_EXAMPLES}" \
  --sleep-seconds "${SLEEP_SECONDS}" \
  --output-dir "results/${STAMP}_phase1_fitd_vigilant"

echo "Phase 1 runs complete. Outputs are under results/${STAMP}_*"
