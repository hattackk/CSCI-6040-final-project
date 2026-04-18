#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/advbench/harmful_behaviors.csv}"
MODEL_NAME="${MODEL_NAME:-llama3-fitd-local}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}"
  echo "Set DATASET_PATH or place file at data/advbench/harmful_behaviors.csv"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

PYTHONPATH=src "${PYTHON_BIN}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend ollama \
  --model "${MODEL_NAME}" \
  --attack standard \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_ollama_standard"

PYTHONPATH=src "${PYTHON_BIN}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend ollama \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_ollama_fitd"

PYTHONPATH=src "${PYTHON_BIN}" -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend ollama \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense vigilant \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_ollama_fitd_vigilant"

echo "Phase 2 Ollama Llama 3 runs complete. Outputs are under results/${STAMP}_phase2_ollama_*"
