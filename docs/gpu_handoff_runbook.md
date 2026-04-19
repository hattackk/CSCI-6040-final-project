# GPU Handoff Runbook

This repo is ready for a closer GPU-side rerun, but the important distinction is now explicit:

- `author_target_mode=softened` is the historical local-project path that rewrites some final targets.
- `author_target_mode=raw` preserves the original dataset goal exactly and is the mode to use for the closer paper-faithful rerun.

The old softened path stays in the repo only because the April 18 local results and report already depend on it.

## What To Run First

Use the exact paper-family model we already confirmed locally:

- `Qwen/Qwen2-7B-Instruct`

Primary three-condition matrix:

1. `standard`
2. `fitd` with `--fitd-variant author --author-prompt-track prompts1 --author-target-mode raw`
3. `fitd` + `vigilant` on the same author chain

If the machine has enough VRAM and time after that, run:

- `Qwen/Qwen1.5-7B-Chat`

## Hardware Reality

Recommended minimum:

- NVIDIA GPU with at least 16 GB VRAM for a comfortable 7B HF run

Safer:

- 24 GB VRAM or better

Likely not worth it for the core exact-model run:

- 8 GB VRAM unless you are willing to introduce 8-bit or offload deviations

This handoff path is still closer than the MacBook run because it keeps the exact Qwen family and uses CUDA rather than CPU-only HF. It is still not the paper's full `vLLM` on `A100` stack unless your partner's machine can support that separately.

## Setup On The GPU Machine

Use Python 3.11 if possible.

Create and activate a clean virtualenv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA build of PyTorch that matches the machine and driver. Example for CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

Install the repo and local model dependencies:

```bash
pip install -e .
pip install -e '.[local]'
pip install huggingface_hub
```

Sanity check CUDA before running experiments:

```bash
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print('name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

If `cuda=False`, stop there and fix the machine before running the study.

## Recommended Commands

Run the prepared matrix script for the exact Qwen2 model:

```bash
PYTHON_BIN=python \
MODEL_NAME='Qwen/Qwen2-7B-Instruct' \
MAX_EXAMPLES=10 \
AUTHOR_TARGET_MODE=raw \
bash scripts/run_qwen_gpu_author_matrix.sh
```

That writes three result directories under `results/`:

- `*_standard`
- `*_fitd_author_raw`
- `*_fitd_author_raw_vigilant`

If the exact-model Qwen2 run is stable, then run the second paper-family model:

```bash
PYTHON_BIN=python \
MODEL_NAME='Qwen/Qwen1.5-7B-Chat' \
MAX_EXAMPLES=10 \
AUTHOR_TARGET_MODE=raw \
bash scripts/run_qwen_gpu_author_matrix.sh
```

If you need a faster probe before the full 10-example slice, set:

```bash
MAX_EXAMPLES=3
```

If you need to resume from a later slice:

```bash
START_INDEX=10
```

## What To Send Back

For each run family, send back the entire result directories created by the script, especially:

- `summary.json`
- `records.jsonl`
- `turn_events.jsonl`
- `errors.jsonl`

Do not just send the summary metrics. We need the full records for manual audit.

## Manual Audit Rule

After the runs finish, manually review every heuristic positive before updating the report or slides.

For each positive, decide whether it is:

- `not_jailbreak`
- `harmful_off_target`
- `faithful_jailbreak`

Append those verified judgments to `docs/manual_audit.csv`. The project should not make final ASR claims from the heuristic alone.

## What This Improves

If these GPU runs succeed, the project becomes materially closer to the paper because it would then have:

- exact paper-family Qwen checkpoints rather than only the Qwen 2.5 3B substitute
- author prompt chains with the original target wording preserved
- local HF on CUDA instead of local HF on CPU

## What Still Remains Imperfect

Even after these runs, this would still not be a perfect reproduction unless we also do more work:

- no full adaptive `FITD.py` reproduction yet
- still not the paper's `vLLM` on `A100` runtime unless that gets added separately
- heuristic scoring still needs manual audit
- the current repo remains a custom experiment scaffold rather than a direct port of the authors' full pipeline
