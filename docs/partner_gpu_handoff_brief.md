# Partner GPU Handoff Brief

## Objective

We need one closer paper-faithful Qwen rerun to finish the project in a defensible way.

The core gap in the current project is that our earlier Qwen runs used `Qwen/Qwen2.5-3B-Instruct`, which is not one of the paper's Qwen models. We already fixed that locally once for `Qwen/Qwen2-7B-Instruct`, but that run was CPU-only and still used the project's historical softened final target in author mode.

The purpose of this handoff is to run the exact paper-family Qwen model on GPU using the copied author prompt chain while preserving the original target wording.

## Required Result

Run this exact three-condition matrix:

1. `standard`
2. `FITD` with:
   - `--fitd-variant author`
   - `--author-prompt-track prompts1`
   - `--author-target-mode raw`
3. `FITD + vigilant` with the same author-chain settings

Use:

- model: `Qwen/Qwen2-7B-Instruct`
- dataset: `data/author_fitd/jailbreakbench.csv`
- slice: first `10` examples
- backend: `hf`

This is the only required GPU experiment for project completion.

## Exact Command

From the repo root:

```bash
PYTHON_BIN=python \
MODEL_NAME='Qwen/Qwen2-7B-Instruct' \
MAX_EXAMPLES=10 \
AUTHOR_TARGET_MODE=raw \
bash scripts/run_qwen_gpu_author_matrix.sh
```

Before that, verify the machine actually sees CUDA:

```bash
nvidia-smi
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print('name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

If CUDA is not available, stop and report that rather than trying to force a CPU run.

## What We Are Trying To Learn

We are not trying to prove the paper right or wrong in general.

We are trying to answer this narrower question:

> When we switch from the substitute Qwen 2.5 3B model to the paper-family `Qwen/Qwen2-7B-Instruct`, and preserve the original target wording in the author-chain path, do we still fail to reproduce the paper's claimed FITD effect on the tested slice?

That result is what we need to finalize the report and slides honestly.

## What Not To Spend Time On

Do not spend time on these unless the required run is already complete and stable:

- `Qwen/Qwen2.5-3B-Instruct`
- Ollama or Llama runs
- scaffold-only FITD runs as the main experiment
- `author_target_mode=softened`
- large full-dataset sweeps
- `prompts2`
- `Qwen/Qwen1.5-7B-Chat`
- `vLLM` setup

Those are optional follow-ons, not the blocker.

## What To Send Back

Send back the full result directories created by the script, not just screenshots or summary numbers.

We specifically need:

- `summary.json`
- `records.jsonl`
- `turn_events.jsonl`
- `errors.jsonl`

If the run fails, send back:

- the command used
- the error text
- whether the failure was download/auth, CUDA visibility, out-of-memory, or runtime crash

## Completion Rule

This handoff is complete when all three `Qwen/Qwen2-7B-Instruct` conditions finish on the first 10 examples and the full output directories are returned.

Once we have that, we can:

1. manually audit every heuristic positive
2. update the report/slides/script one final time
3. rerun `pytest -q`
4. submit

## Supporting Docs

For setup details, use:

- [gpu_handoff_runbook.md](./gpu_handoff_runbook.md)
- [README.md](../README.md)

For the actual run entrypoint, use:

- [run_qwen_gpu_author_matrix.sh](../scripts/run_qwen_gpu_author_matrix.sh)
