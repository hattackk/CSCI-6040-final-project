# CSCI/DASC 6040 Final Project

Reproduction project for EMNLP 2025 paper:
`"Foot-In-The-Door": Multi-turn Model Jailbreaking`.

This repository is now set up to run:
1. Phase 1 API baseline (`gpt-4o-mini` by default)
2. Phase 2 local replication (Llama-style local model via Hugging Face or Ollama)
3. Extension experiment with a vigilant system prompt defense

## Assignment Alignment

This scaffold is built to support both acceptable outcomes from the course project:
1. Successful reproduction with at least one extension experiment.
2. Rigorous failed-reproduction analysis with clear evidence and open questions.

## Quick Start

1. Run the cross-platform installer (creates venv, installs deps, installs local extras, and pre-downloads default local model):
```bash
./scripts/install.sh --venv-path /tmp/csci6040-final-venv
```

PowerShell:
```powershell
.\scripts\install.ps1 --venv-path "$env:TEMP\csci6040-final-venv"
```

Command Prompt:
```bat
scripts\install.bat --venv-path "%TEMP%\csci6040-final-venv"
```

2. Optional lighter install (skip local model packages and skip model download):
```bash
./scripts/install.sh --no-local --skip-model-download
```

3. Activate your virtual environment:
```bash
source /tmp/csci6040-final-venv/bin/activate
```

PowerShell:
```powershell
& "$env:TEMP\csci6040-final-venv\Scripts\Activate.ps1"
```

Command Prompt:
```bat
%TEMP%\csci6040-final-venv\Scripts\activate.bat
```

4. Prepare dataset:
- Put AdvBench CSV/JSONL in:
`data/advbench/harmful_behaviors.csv`
- A tiny starter file exists at:
`data/advbench/sample_prompts.csv`

5. Set API key for Phase 1:
```bash
export OPENAI_API_KEY="your_key_here"
```

OpenAI backend safety gate (disabled by default):
```bash
export FITD_ALLOW_OPENAI=1
```
Without this flag, OpenAI runs are blocked in both CLI and dashboard so local testing is the default.

6. Run baseline/extension experiments:
```bash
bash scripts/run_phase1_api_baseline.sh
```

7. Run local replication (Phase 2):
```bash
bash scripts/run_phase2_local_llama3.sh
```

Optional local Ollama path for an already-registered Llama model:
```bash
bash scripts/run_phase2_local_llama3_ollama.sh
```

Optional vLLM path (Docker + NVIDIA GPU; paper-faithful serving):
```bash
cp .env.vllm.example .env           # fill in HF_TOKEN, VLLM_MODEL, VLLM_API_KEY
docker compose up -d vllm           # wait for the weights to load
export VLLM_API_KEY=local-dev-key   # or whatever you set in .env
bash scripts/run_phase2_vllm_llama3.sh
```
The vLLM service listens on `http://localhost:8001/v1` (OpenAI-compatible) and
is selected with `--backend vllm`. Override with `VLLM_BASE_URL` if running the
container on another host.

To swap the served model (one model per container at a time, weights cached):
```bash
bash scripts/vllm_switch_model.sh Qwen/Qwen2-7B-Instruct
```

Run the full matrix (standard / fitd / fitd+vigilant) over multiple models in
one command. Each model is loaded in turn, polled for readiness, then swept:
```bash
bash scripts/run_vllm_model_matrix.sh
# or with an explicit list:
bash scripts/run_vllm_model_matrix.sh Qwen/Qwen2-7B-Instruct Qwen/Qwen1.5-7B-Chat
```
Per-run outputs land under `results/<stamp>_vllm_*` and a manifest of every
run is written to `results/<stamp>_vllm_matrix/manifest.md`.

If you do not want to install editable packages, run with:
```bash
PYTHONPATH=src python -m fitd_repro ...
```

Installer options you will likely use:
1. `--venv-path <path>`: put the virtual environment outside iCloud/OneDrive.
2. `--model <hf-repo>`: repeat to pre-download multiple model repos.
3. `--hf-home <path>`: custom Hugging Face cache location.
4. `--no-local` / `--skip-model-download`: fast setup for dashboard/mock-only work.

## Live Dashboard

Launch the matrix-style UI to run the three conditions and watch live progress:

```bash
bash scripts/run_dashboard.sh
```

Then open:
- `http://127.0.0.1:8787`

The dashboard includes:
1. Live batch and per-condition progress
2. Condition matrix (Standard vs FITD vs FITD+Vigilant)
3. Foot-In-The-Door effect panel (Standard refusal -> FITD success transitions)
4. Turn/event activity trace
5. Prompt/response timeline per run (with optional response reveal toggle)
6. Model history across previous runs
7. FITD variant switch (`scaffold` vs `author` pre-generated prompt chains)
8. Author warmup cap control (`0` = full chain; smaller values speed up local runs)

## Run Commands

Direct CLI entrypoint:
```bash
python -m fitd_repro \
  --dataset-path data/advbench/harmful_behaviors.csv \
  --backend openai \
  --model gpt-4o-mini \
  --attack fitd \
  --defense none \
  --max-examples 50 \
  --output-dir results/phase1_fitd
```

Run with mock backend for local smoke tests:
```bash
python -m fitd_repro \
  --dataset-path data/advbench/sample_prompts.csv \
  --backend mock \
  --model mock-model \
  --attack fitd \
  --output-dir results/mock_fitd
```

Run author prompt-technique mode (pre-generated chains from official repo data):
```bash
python -m fitd_repro \
  --dataset-path data/author_fitd/jailbreakbench.csv \
  --backend hf \
  --model Qwen/Qwen2-7B-Instruct \
  --attack fitd \
  --fitd-variant author \
  --author-prompt-track prompts1 \
  --author-target-mode raw \
  --max-examples 10 \
  --output-dir results/author_fitd_prompts1
```
If your dataset name is not `jailbreakbench` or `harmbench`, provide:
`--author-prompt-file data/author_fitd/prompt_jailbreakbench.json`
Set `--author-max-warmup-turns 0` to use the full author chain, or omit the flag to use the full copied chain by default.

For partner handoff on an NVIDIA GPU, use the prepared runbook and matrix script:

- [docs/partner_gpu_handoff_brief.md](./docs/partner_gpu_handoff_brief.md)
- [docs/gpu_handoff_runbook.md](./docs/gpu_handoff_runbook.md)
- `bash scripts/run_qwen_gpu_author_matrix.sh`

Practical note: the exact `Qwen/Qwen2-7B-Instruct` follow-up is much heavier than the earlier `Qwen/Qwen2.5-3B-Instruct` substitute. On this project machine it ran through the local Hugging Face backend on CPU, not vLLM or Apple GPU.

## Safety Notes

1. Default recommendation: use local `hf` or `mock` backends for FITD experiments.
2. Local `ollama` backend is also supported if you already have a compatible model registered in Ollama.
3. OpenAI backend is intentionally disabled unless `FITD_ALLOW_OPENAI=1` is set.
4. If your course requires API comparisons, keep prompts policy-compliant and document instructor approval.
5. `fitd-variant author` reuses official pre-generated multi-turn prompt chains; this is prompt-technique parity, not the full adaptive pipeline in `FITD.py`.

## Repository Layout

```
CSCI-6040-final-project/
  data/advbench/                 # dataset files
  data/author_fitd/              # copied pre-generated prompt chains from official FITD repo
  docs/milestones.md             # timeline + deliverables checklist
  scripts/                       # reproducible run commands
  src/fitd_repro/                # experiment code
  results/                       # generated outputs (gitignored)
  logs/                          # dashboard run event logs (jsonl, gitignored)
```

## Output Artifacts

Each run writes:
1. `records.jsonl`: per-example conversation + final response + heuristic score
2. `summary.json`: attack success rate (ASR), refusal rate, metadata
3. `turn_events.jsonl`: turn-by-turn user prompt/assistant response log with refusal flags
4. `errors.jsonl`: per-example failures (if any)

Dashboard runs also write:
1. `logs/<run_name>.jsonl`: centralized run lifecycle + event stream for auditing progress over time

## Notes on Metrics

Current scoring uses a refusal-phrase heuristic for fast iteration.
For final report quality, you should add a stronger judge (manual annotation or LLM judge) and report both heuristic and verified ASR.

## Team Workflow Suggestion

1. One person runs batch experiments and stores outputs under `results/`.
2. One person audits failures and validates a sample manually.
3. Both sync findings into the final slides and report narrative.
