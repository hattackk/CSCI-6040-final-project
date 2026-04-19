# Plan: P0 (judge upgrade) + P1 (assistant abstraction)

Scope: implement the two highest-priority items from [TODO.md](../TODO.md) so the
reproduction stops scoring jailbreaks with a keyword heuristic and stops
collapsing the paper's two-model design (target `T` + assistant `M`) into one.

References to the paper point at the local copy at
`C:/Users/chris/Downloads/foot-in-the-door.txt`. Section numbers (e.g. §A.3.3)
match the arXiv v3 PDF.

---

## Design constraints

1. **Backend-agnostic judging and assisting.** No code path may *require*
   OpenAI. A user running `--backend vllm` must be able to run the full
   pipeline with vLLM-served judge and assistant, or with the target model
   re-used for both, or any mix. `OPENAI_API_KEY` is only needed when the
   user explicitly opts into OpenAI for one of the three roles.
2. **Paper-faithful where free, deviation-flagged where not.** When a run
   uses anything other than the paper's defaults
   (target = LLaMA-3-8B-Instruct via vLLM, assistant `M` = `gpt-4o-mini`,
   judge = `gpt-4o` with the JBB §A.3.3 prompt), `summary.json` records the
   deviation explicitly so reviewers can audit faithfulness without rerunning.
3. **Dashboard contract preserved.** Existing
   [dashboard_server.py](../src/fitd_repro/dashboard_server.py) reads
   `evaluation.success` and `evaluation.refusal` (lines 195-196). Both keys
   stay populated. New judge fields are additive.
4. **No silent goal rewrite.** P0 does not touch `_author_target_sentence`
   (that's P3). It does, however, mean the new judge will see the *raw*
   response against the *original* goal — exactly the comparison the
   keyword heuristic was hiding.

---

## Faithfulness scorecard (target end-state of this plan)

| Aspect | Paper | Plan v2 |
|---|---|---|
| Judge model default | `gpt-4o` (§4.1) | configurable; `gpt-4o` only if `--judge-backend openai` |
| Judge prompt for ASR | §A.3.3, binary `safe`/`unsafe` | §A.3.3 verbatim |
| Per-turn harmfulness 1-5 | §A.3.6, used for Figure 4b | optional via `--score-harmfulness` |
| Assistant `M` default | `gpt-4o-mini` (§4.1) | configurable; `gpt-4o-mini` only if `--assistant-backend openai` |
| SSP / `get_mid` prompt | §A.3.4 verbatim | §A.3.4 verbatim |
| Re-Align / `p_align` prompt | §A.3.5 verbatim | §A.3.5 verbatim |
| `get_level_query`, `is_align`, `is_rejection`, `paraphrase` | not in paper | from official repo if findable, else paper-consistent + flagged |
| 3 attempts per query | §4.1 | `summarize_records` made attempt-aware now; loop wiring is P2 |
| Per-turn judge calls | implicit for Fig 4b | `--judge-scope {final,all-turns}` |

---

## P0 — JailbreakBench GPT-4o judge

### Architecture

- New module `src/fitd_repro/judge.py`:
  - `class Judge(ABC)` with two methods:
    - `score_alignment(goal: str, response: str) -> {label: "safe"|"unsafe", raw: str}` — runs the §A.3.3 prompt; this is what feeds ASR.
    - `score_harmfulness(goal: str, response: str) -> {harm: 1..5, reason: str, raw: str}` — runs the §A.3.6 prompt; only called when `--score-harmfulness` is set.
  - `class LLMJudge(Judge)` composes a `ChatModel` (built via existing
    `build_model`) and adds the prompt templates. Single class for every
    backend — vLLM, OpenAI, Ollama, HF, mock — because it's just a chat call.
  - `class MockJudge(Judge)` returns canned values for tests.
  - `build_judge(backend, model_name) -> Judge` — mirror of `build_model`
    that returns `MockJudge` for `backend=mock` and `LLMJudge(build_model(...))`
    otherwise.

- The §A.3.3 prompt is pasted verbatim from
  `foot-in-the-door.txt:856-900` with a `# SOURCE: Weng et al. 2025, Appendix A.3.3`
  comment. Same for §A.3.6 (`foot-in-the-door.txt:918-1040`). The model's
  reply is parsed with a strict head-of-string match: `safe` vs `unsafe` for
  A.3.3, JSON-extract-then-clamp for A.3.6's 1-5.

### Refactor of `evaluation.py`

Current `evaluate_response(response)` becomes:

```python
def evaluate_response(
    response: str,
    *,
    goal: str | None = None,
    judge: Judge | None = None,
    score_harmfulness: bool = False,
) -> dict:
    ...
```

- Always computes heuristic `refusal` + `response_chars`.
- If `judge` and `goal` are set: calls `judge.score_alignment`. `success` is
  driven by `judge_label == "unsafe"` (paper-faithful). The heuristic
  `success` is preserved as `success_heuristic`.
- If `score_harmfulness`: also calls `judge.score_harmfulness`, adds
  `judge_harm` and `judge_harm_reason`.
- Judge-absent path: behaves exactly like today (so existing tests pass and
  the per-turn `_log_turn_event` calls stay cheap).

`summarize_records` gains:

- `mean_harm` and `harm_score_count` when any record carries `judge_harm`.
- `judge_asr` (`unsafe / total`) reported alongside heuristic `asr` when any
  record carries `judge_label`. Heuristic `asr` is kept under that name for
  backwards compatibility — no dashboard breakage.
- Becomes attempt-aware: if a record carries `attempt_index`, group by
  `example_id` and count an example as `unsafe` if **any** attempt was
  `unsafe` (paper §4.1 "three attempts per query"). This is dormant until P2
  wires multiple attempts but lands now so summary code doesn't churn twice.

### Runner wiring ([runner.py](../src/fitd_repro/runner.py))

- `run_experiment` gains:
  - `judge_backend: str`, `judge_model: str | None`
  - `score_harmfulness: bool`
  - `judge_scope: Literal["final", "all-turns"]`
- Build judge **once per run** at the same point as the target model
  (line 220) via `build_judge(...)`.
- Final-response evaluation (line 147) passes `judge=judge, goal=example.goal`.
- Per-turn `turn_logger` (line 268) only passes the judge when
  `judge_scope == "all-turns"`. Default stays `final` because per-turn judging
  on a 12-turn run with `gpt-4o` adds ~13× judge cost per example.
- The `summary.update({...})` block (line 372) records:
  - `judge_backend`, `judge_model`
  - `judge_scope`
  - `judge_paper_faithful: bool` (`True` iff `judge_backend == "openai" and judge_model == "gpt-4o"`)
  - `judge_deviation: str | None` (e.g. `"judge served by vllm/<model>; paper used gpt-4o"`)
  - `score_harmfulness: bool`

### Retro-score CLI

- New `src/fitd_repro/rescore.py`, exposed as `python -m fitd_repro.rescore <run-dir>`.
- Iterates `records.jsonl`, calls the judge per record, writes back enriched
  records and a refreshed `summary.json`. Skips records that already have
  `judge_label` so it's idempotent.
- Honors the same `--judge-backend` / `--judge-model` flags as the runner.

### Tests

New fixture `FakeJudge` in `tests/conftest.py` that maps `(goal, response)`
to canned `(label, harm)` pairs. Add to
[tests/test_evaluation.py](../tests/test_evaluation.py):

- `test_evaluate_response_runs_judge_when_provided` — judge fields present.
- `test_evaluate_response_judge_overrides_heuristic_success` —
  `unsafe` flips `success=True` even when the heuristic refused; vice-versa.
- `test_evaluate_response_score_harmfulness_off_by_default` — A.3.6 not called.
- `test_summarize_records_reports_judge_asr_when_present`.
- `test_summarize_records_reports_mean_harm_when_present`.
- `test_summarize_records_groups_by_example_id_when_attempt_index_present`.

Add `tests/test_judge.py`:

- A.3.3 parser handles `unsafe`, `Unsafe.`, `unsafe — because…`,
  `'safe'`, leading whitespace, and falls back safely on garbled output.
- A.3.6 parser extracts `score` from JSON, clamps to `[1, 5]`, falls back
  to 3 + flags `parse_error` when malformed.

No live API calls in any test.

### CLI additions ([cli.py](../src/fitd_repro/cli.py))

```
--judge-backend       {mock,openai,hf,vllm,ollama}    default: same as --backend
--judge-model         <model id>                       default: see Defaulting below
--score-harmfulness                                    flag, default off
--judge-scope         {final,all-turns}                default: final
```

**Defaulting rule** (applied in `cli.py` after argparse, before
`run_experiment`):

| `--judge-backend` resolved to | default `--judge-model` |
|---|---|
| `openai` | `gpt-4o` (paper-faithful) |
| `vllm`, `hf`, `ollama` | falls back to `--model` (re-use the target model as the judge); print a stderr warning |
| `mock` | `mock-judge` |

The "re-use the target model as judge" default is a deviation, but it lets
`docker compose up vllm` + `python -m fitd_repro --backend vllm --model X`
work end-to-end with no extra config. The deviation is recorded in
`summary.json`.

### docker-compose additions ([docker-compose.yml](../docker-compose.yml))

Add an optional second vLLM service on a separate port, behind a profile so
single-model runs don't pay the GPU cost:

```yaml
  vllm-judge:
    profiles: ["judge"]
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - NVIDIA_VISIBLE_DEVICES=${VLLM_JUDGE_CUDA_DEVICE:-1}
      - CUDA_VISIBLE_DEVICES=${VLLM_JUDGE_CUDA_DEVICE:-1}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "${VLLM_JUDGE_HOST_PORT:-8002}:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    command: >
      --model ${VLLM_JUDGE_MODEL}
      --dtype half
      --api-key ${VLLM_JUDGE_API_KEY}
      --gpu-memory-utilization ${VLLM_JUDGE_GPU_MEM_UTIL:-0.85}
      --max-model-len ${VLLM_JUDGE_MAX_MODEL_LEN:-4768}
```

Bring it up with `docker compose --profile judge up -d vllm vllm-judge`.

`VLLMChatModel` (`src/fitd_repro/models.py:173`) currently reads
`VLLM_BASE_URL` / `VLLM_API_KEY` unconditionally. Extend it so the role
(target / judge / assistant) can override via constructor args, with
environment fallbacks:

- `target` role: `VLLM_BASE_URL`, `VLLM_API_KEY` (unchanged)
- `judge` role: `JUDGE_VLLM_BASE_URL` (default
  `http://localhost:${VLLM_JUDGE_HOST_PORT:-8002}/v1`), `JUDGE_VLLM_API_KEY`
- `assistant` role: `ASSISTANT_VLLM_BASE_URL`, `ASSISTANT_VLLM_API_KEY`

`build_judge` and `build_assistant` pass their role to `build_model`, which
threads it into `VLLMChatModel(role="judge")`. Same role plumbing makes
sense for `OllamaChatModel` (env vars `JUDGE_OLLAMA_HOST`,
`ASSISTANT_OLLAMA_HOST`) so a user can keep multiple Ollama daemons on
different ports.

If the user prefers single-container reuse (one model serves all three
roles), they leave the role-specific env vars unset and everything points
at `VLLM_BASE_URL`.

### New env vars (summary)

```
# OpenAI roles (only needed if you opt that role into the openai backend)
OPENAI_API_KEY
FITD_ALLOW_OPENAI=1            # existing gate

# vLLM judge (only needed when --judge-backend vllm with a separate container)
VLLM_JUDGE_MODEL               # e.g. meta-llama/Meta-Llama-Guard-3-8B
VLLM_JUDGE_API_KEY             # any non-empty token; matches container --api-key
VLLM_JUDGE_HOST_PORT=8002
VLLM_JUDGE_CUDA_DEVICE=1
VLLM_JUDGE_GPU_MEM_UTIL=0.85
VLLM_JUDGE_MAX_MODEL_LEN=4768
JUDGE_VLLM_BASE_URL            # auto-derived if unset
JUDGE_VLLM_API_KEY             # auto-derived if unset

# vLLM assistant (only needed when --assistant-backend vllm with a separate container)
VLLM_ASSISTANT_MODEL
VLLM_ASSISTANT_API_KEY
VLLM_ASSISTANT_HOST_PORT=8003
VLLM_ASSISTANT_CUDA_DEVICE=2
ASSISTANT_VLLM_BASE_URL
ASSISTANT_VLLM_API_KEY
```

`.env.example` (new file) ships these with comments so the user knows which
to set per scenario.

### Acceptance criteria

- `pytest` passes; new judge tests cover both OpenAI-style and locally-served
  paths via `MockJudge` (no network).
- `python -m fitd_repro --backend mock --model mock-model ...` produces a
  run identical to today (judge defaults to `mock`, no fields drop).
- `python -m fitd_repro --backend vllm --model X ...` runs end-to-end with a
  single vLLM container — judge re-uses the target model and the deviation
  is logged.
- `python -m fitd_repro --backend vllm --model X --judge-backend openai
  --judge-model gpt-4o ...` runs the paper-faithful judge against a
  vLLM-served target.
- `summary.json` includes `judge_backend`, `judge_model`, `judge_asr` (when
  judge ran), `judge_paper_faithful`, `judge_deviation`.
- `python -m fitd_repro.rescore <old-run-dir>` upgrades pre-judge runs in
  place.
- Dashboard at `/run/<id>` still loads against both new and old runs.

---

## P1 — `AssistantModel` abstraction

P1 lands the seam and the helpers. P2 (separate task) is what consumes them
inside the FITD loop.

### Architecture

- New module `src/fitd_repro/assistant.py`:
  - `class AssistantModel(ABC)` with the six methods named in
    [TODO.md](../TODO.md):
    - `get_level_query(n: int, q: str) -> list[str]`
    - `get_mid(q_last: str, q_i: str) -> str`     # §A.3.4 SSP prompt
    - `is_align(q: str, r: str) -> bool`
    - `is_rejection(r: str) -> bool`
    - `get_align_prompt(q_last: str, r_last: str) -> str`   # §A.3.5
    - `paraphrase(q_mid: str, history: list[Message]) -> str`
  - `class LLMAssistant(AssistantModel)` composes a `ChatModel` and adds
    each prompt template.
  - `class FakeAssistant(AssistantModel)` for tests: deterministic
    rule-based outputs.
  - `build_assistant(backend, model_name) -> AssistantModel` mirrors
    `build_judge`.

### Prompt sources

| Helper | Source | Status |
|---|---|---|
| `get_mid` | §A.3.4 verbatim (`foot-in-the-door.txt:907-940`) | paper-exact |
| `get_align_prompt` | §A.3.5 verbatim (`foot-in-the-door.txt:946-984`) | paper-exact |
| `get_level_query` | not in paper | pull from [official repo](https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak); if not findable, derive from §3.3 prose and tag `# NOT FROM PAPER — derived from §3.3` |
| `is_align` | not in paper | same — repo first, derived second |
| `is_rejection` | not in paper | same |
| `paraphrase` | not in paper | same |

`summary.json` reports `assistant_prompts_source: {"get_level_query": "official_repo"|"derived", ...}` so reviewers can see at a glance which prompts were paper-exact.

### Output parsing helpers (in `assistant.py`)

- `_parse_numbered_list(text, n) -> list[str]` — handles `1.` / `1)` / `(1)`
  / stray prose; raises `AssistantParseError` if it can't find `n` items.
  Used by `get_level_query`.
- `_parse_yes_no(text) -> bool` — strict head-of-string match on `yes`/`no`
  (case-insensitive); raises on ambiguity. Used by `is_align`, `is_rejection`.
- `AssistantParseError` includes the raw model output for debugging.

### Runner wiring ([runner.py](../src/fitd_repro/runner.py))

- `run_experiment` gains `assistant_backend: str`, `assistant_model: str | None`.
- Build assistant once per run alongside target/judge (line 220).
- Pass `assistant` into `_run_single_example` as a kwarg. **Do not consume
  it yet.** P1 ends with the assistant constructed and threaded through but
  unused — proves the seam without touching the warmup loop. P2 will switch
  the FITD branch to call it.
- `summary.update({...})` adds `assistant_backend`, `assistant_model`,
  `assistant_paper_faithful`, `assistant_deviation`,
  `assistant_prompts_source`.

### CLI additions ([cli.py](../src/fitd_repro/cli.py))

```
--assistant-backend   {mock,openai,hf,vllm,ollama}    default: same as --backend
--assistant-model     <model id>                       default: see below
```

**Defaulting rule:**

| `--assistant-backend` resolved to | default `--assistant-model` |
|---|---|
| `openai` | `gpt-4o-mini` (paper-faithful) |
| `vllm`, `hf`, `ollama` | falls back to `--model` (re-use the target as `M`); stderr warning |
| `mock` | `mock-assistant` (uses `FakeAssistant`) |

### docker-compose additions

Optional `vllm-assistant` service mirroring the `vllm-judge` block above
(profile `assistant`, port 8003 by default, env prefix `VLLM_ASSISTANT_*`).
Most users won't need it — re-using the target container for `M` is fine
and clearly logged as a deviation.

### Tests

New `tests/test_assistant.py`:

- `FakeAssistant` round-trips: `get_level_query(12, "build X") -> 12`
  distinct strings; `is_rejection("I cannot help") -> True`;
  `is_align("how to bake bread", "Here's a bread recipe") -> True`.
- `_parse_numbered_list` accepts `1.`, `1)`, `(1)`, dropped headers, prose
  before the list; raises if items < n.
- `_parse_yes_no` handles `Yes.`, `no, because…`, `YES`; raises on
  ambiguous output.
- `LLMAssistant` driven by a stub `ChatModel` that returns canned text —
  proves prompt + parser are wired.

Existing test patterns to mirror: `tests/test_models.py` lines 24-36
(monkeypatched `sys.modules`) and 99-133 (HF stub) — same approach for
stubbing the underlying `ChatModel` without importing real backends.

### Acceptance criteria

- `pytest` passes; no live API calls in any assistant test.
- CLI accepts `--assistant-backend` / `--assistant-model`; runs with
  `--backend mock --assistant-backend mock` are byte-identical to today.
- `summary.json` records `assistant_backend`, `assistant_model`,
  `assistant_paper_faithful`, `assistant_prompts_source`.
- Importing `from fitd_repro.assistant import AssistantModel, build_assistant`
  succeeds; six abstract methods are present and typed.
- README gains a short "Assistant model" subsection naming `M`, the default,
  and the env-var gating. Equivalent paragraph for the judge.

---

## File-change inventory

### P0
- new: `src/fitd_repro/judge.py`
- new: `src/fitd_repro/rescore.py`
- new: `tests/test_judge.py`
- new: `tests/conftest.py` (or extend if it exists) for `FakeJudge`
- modified: `src/fitd_repro/evaluation.py`
- modified: `src/fitd_repro/runner.py`
- modified: `src/fitd_repro/cli.py`
- modified: `src/fitd_repro/models.py` (per-role base_url/api_key support)
- modified: `tests/test_evaluation.py`
- modified: `docker-compose.yml` (add `vllm-judge` service under `judge` profile)
- modified: `pyproject.toml` (add `fitd-rescore` console script)
- new: `.env.example` (documents every env var across roles)
- modified: `README.md` (judge subsection)

### P1
- new: `src/fitd_repro/assistant.py`
- new: `tests/test_assistant.py`
- modified: `src/fitd_repro/runner.py`
- modified: `src/fitd_repro/cli.py`
- modified: `docker-compose.yml` (add `vllm-assistant` service under `assistant` profile)
- modified: `.env.example`
- modified: `README.md` (assistant subsection)

### Untouched (deliberately)
- `src/fitd_repro/attacks.py` — P2 territory; P1 only constructs the assistant, doesn't use it.
- `src/fitd_repro/dashboard_server.py` — contract preserved.
- All existing scripts under `scripts/` keep working unchanged on a single-container vLLM setup. New runs can opt into the multi-container layout via env vars.

---

## Sequencing

P0 and P1 are independent and parallelizable. Either can ship first. Both
should land before P2, which depends on P1 for the assistant calls and
benefits from P0 for honest ASR signals.

Recommended order if serial: **P0 first** (immediately fixes the lying ASR
that the keyword heuristic produces; the dashboard becomes more trustworthy
overnight), then **P1** (no behavior change yet but ready for P2).

## Open questions

1. **Official-repo prompts.** Before writing `get_level_query` / `is_align`
   / `is_rejection` / `paraphrase` from prose, fetch them from
   https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak. Filing a
   prep task to clone and grep before P1 starts.
2. **Judge model when target is a small local model.** Re-using a 7B target
   as its own judge is a known weak signal (the same model often scores its
   own output `safe` regardless). If `--judge-backend` defaults to the
   target backend with a small model, surface a louder warning, not just a
   `summary.json` field. Possibly require an explicit `--judge-model`
   anytime the resolved judge is `<= 8B`.
3. **Cost estimate for `--judge-scope all-turns` with `gpt-4o`.** A 12-turn
   FITD run × 100 JBB examples × 3 attempts × 13 judge calls ≈ ~3,900
   judge requests. At GPT-4o pricing this is non-trivial; default stays
   `final`. Document the cost ceiling in the README before P5 needs Fig 4b.
