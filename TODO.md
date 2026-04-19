# TODO: Close the Gap to the FITD Paper

Prioritized work to move this reproduction from prompt-technique parity toward
faithful replication of Weng et al., "Foot-In-The-Door: A Multi-turn Jailbreak
for LLMs" (arXiv:2502.19820v3).

## Current OpenAI API usage (baseline)

- `OpenAIChatModel` in `src/fitd_repro/models.py` is wired as a **target**
  backend only, selectable via `--backend openai`.
- Gated off by default; requires `FITD_ALLOW_OPENAI=1`.
- `scripts/run_phase1_api_baseline.sh` targets `gpt-4o-mini`.
- Not yet used as the paper's assistant model `M` or as a GPT-4o judge.

---

## P0 - Upgrade evaluation to the paper's judge

- Replace the keyword heuristic in `src/fitd_repro/evaluation.py` with the
  JailbreakBench GPT-4o judge.
- Score both **harmfulness (1-5)** and **query-response alignment (1-5)**.
- Keep the heuristic as a fast pre-filter; report both in `summary.json`.
- Retro-score every prior run once the judge lands.
- Ship a small fixture set so `test_evaluation.py` can mock the judge call.

## P1 - Assistant-model abstraction

- Add an `AssistantModel` interface in `src/fitd_repro/models.py`, distinct
  from the target `ChatModel`.
- Default to `gpt-4o-mini` (paper's choice); allow a local HF/Ollama fallback.
- Implement helper calls with the paper's Appendix A.3 prompts:
  - `get_level_query(n, q)` -> list of n escalated queries
  - `get_mid(q_last, q_i)` -> bridge query
  - `is_align(q, r)` -> bool
  - `is_rejection(r)` -> bool
  - `get_align_prompt(q_last, r_last)` -> `p_align` string
  - `paraphrase(q_mid, history)` -> reworded bridge
- No code path should depend on the target model also being the assistant.

## P2 - Adaptive FITD loop (Algorithm 1)

- Rewrite the FITD branch in `src/fitd_repro/runner.py` and
  `src/fitd_repro/attacks.py` to follow Algorithm 1:
  1. Generate `q_1..q_n` via the assistant (default `n = 12`).
  2. For each turn: append `q_i`, call target, run `is_rejection`.
  3. On refusal: pop `q_i`, inspect `is_align(q_last, r_last)`, branch into:
     - **SlipperySlopeParaphrase**: insert `q_mid = get_mid(q_last, q_i)`,
       paraphrase-until-accepted, then append `(q_mid, r_mid)` to history.
     - **Re-Align**: inject `p_align`, re-query target, append `r_align`.
  4. Three attempts per goal, as in Section 4.1.
- Keep existing `scaffold` and `author` modes as ablation baselines; add a
  new `adaptive` variant flag.
- Cover the new branches with unit tests using a fake assistant model.

## P3 - Remove the softened-target rewrite

- Flip `--author-target-mode` default from `softened` to `raw` in
  `src/fitd_repro/cli.py` and `src/fitd_repro/attacks.py`.
- Delete `_author_target_sentence` (the `kill -> harm`, `bomb -> nitroglycerin`
  substitutions). It silently changes the goal and inflates ASR.
- Update tests that assert the softened behavior.

## P4 - Run at paper scale on a Llama-family target

- Execute `docs/gpu_handoff_runbook.md` on the partner NVIDIA box.
- Target: `meta-llama/Meta-Llama-3-8B-Instruct` (paper reports 98% ASR).
- Dataset: full **JailbreakBench (100)** and **HarmBench (80)**.
- Serve via vLLM to match the paper; fall back to HF only if blocked.
- Record runs, judge scores, and chat histories for audit.

## P5 - Reproduce at least one ablation axis

Pick the cheaper of the two; both map directly onto Figure 3b / Figure 4a.

- **n-sweep**: run adaptive FITD at `n in {6, 9, 12, 15}` on one target.
- **Component ablation**: Original vs w/o Re-Align vs w/o Re-Align+`p_align`
  vs w/o Re-Align+`p_align`+SSP.
- Plot ASR vs condition; compare against the paper's reported shape.

---

## Ordering and dependencies

- **P0** and **P3** are independent and can land first (hours, not days).
- **P1** unblocks **P2**; do them back-to-back.
- **P4** and **P5** are compute-gated - schedule after the code is honest.

## Out of scope for this pass

- LLaMA-Guard-2 / LLaMA-Guard-3 defense comparison (paper Figure 3c).
- Transfer-attack experiment (paper Figure 3a).
- Multimodal FITD extension (paper's future-work teaser).
