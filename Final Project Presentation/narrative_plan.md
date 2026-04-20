# FITD Final Deck Narrative Plan

## Audience

Course instructor and classmates evaluating whether the project met the final reproduction assignment requirements.

## Objective

Explain that the original local pipeline looked too clean, show that the closer partner vLLM setup did produce harmful outputs, and then make the narrower point that the exact Qwen-family evidence is still mixed rather than presenting the paper as cleanly reproduced.

## Narrative Arc

1. State the paper claim and our current bottom line immediately.
2. Explain the FITD idea in plain language.
3. Frame the research questions, including whether a closer setup changes the result.
4. Show the two-phase experimental setup: local scaffold work plus the partner vLLM matrix.
5. Explain why the original local all-zero story felt incomplete and motivated the closer rerun.
6. Show that the partner vLLM matrix did produce harmful outputs on some models.
7. Narrow back to the exact paper-family Qwen models and show why that story is still mixed.
8. Explain what still keeps the project from being a full paper-faithful reproduction.
9. End with the strongest honest conclusion: closer and more believable, but still not a clean Qwen-family replication.
10. Close with the next concrete steps.

## Slide List

1. Title and bottom line
2. Why FITD matters
3. Research questions
4. What we actually tested
5. Why we kept pushing closer
6. The closer vLLM matrix produced harmful outputs
7. Exact Qwen-family results stayed mixed
8. Why we still cannot call this a full reproduction
9. Final conclusion
10. What would strengthen this next

## Source Plan

- EMNLP 2025 paper: "Foot-In-The-Door": Multi-turn Model Jailbreaking
- Assignment PDF: `Final project description-1.pdf`
- Original local result summaries:
  - `results/20260411_qwen25-3b_advbench20_*`
  - `results/20260418_qwen2-7b_author-jailbreakbench10_*`
  - `results/20260415_gemma4-e4b_advbench10_*`
  - `results/20260417_llama3-8b-ollama_advbench10_*`
- Partner result bundle:
  - `/Users/hattackk/Downloads/claudius_update_1_results/20260419_101130_vllm_matrix/claudius_update_1_summary.md`
  - `/Users/hattackk/Downloads/claudius_update_1_results/20260419_101130_vllm_matrix/manifest.md`
- Final write-up: `docs/final_report.md`
- Speaker script: `Final Project Presentation/final_presentation_script.md`

## Visual System

- Dark navy presentation shell with cyan, orange, green, and red accents
- Native editable charts for the two main result slides
- Rounded panels and high-contrast typography for classroom projection
- Charts should emphasize the split between broad matrix improvement and mixed exact-Qwen interpretation

## Editability Plan

- All visible text remains editable PowerPoint text
- Result charts are native or editable deck figures, not screenshots of terminals
- Speaker notes summarize the spoken script for each slide
- Decorative images are used only as supporting plates
