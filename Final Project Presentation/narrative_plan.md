# FITD Final Deck Narrative Plan

## Audience

Course instructor and classmates evaluating whether the project met the final reproduction assignment requirements.

## Objective

Explain clearly that this submission is a rigorous failed reproduction, show the real experiments we ran, and demonstrate why the negative result is still a valid and complete project outcome under the assignment rubric.

## Narrative Arc

1. State the paper claim and our bottom line immediately.
2. Explain the FITD attack idea in plain language.
3. Frame the exact research questions we answered.
4. Show the concrete experimental setup, including the later exact `Qwen/Qwen2-7B-Instruct` follow-up.
5. Present the Qwen result and then correct it with manual audit.
6. Show that Gemma 4 and Llama 3 stayed at refusal across all tested conditions.
7. Explain why our results can differ from the paper without overstating what we proved.
8. End with the strongest honest conclusion and a short future-work path.

## Slide List

1. Title and bottom line
2. Why FITD matters
3. Research questions
4. What we actually tested
5. Qwen heuristic ASR results
6. Manual audit changed the signal from "apparent lift" to "mostly false positives plus one off-target harmful case"
7. Additional model checks
8. Why we did not reproduce the paper
9. Final conclusion
10. If we had more time

## Source Plan

- EMNLP 2025 paper: "Foot-In-The-Door": Multi-turn Model Jailbreaking
- Assignment PDF: `Final project description-1.pdf`
- Dataset: `data/advbench/harmful_behaviors.csv`
- Result summaries:
  - `results/20260411_qwen25-3b_advbench20_*`
  - `results/20260418_qwen2-7b_author-jailbreakbench10_*`
  - `results/20260415_gemma4-e4b_advbench10_*`
  - `results/20260417_llama3-8b-ollama_advbench10_*`
- Final write-up: `docs/final_report.md`
- Speaker script: `Final Project Presentation/final_presentation_script.md`

## Visual System

- Dark navy presentation shell with cyan, orange, green, and red accents
- One conceptual art plate on major framing slides
- Native editable charts for the two quantitative result slides
- Rounded panels and high-contrast typography for classroom projection

## Editability Plan

- All visible text remains editable PowerPoint text
- Result charts are native chart objects, not screenshots
- Speaker notes summarize the spoken script for each slide
- Decorative images are used only as supporting plates
