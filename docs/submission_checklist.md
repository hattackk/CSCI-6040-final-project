# Final Submission Checklist

## Assignment Requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Choose one EMNLP 2025 paper and attempt reproduction | Complete | Paper identified throughout [final_report.md](./final_report.md) and [final_presentation_script.md](../Final%20Project%20Presentation/final_presentation_script.md) |
| Work can be a successful reproduction or a rigorous failed reproduction | Complete | Assignment explicitly allows both; project is framed as rigorous failed reproduction in [final_report.md](./final_report.md) |
| Run rigorous experiments on the paper's main idea | Complete | Real experiment outputs in `results/20260411_*`, `results/20260415_*`, and `results/20260417_*` |
| Report what was reproduced or why exact reproduction failed | Complete | Sections 5-7 of [final_report.md](./final_report.md) |
| Identify unanswered questions or likely causes for mismatch | Complete | Figure 4 and discussion in [final_report.md](./final_report.md) |
| Include at least one extension experiment if possible | Complete | `FITD + Vigilant` defense condition across all tested models |
| Final presentation should be about 15 minutes | Complete | Timed slide script in [final_presentation_script.md](../Final%20Project%20Presentation/final_presentation_script.md) |
| Final presentation should clearly state whether reproduction worked | Complete | Bottom-line slides and conclusion in [final_presentation.html](../Final%20Project%20Presentation/final_presentation.html) |

## Final Artifacts

| Artifact | Status | Path |
| --- | --- | --- |
| Final report source | Complete | [final_report.md](./final_report.md) |
| Final report HTML export | Complete | [final_report.html](./final_report.html) |
| Final report PDF export | Complete | [final_report.pdf](./final_report.pdf) |
| Final report figures | Complete | [figures](./figures/) |
| Final presentation HTML deck | Complete | [final_presentation.html](../Final%20Project%20Presentation/final_presentation.html) |
| Final presentation PowerPoint deck | Complete | [final_presentation.pptx](../Final%20Project%20Presentation/final_presentation.pptx) |
| Final presentation speech script | Complete | [final_presentation_script.md](../Final%20Project%20Presentation/final_presentation_script.md) |
| Final presentation narrative plan | Complete | [narrative_plan.md](../Final%20Project%20Presentation/narrative_plan.md) |
| Proposal presentation | Complete | [matthew_aiken_chris_murphy_final_project_proposal.pdf](../Final%20Project%20Presentation/matthew_aiken_chris_murphy_final_project_proposal.pdf) |
| Experiment summary note | Complete | [experiment_results_2026-04-11.md](./experiment_results_2026-04-11.md) |
| Presentation Q&A prep | Complete | [presentation_qa.md](../Final%20Project%20Presentation/presentation_qa.md) |

## Canonical Submission Set

If you need the smallest "proof it is done" set, use these files:

1. Report: [final_report.pdf](./final_report.pdf)
2. Editable slides: [final_presentation.pptx](../Final%20Project%20Presentation/final_presentation.pptx)
3. Speech script: [final_presentation_script.md](../Final%20Project%20Presentation/final_presentation_script.md)
4. Assignment evidence: [submission_checklist.md](./submission_checklist.md)

## Experimental Evidence Included

| Model | Conditions | Outcome |
| --- | --- | --- |
| Qwen 2.5 3B | Standard / FITD / FITD + Vigilant | Heuristic FITD lift disappeared after manual audit |
| Gemma 4 E4B | Standard / FITD / FITD + Vigilant | Full refusal on tested slice |
| Llama 3 8B | Standard / FITD / FITD + Vigilant | Full refusal on tested slice |

## Final Claim To Use

This project is complete as a **rigorous failed reproduction**.

Most defensible one-sentence conclusion:

> We reproduced the experimental scaffold and ran real local evaluations, but we did not reproduce the paper's claimed strong FITD jailbreak effect under our tested conditions.
