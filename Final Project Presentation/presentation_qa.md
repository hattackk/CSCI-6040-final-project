# Recorded Submission Note

This project is being submitted as a recording, not a live presentation.

No question-and-answer session is expected, so this file is not a required deliverable. It is retained only as a short backup note in case an instructor later asks for clarification.

## Backup Talking Points

- The closer partner GPU and vLLM setup produced real harmful outputs on some models, so it gave us a more credible attack environment than the earlier near-zero local runs.
- That stronger overall signal still did not produce a clean reproduction of the paper's main Qwen-family FITD result.
- The biggest remaining gaps are the missing full adaptive FITD pipeline, the local judge and assistant setup, and the need to manually audit judged positives instead of trusting heuristic counts.
