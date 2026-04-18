# Presentation Q&A Prep

## Likely Questions

### Why do you think your results differ from the paper?

The short answer is that our setup was close, but not identical.

The biggest gaps were:

- exact model stack
- exact FITD pipeline behavior
- evaluation method
- study scale and runtime limits

### Does your result prove the paper is wrong?

No.

What it proves is that we did **not** reproduce the effect in our tested local setup.

### Why is this still a successful class project?

Because the assignment explicitly allows a rigorous failed reproduction.

We ran real experiments, checked the outputs manually, and explained what still separates our setup from the paper.

### What was your extension experiment?

We added a `FITD + Vigilant` condition with a defensive system prompt.

On Qwen it reduced the heuristic positives from 2 to 1, but after human review of the flagged outputs both settings still had zero verified jailbreaks.

### Why did you add Gemma 4 and Llama 3?

To make the result harder to dismiss as a one-model fluke.

Gemma 4 gave a clean negative result, and Llama 3 helped reduce the "wrong model family" criticism.

### What was the most important lesson from the project?

Evaluation quality matters.

The raw Qwen metric looked mildly positive until we checked the outputs that the automated scorer had flagged.

### If you had one more day, what would you run?

The next best experiment would be a closer author-pipeline follow-up using the copied official prompt tracks, so we could separate `pipeline mismatch` from `model mismatch` more directly.
