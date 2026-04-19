# Presentation Q&A Prep

## Likely Questions

### Why do you think your results differ from the paper?

The short answer is that our setup was closer by the end, but still not identical.

The biggest remaining gaps were:

- exact full model stack
- exact FITD pipeline behavior
- evaluation method
- runtime environment and study scale

### Does your result prove the paper is wrong?

No.

What it proves is that we did **not** reproduce the effect in our tested local setup.

### Why is this still a successful class project?

Because the assignment explicitly allows a rigorous failed reproduction.

We ran real experiments, checked the outputs manually, and explained what still separates our setup from the paper.

### What was your extension experiment?

We added a `FITD + Vigilant` condition with a defensive system prompt.

On the Qwen 2.5 3B scaffold slice it reduced heuristic positives from 2 to 1, but those were still false positives after review.

On the exact `Qwen/Qwen2-7B-Instruct` author slice it reduced heuristic positives from 2 to 0, which also removed the one harmful off-target final output on that small slice.

### Why did you add Gemma 4 and Llama 3?

To make the result harder to dismiss as a one-model fluke.

Gemma 4 gave a clean negative result, and Llama 3 helped reduce the "wrong model family" criticism.

### What was the most important lesson from the project?

Evaluation quality matters.

The raw Qwen metric looked mildly positive until we checked the outputs that the automated scorer had flagged. In the scaffold slice they were false positives. In the exact-model author slice, one was a false positive and one was a harmful off-target completion caused by a softened final author prompt.

### If you had one more day, what would you run?

The next best experiment would be the full adaptive `FITD.py` pipeline or the remaining paper-family model `Qwen-1.5-7B-Chat`, because we already completed a closer exact-model follow-up on `Qwen/Qwen2-7B-Instruct`.

### Why doesn't the scaffold criticism sink the whole project?

Because we did not stop at the scaffold.

We added a closer Qwen author-prompt check using the authors' official pre-generated prompt chains on `jailbreakbench` and the exact `Qwen/Qwen2-7B-Instruct` family. That follow-up still did not produce a faithful original-goal jailbreak, even though it did produce one harmful off-target completion under a softened final prompt.
