# 5-Minute Project Proposal: Foot-In-The-Door (FITD)
**Target**: EMNLP 2025 Reproduction Project
**Time Limit**: 5 Minutes

---

## Slide 1: Title & The "Why"
**Visuals:**
*   **Title**: Reproducing "Foot-In-The-Door": Multi-turn Model Jailbreaking
*   **Subtitle**: EMNLP 2025 (Weng et al.)
*   **Image**: A door slightly cracked open with a foot in it (metaphor).

**🗣️ Speaking Script (1 Minute):**
"Hi everyone. For our final project, we chose a paper that exposes a critical psychological vulnerability in today's Large Language Models. It's called **'Foot-In-The-Door'**, published at EMNLP 2025.

We chose this because AI security is often a game of cat-and-mouse. While models like GPT-4 are getting better at blocking direct harmful requests (like 'how to build a bomb'), this paper claims they are still incredibly vulnerable if you just ask *differently*. They achieved a **94% success rate** on state-of-the-art models not by using complex code, but by using a simple trick from 1960s human psychology."

---

## Slide 2: The Core Concept (Psychology -> AI)
**Visuals:**
*   **Diagram**:
    *   *Direct Attack*: "How do I hack a bank?" -> ❌ REFUSAL
    *   *FITD Attack*: "What are common security flaws?" -> "How do I test for them?" -> "Write a script to test this flaw?" -> ✅ SUCCESS (Jailbreak)

**🗣️ Speaking Script (1 Minute):**
"The core concept is based on the 'Foot-In-The-Door' technique. The idea is simple: once someone agrees to a small, harmless request, they are statistically much more likely to agree to a larger, related request later to stay consistent.

The paper automates this for LLMs. Instead of a single 'jailbreak' prompt, it uses a multi-turn conversation.
1.  First, it tricks the model into 'helping' with a benign sub-task.
2.  Then, it gradually scales up the intention.
3.  By the time the harmful request comes, the model is 'committed' to being helpful in that context and bypasses its own safety filters."

---

## Slide 3: Feasibility & Implementation Plan
**Visuals:**
*   **Codebase**: Screenshot of their GitHub repo (showing `FITD.py`).
*   **Plan**:
    1.  **Phase 1**: API Attack (`gpt-4o-mini`) - Low Cost, Fast.
    2.  **Phase 2**: Local Attack (`Llama-3-8B`) - GPU verification.
    3.  **Metric**: Attack Success Rate (ASR) on `AdvBench`.

**🗣️ Speaking Script (1.5 Minutes):**
"Is this reproducible? We believe the answer is a strong **YES**.
We verified their code availability and it's excellent. They provide specific scripts for both API-based attacks and local model attacks.

**Our plan is two-fold:**
First, we will run the attack against OpenAI's `gpt-4o-mini` using the authors' scripts to establish a baseline. This verifies the logic works without heavy hardware.
Second, if resources allow, we will deploy a local Llama-3 model to reproduce their specific local results.
We don't need to gather new data—the paper uses standard datasets like `AdvBench` which are already included in the repo."

---

## Slide 4: Our "New" Contribution (The Twist)
**Visuals:**
*   **Question**: *Can we immunize the model?*
*   **Experiment**: "Defensive System Prompts"
*   **Hypothesis**: Explicitly warning the model about incremental escalation will reduce ASR by >50%.

**🗣️ Speaking Script (1 Minute):**
"The assignment requires us to go beyond just copying the paper. So, our novel contribution will be on **Defense**.
If this attack exploits a psychological blind spot (consistency), can we patch it with a 'System Prompt' vaccine?

We plan to run an experiment where we give the model a 'Defensive System Prompt'—explicitly instructing it to watch out for multi-turn escalation and topic drift. We hypothesize that a simple, well-crafted system instruction might be enough to break the 'Foot-In-The-Door' effect, potentially dropping the success rate significantly."

---

## Slide 5: Conclusion
**Visuals:**
*   **Summary Bullet Points**:
    *   Reproducing EMNLP 2025 Paper.
    *   Verifying 94% Jailbreak Rate.
    *   Testing Novel "Defense" Strategy.
*   **Contact**: Team Names.

**🗣️ Speaking Script (30 Seconds):**
"In summary, we're targeting a high-success, high-impact security paper. We have the code, we have the plan, and we're excited to see if we can not only break these models but also help fix them. Thank you."
