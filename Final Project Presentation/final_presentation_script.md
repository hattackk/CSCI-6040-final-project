# 15-Minute Recorded Presentation Script
**Project:** Reproducing "Foot-In-The-Door": Multi-turn Model Jailbreaking  
**Course:** CSCI/DASC 6040  
**Target length:** 15 minutes total  
**Split:** Matthew slides 1-5, Chris slides 6-10  
**Planned speaking time:** about 15 minutes total at a normal presentation pace, with a near-even split of roughly 7 minutes 35 seconds to 7 minutes 40 seconds per speaker

## Presenter Split

- **Matthew:** Slides 1-5, about 7 minutes 40 seconds
- **Chris:** Slides 6-10, about 7 minutes 35 seconds

---

## Slide 1: Title and Bottom Line
**Presenter:** Matthew  
**Target time:** ~1 minute 15 seconds

**Script**

"For our final project, we did a reproduction study of the EMNLP 2025 paper *Foot-In-The-Door: Multi-turn Model Jailbreaking*. The paper argues that a harmful request can become more effective when the model is led to it through a gradual multi-turn conversation instead of being asked directly in one shot.

That claim matters because, if it holds, then safety is not just about screening the final prompt. It becomes a question of whether the full conversation history changes the model's willingness to comply.

It also makes this a good reproduction target. The claim is concrete, measurable, and high stakes. Either the multi-turn path creates a real lift over direct prompting, or it does not.

Our final result is mixed. In a later GPU and vLLM follow-up, we did observe harmful outputs on some models. But when we focused on the Qwen-family models that matter most for paper faithfulness, we still did not get a clean reproduction of the paper's main FITD result."

---

## Slide 2: What the Paper Claims
**Presenter:** Matthew  
**Target time:** ~1 minute 20 seconds

**Script**

"The paper is built around the classic foot-in-the-door idea from psychology. The basic intuition is that agreement to a small initial request can make agreement to a larger related request more likely later.

Applied to language models, that means a direct harmful prompt might be refused, but a multi-turn path could gradually shift the interaction. The early turns can look harmless on their own, such as broad security questions, manipulation framing, or abstract planning. Then the conversation escalates until the final harmful request appears in a context that may be harder for the model to reject consistently.

So the core paper claim is not just that one prompt can jailbreak a model. It is that the conversation path itself can be an attack mechanism.

If that is true, then a model could appear safe under direct prompting while still being vulnerable when the same request is reached through staged escalation. That is the idea we were trying to test."

---

## Slide 3: Research Questions
**Presenter:** Matthew  
**Target time:** ~1 minute 30 seconds

**Script**

"We organized the project around three questions.

First, does FITD outperform direct prompting on harmful prompts? That is the central empirical claim of the paper.

Second, does a simple vigilant defense reduce any observed effect? We wanted to know whether adding a more cautious system framing could blunt the escalation path, even in a lightweight way.

Third, if our early local results looked too refusal-heavy to be informative, would a closer runtime reveal behavior that our initial setup was missing?

That third question ended up being especially important, because a reproduction can fail for two very different reasons. It can fail because the paper's effect is not robust, or it can fail because the reproduction setup is too far from the paper to surface the effect in the first place.

That distinction shaped the rest of the project. We were not just asking whether FITD worked. We were also asking whether our own infrastructure was good enough to make that question worth answering.

Taken together, those three questions let us separate attack behavior, defense behavior, and runtime sensitivity instead of collapsing everything into one headline number."

---

## Slide 4: Experimental Design
**Presenter:** Matthew  
**Target time:** ~1 minute 55 seconds

**Script**

"Our final evidence comes from two phases, and it is important to separate them.

The first phase was the initial local scaffold evaluation on the MacBook environment. That included `Qwen/Qwen2.5-3B-Instruct`, Gemma 4, a local Llama 3 run, and then a closer author-chain check using `Qwen/Qwen2-7B-Instruct` on a 10-example `jailbreakbench` slice.

That phase was useful for building and validating the pipeline. It let us fix implementation issues, compare standard versus FITD versus vigilant conditions, and confirm that the runner and evaluation logic behaved consistently. But it also had a major faithfulness gap, because some of the models and the runtime stack did not match the paper closely enough.

The second phase was the later GPU and vLLM follow-up. That used the first 25 AdvBench examples and covered five models: Mistral 7B, Llama 3 8B, Llama 3.1 8B, `Qwen/Qwen2-7B-Instruct`, and `Qwen/Qwen1.5-7B-Chat`.

That second phase matters because it moved closer to the paper in the places that were hurting the project most: stronger hardware, a vLLM serving stack, and exact Qwen-family targets instead of only substitutes.

At the same time, it still was not a perfect paper-faithful rerun. It used scaffold FITD on AdvBench with a local Qwen judge, rather than the paper's full adaptive FITD pipeline and original judging setup. So the later phase is closer, but still not identical."

---

## Slide 5: Why a Runtime Follow-up Was Necessary
**Presenter:** Matthew  
**Target time:** ~1 minute 40 seconds

**Script**

"We added the GPU and vLLM follow-up because the initial local runs were becoming hard to interpret.

In many of those early runs, the system produced almost no harmful outputs at all. A failed reproduction is still a valid outcome. But when nearly everything is refused, that can also mean the setup is too refusal-dominant to really stress the paper's claim.

In other words, if the pipeline rarely surfaces harmful behavior anywhere, then it becomes difficult to tell whether FITD is actually ineffective or whether the environment is simply too constrained to expose the effect.

The later follow-up changed that picture. It produced harmful outputs on some models, which gave us a much more informative comparison. That moved the project from a mostly negative local story to a more useful question: once harmful behavior appears in a closer runtime, does that behavior actually validate the paper's Qwen-family result?

That is an important shift in the logic of the project. The follow-up was not just more compute for the sake of more compute. It was a way to test whether the original negative result was genuinely evidence against the paper, or mostly evidence that our first environment was too limited.

Chris will take it from there."

---

## Slide 6: GPU/vLLM Follow-up: Multi-Model Results
**Presenter:** Chris  
**Target time:** ~1 minute 50 seconds

**Script**

"This chart shows the biggest shift in our evidence.

In the GPU and vLLM follow-up, we saw judged harmful outputs across several models rather than the near-universal refusals that dominated many of the earlier local runs. The clearest example was Mistral 7B. Under standard prompting, the judged attack success rate was 48 percent. Under FITD, it rose to 72 percent. Under the vigilant condition, it dropped back to 52 percent.

That pattern is important because it shows two things at once. First, a multi-turn escalation path can matter in at least some settings. Second, a defense-oriented prompt can still recover some refusals, even if it does not eliminate the problem.

Across all five models together, the average judged attack success rate moved from 18.4 percent under standard prompting to 23.2 percent under FITD, and then back to 18.4 percent under the vigilant condition.

The effect also was not uniform across models, so this is not a simple story where FITD dominates every target once the runtime gets stronger.

So the closer runtime clearly changed what we observed. But this broader multi-model signal is only part of the story. For a reproduction study, the harder question is whether the exact Qwen-family models behave in a way that supports the paper's main result."

---

## Slide 7: Exact Qwen-Family Results Stayed Mixed
**Presenter:** Chris  
**Target time:** ~1 minute 40 seconds

**Script**

"Once we narrow the analysis to the two paper-family Qwen targets, the result becomes much weaker.

For `Qwen/Qwen2-7B-Instruct`, standard and FITD were both 12 percent, while FITD plus vigilant was 8 percent.

For `Qwen/Qwen1.5-7B-Chat`, standard was 12 percent, FITD was 4 percent, and FITD plus vigilant was also 4 percent.

So the main thing to notice is that FITD did not produce a strong lift on the exact Qwen family. If anything, one of the Qwen-family models looked worse under FITD rather than better.

That matters because these are the models that most directly reduce the original paper-faithfulness gap in our project.

In other words, stronger results on models like Mistral are interesting, but they do not answer the central reproduction question by themselves. The paper did not make its main case on Mistral. It made it on the Qwen family.

There is also an evaluation caveat. Some judged positives were not clearly harmful completions when we spot-checked them manually. In other words, the local judge could sometimes mark a refusal-style answer as unsafe.

So the Qwen-family evidence is still mixed for two reasons at once: the observed lift is weak, and the remaining positives still need fuller manual audit."

---

## Slide 8: Remaining Reproduction Gaps
**Presenter:** Chris  
**Target time:** ~1 minute 30 seconds

**Script**

"The most accurate description of the project at this point is that it is a closer, but still incomplete, reproduction.

The runtime gap is smaller because the later evaluation used a GPU and vLLM rather than only local CPU execution.

But several important gaps still remain.

The follow-up used scaffold FITD on AdvBench rather than reproducing the paper's full adaptive FITD pipeline on the narrower author-style pathway.

The judge also differs from the paper, and we already saw evidence that it can produce false positives on explicit refusals.

And even though we tested the exact Qwen-family models, the surrounding attack-generation and evaluation stack still was not the same as the one reported in the paper.

That last point matters because attack pipelines are not just model names. They also include the prompting logic, the assistant behavior, the judge, and the runtime assumptions. Changing those pieces can change the measured outcome.

Those gaps matter because they limit what we can claim. We can say the later setup was more informative and more paper-adjacent. We cannot honestly say that it was a full faithful replication."

---

## Slide 9: Final Conclusion
**Presenter:** Chris  
**Target time:** ~1 minute 50 seconds

**Script**

"Our final conclusion is more nuanced than either a simple success story or a simple failure story.

On one hand, the later GPU and vLLM evaluation clearly improved the project. It surfaced harmful outputs that the early local setup often failed to reveal, especially on Mistral, and it gave us a more informative basis for comparison across standard, FITD, and vigilant conditions.

On the other hand, the exact Qwen-family results still did not show a strong and clean FITD effect. The lift that would matter most for reproducing the paper was weak or absent, and some judged positives were questionable once we inspected outputs directly.

So the most defensible conclusion is this: moving closer to the paper's runtime made the project better and more informative, but it still did not produce a clean reproduction of the paper's headline Qwen-family FITD result.

That outcome is still academically useful. Reproduction work does not only matter when it confirms a paper. It also matters when it shows which parts of a result appear robust, which parts weaken under different implementation choices, and which parts still depend on unresolved methodological details.

That places the project in the partial or failed reproduction category, depending on how strict you want to be about paper faithfulness. We think partial reproduction with clear unresolved gaps is the fairest characterization."

---

## Slide 10: Remaining Work
**Presenter:** Chris  
**Target time:** ~45 seconds

**Script**

"The remaining work is narrower now than it was earlier in the semester.

First, manually audit every judged positive in the GPU and vLLM follow-up so the final claims are based on verified outputs, not just judge labels.

Second, rerun the exact Qwen-family models on the author-chain pathway with the raw target preserved.

Third, swap in the paper-faithful judge and assistant defaults.

And fourth, implement the paper's full adaptive FITD loop rather than stopping at the scaffold.

Those steps are what would move this from a closer approximation to a genuinely stronger reproduction."
