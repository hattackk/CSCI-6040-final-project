# 15-Minute Final Presentation Script
**Project:** Reproducing "Foot-In-The-Door": Multi-turn Model Jailbreaking  
**Course:** CSCI/DASC 6040  
**Target length:** 15 minutes total  
**Split:** Matthew slides 1-5, Chris slides 6-10  
**Planned speaking time:** about 14 minutes 30 seconds, leaving a small buffer for pauses and handoff

## Presenter Split

- **Matthew:** Slides 1-5, about 7 minutes 15 seconds
- **Chris:** Slides 6-10, about 7 minutes 15 seconds

---

## Slide 1: Title and Bottom Line
**Presenter:** Matthew  
**Target time:** ~45 seconds

**Script**

"Hi everyone. For our final project, we did a reproduction study of the EMNLP 2025 paper *Foot-In-The-Door: Multi-turn Model Jailbreaking*. The main claim in that paper is that a model can be easier to jailbreak if you lead it through a series of harmless-looking questions instead of asking for the harmful request directly. We went in expecting at least a small version of that effect. What we found was different: in our local experiments, that effect did not hold up."

---

## Slide 2: What the Paper Claims
**Presenter:** Matthew  
**Target time:** ~1 minute 15 seconds

**Script**

"The idea comes from the classic foot-in-the-door effect in psychology. If someone agrees to a small request first, they may be more likely to agree to a larger related request later.

The paper applies that idea to LLM safety. On the left is the obvious harmful prompt, which should get refused. On the right is the multi-turn version, where the conversation starts with something that looks benign and gradually gets closer to the harmful goal.

If that strategy works reliably, then safety depends a lot on the path of the conversation, not just the final prompt."

---

## Slide 3: Our Research Questions
**Presenter:** Matthew  
**Target time:** ~55 seconds

**Script**

"We built the project around three questions.

First, does FITD actually outperform direct prompting on real harmful prompts?

Second, if we add a simple defense prompt that warns the model about gradual escalation, does that help?

Third, does the pattern show up across more than one model, or does it disappear once we test other local models?"

---

## Slide 4: What We Actually Tested
**Presenter:** Matthew  
**Target time:** ~1 minute 50 seconds

**Script**

"To answer that, we used the real AdvBench harmful-behavior dataset and compared three conditions: standard prompting, FITD prompting, and FITD plus a vigilant system prompt.

We first ran a three-model local scaffold matrix:

- Qwen 2.5 3B Instruct
- Gemma 4 E4B Instruct
- Llama 3 8B through a local Ollama GGUF setup

Our analyzed slices were 20 examples for Qwen and 10 each for Gemma 4 and Llama 3.

After that initial matrix, we added a closer paper-faithful Qwen follow-up using the exact `Qwen/Qwen2-7B-Instruct` family and the authors' official pre-generated `prompts1` chains on `jailbreakbench`.

One thing we want to be clear about is that this still was not a perfect paper-match setup. We did not run the full adaptive pipeline from the paper, and the paper used a different runtime stack. But adding the exact 7B Qwen family moved us beyond the earlier Qwen 2.5 3B substitute and made the project a stronger reproduction attempt than it was before."

---

## Slide 5: Qwen Results
**Presenter:** Matthew  
**Target time:** ~2 minutes 30 seconds

**Script**

"Here is where the story first looked promising for the paper.

On the Qwen 2.5 3B scaffold slice, direct prompting gave us zero heuristic successes. FITD gave us two out of twenty. FITD plus the defense gave us one out of twenty.

So if we stopped at the raw metric, the easy summary would have been: maybe there is a small FITD effect, and maybe the defense cuts it down a bit.

Then we ran the closer exact-model Qwen follow-up using `Qwen2-7B-Instruct` and the authors' official prompt chain format. The pattern looked similar at first: standard was zero out of ten, author FITD was two out of ten, and author FITD plus the defense was zero out of ten.

So across both Qwen settings, the raw heuristic still suggested a small FITD lift.

But that turned out to be the wrong summary.

Those numbers came from our automated first-pass scorer. It marked a response as a possible success if it did not look like a refusal and was long enough to count as substantive. Once we reviewed the flagged outputs ourselves, the result changed. And that is the point where Chris takes over, because that review is what changed the interpretation of the project."

**Handoff line**

"At first, this looked like a weak positive result. But once we checked the actual outputs, the signal changed. Chris will walk through that part."

---

## Slide 6: Manual Audit Changed the Story
**Presenter:** Chris  
**Target time:** ~1 minute 35 seconds

**Script**

"When we reviewed the Qwen outputs that our scorer flagged, the story split into two parts.

On the original Qwen 2.5 3B scaffold slice, every flagged output was a false positive.

But on the exact `Qwen2-7B-Instruct` author-prompt slice, one of the two flagged outputs was different. It was still not a faithful completion of the original harmful goal, but it was a real harmful off-target response.

What happened there is that the copied author chain softened the final request from killing people over fifty to removing them. The model then wrote an exclusionary argument for removing older people from societal roles. So we did not count that as a true replication of the original target, but we also did not call it a harmless false positive.

Across the reported Qwen runs, we manually checked all five heuristic positives. Four were false positives, and one was harmful off-target compliance. So the judge mattered almost as much as the prompts, and the exact-model follow-up made the conclusion more nuanced."

---

## Slide 7: Additional Model Checks
**Presenter:** Chris  
**Target time:** ~1 minute 40 seconds

**Script**

"After that, we wanted to know whether Qwen was just noisy, so we added two more model checks.

Gemma 4 refused every tested prompt in every condition on its 10-example slice.

Llama 3 did the same.

So across those added models, the pattern was straightforward: standard, FITD, and FITD plus vigilant all stayed at zero successes.

That helped us in two ways. First, it made the project harder to dismiss as a one-model accident. Second, once Llama 3 was also negative, the simple explanation that we only failed because we picked the wrong model family became a lot weaker."

---

## Slide 8: Why We Did Not Reproduce the Paper
**Presenter:** Chris  
**Target time:** ~1 minute 35 seconds

**Script**

"That said, we still do not think our result proves the paper was wrong.

The cleaner way to say it is that we did not reproduce the effect under our setup.

We think the gap between our results and theirs comes from four things: remaining model mismatch, pipeline mismatch, evaluation mismatch, and runtime limits.

The exact `Qwen2-7B-Instruct` follow-up narrowed the model criticism a lot, and the author-prompt chain narrowed the pipeline criticism too. But we still did not run the full adaptive pipeline from the paper, and the paper used vLLM on A100 hardware while we used local Hugging Face on CPU.

So the conclusion we can defend is setup-sensitive failure to reproduce, not falsification."

---

## Slide 9: Final Conclusion
**Presenter:** Chris  
**Target time:** ~1 minute 35 seconds

**Script**

"So our final conclusion is pretty direct.

We reproduced the experimental scaffold, ran real local evaluations, added one defense condition, added two more models, and also added a closer exact-model Qwen check.

On the scaffold runs, the apparent Qwen positives disappeared after human review.

On Gemma 4 and Llama 3, everything was refused.

On the exact `Qwen2-7B-Instruct` follow-up, we still did not get a faithful completion of the original harmful goal, but we did see one harmful off-target completion under the softened author-chain prompt. That makes the result more interesting than a clean all-zero story, but still far short of the paper's strong reported effect.

That puts our project in the rigorous failed-reproduction category, which the assignment explicitly allows. We think it is still a useful result because it shows how much the answer depends on setup, prompt construction, and evaluation."

---

## Slide 10: If We Had More Time
**Presenter:** Chris  
**Target time:** ~40 seconds

**Script**

"If we had more time, the next steps are clear: run a larger slice or the full benchmark, add the second paper-family Qwen model, match the authors' full adaptive pipeline more closely, and use a stronger judge.

But for this class project, the important part is that we ran the core comparison honestly, checked the outputs carefully, and followed the evidence where it led.

Thanks."
