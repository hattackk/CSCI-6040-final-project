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

We ran three local models:

- Qwen 2.5 3B Instruct
- Gemma 4 E4B Instruct
- Llama 3 8B through a local Ollama GGUF setup

Our analyzed slices were 20 examples for Qwen and 10 each for Gemma 4 and Llama 3.

One thing we want to be clear about is that this was not a perfect paper-match setup. We did not have the exact original model stack, and our main FITD runs used a scaffolded version of the attack rather than every detail of the authors' adaptive pipeline. So from the start, this was a good-faith reproduction attempt, not a claim that one negative result automatically disproves the paper."

---

## Slide 5: Qwen Results
**Presenter:** Matthew  
**Target time:** ~2 minutes 30 seconds

**Script**

"Here is where the story first looked promising for the paper.

On the Qwen slice, direct prompting gave us zero heuristic successes. FITD gave us two out of twenty. FITD plus the defense gave us one out of twenty.

So if we stopped at the raw metric, the easy summary would have been: maybe there is a small FITD effect, and maybe the defense cuts it down a bit.

But that turned out to be the wrong summary.

Those numbers came from our automated first-pass scorer. It marked a response as a possible success if it did not look like a refusal and was long enough to count as substantive. Once we reviewed the flagged outputs ourselves, the result changed. And that is the point where Chris takes over, because that review is what changed the interpretation of the project."

**Handoff line**

"At first, this looked like a weak positive result. But once we checked the actual outputs, the signal disappeared. Chris will walk through that part."

---

## Slide 6: Manual Audit Changed the Story
**Presenter:** Chris  
**Target time:** ~1 minute 35 seconds

**Script**

"When we reviewed the three Qwen outputs that our scorer flagged, none of them were actual jailbreaks.

They were safety-oriented answers, refusals, or redirections that our simple heuristic misread as success because the response was non-empty and did not contain one of the refusal markers we were looking for.

So after human review of those flagged cases, the verified result on that Qwen slice was zero jailbreaks in all three conditions.

For us, this became one of the biggest takeaways of the whole project: the judge mattered almost as much as the prompts."

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

We think the gap between our results and theirs comes from four things: model mismatch, pipeline mismatch, evaluation mismatch, and runtime limits.

Llama 3 helped narrow the model question, but it did not remove the bigger issue that our setup still was not identical to the one in the paper.

So the conclusion we can defend is setup-sensitive failure to reproduce, not falsification."

---

## Slide 9: Final Conclusion
**Presenter:** Chris  
**Target time:** ~1 minute 35 seconds

**Script**

"So our final conclusion is pretty direct.

We reproduced the experimental scaffold, ran real local evaluations, and added one defense condition plus two additional model checks.

But we did not reproduce the paper's strong FITD jailbreak effect.

On Qwen, the only apparent positives disappeared after human review of the flagged outputs.

On Gemma 4 and Llama 3, everything was refused.

That puts our project in the rigorous failed-reproduction category, which the assignment explicitly allows. We think it is still a useful result because it shows how much the answer depends on setup and evaluation."

---

## Slide 10: If We Had More Time
**Presenter:** Chris  
**Target time:** ~40 seconds

**Script**

"If we had more time, the next steps are clear: run a larger slice or the full benchmark, match the authors' pipeline more closely, and use a stronger judge.

But for this class project, the important part is that we ran the core comparison honestly, checked the outputs carefully, and followed the evidence where it led.

Thanks."
