# Final Project Milestones

## Scope

Paper: `"Foot-In-The-Door": Multi-turn Model Jailbreaking` (EMNLP 2025)

Primary goal:
- Reproduce core jailbreak success-rate claims for standard vs FITD prompts.

Extension goal:
- Test defensive system prompt ("vigilant" prompt) as mitigation.

## Milestone Checklist

1. Environment and data
- [ ] Confirm Python environment and dependencies
- [ ] Acquire AdvBench file and place under `data/advbench/`
- [ ] Run mock smoke test

2. Reproduction experiments
- [ ] Phase 1 API baseline (`gpt-4o-mini`)
- [ ] Phase 2 local model replication (Llama family)
- [ ] Save `summary.json` and `records.jsonl` for each run

3. Extension experiment
- [ ] Re-run FITD with `--defense vigilant`
- [ ] Compare ASR with no-defense condition

4. Analysis and presentation
- [ ] Validate heuristic labels on a sampled subset manually
- [ ] Build comparison table (standard vs FITD vs FITD+defense)
- [ ] Document reproducibility blockers and assumptions
- [ ] Prepare final 15-minute talk and final report

## Risks To Track

1. Dataset mismatch with original paper preprocessing
2. Model/version mismatch (API model drift and local checkpoints)
3. Heuristic evaluation error (false positives/negatives)
4. Compute limits for local replication
