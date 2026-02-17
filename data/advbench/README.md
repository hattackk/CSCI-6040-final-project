# AdvBench Data

Place your main project dataset file here:

- `data/advbench/harmful_behaviors.csv`

Supported formats: CSV, JSON, JSONL.

The loader auto-detects prompt columns named one of:
- `goal`, `prompt`, `instruction`, `query`, `request`

Optional columns:
- id column: `id`, `idx`, `example_id`, `qid`
- target column: `target`, `expected`, `reference`
