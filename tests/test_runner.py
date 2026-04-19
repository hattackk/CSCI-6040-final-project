from __future__ import annotations

import json

from fitd_repro import runner
from fitd_repro.types import PromptExample


def test_run_experiment_counts_errors_in_summary(monkeypatch, tmp_path):
    examples = [
        PromptExample(example_id="1", goal="goal 1", target="target 1"),
        PromptExample(example_id="2", goal="goal 2", target="target 2"),
    ]

    monkeypatch.setattr(runner, "build_model", lambda **kwargs: object())
    monkeypatch.setattr(runner, "load_examples", lambda **kwargs: examples)

    def fake_run_single_example(**kwargs):
        example = kwargs["example"]
        if example.example_id == "2":
            raise RuntimeError("synthetic failure")
        return {
            "example_id": example.example_id,
            "goal": example.goal,
            "target": example.target,
            "attack": kwargs["attack_name"],
            "defense": kwargs["defense_name"],
            "warmup_trace": [],
            "final_prompt": "prompt",
            "final_response": "Mock unsafe completion [redacted for safety].",
            "evaluation": {
                "success": True,
                "refusal": False,
                "response_chars": 45,
            },
        }

    monkeypatch.setattr(runner, "_run_single_example", fake_run_single_example)

    summary = runner.run_experiment(
        dataset_path="data/sample.csv",
        backend="mock",
        model_name="mock-model",
        attack="fitd",
        defense="none",
        fitd_variant="scaffold",
        author_prompt_file=None,
        author_prompt_track="prompts1",
        author_max_warmup_turns=None,
        author_target_mode="softened",
        output_dir=str(tmp_path / "results"),
        max_examples=2,
        start_index=0,
        max_tokens=64,
        temperature=0.0,
        sleep_seconds=0.0,
    )

    assert summary["total_examples"] == 2
    assert summary["completed_examples"] == 1
    assert summary["error_count"] == 1
    assert summary["successes"] == 1
    assert summary["refusals"] == 0
    assert summary["failures"] == 1
    assert summary["asr"] == 0.5
    assert summary["refusal_rate"] == 0.0

    records_path = tmp_path / "results" / "records.jsonl"
    errors_path = tmp_path / "results" / "errors.jsonl"
    summary_path = tmp_path / "results" / "summary.json"

    assert records_path.exists()
    assert errors_path.exists()
    assert summary_path.exists()

    error_lines = [line for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(error_lines) == 1
    payload = json.loads(error_lines[0])
    assert payload["example_id"] == "2"
    assert payload["error"] == "synthetic failure"

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["error_count"] == 1
    assert summary_payload["author_target_mode"] == "softened"
