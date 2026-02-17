from __future__ import annotations

import json

import pytest

from fitd_repro.dashboard_server import DashboardState, RunState


def _write_dataset(tmp_path):
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text("goal,target\nplaceholder,placeholder\n", encoding="utf-8")
    return dataset_path


def test_normalize_settings_blocks_openai_without_opt_in(monkeypatch, tmp_path):
    dataset_path = _write_dataset(tmp_path)
    monkeypatch.delenv("FITD_ALLOW_OPENAI", raising=False)

    state = DashboardState(project_root=tmp_path)

    with pytest.raises(ValueError, match="disabled by default"):
        state._normalize_settings(
            {
                "backend": "openai",
                "model": "gpt-4o-mini",
                "dataset_path": str(dataset_path),
                "max_examples": 2,
                "max_tokens": 64,
            }
        )


def test_normalize_settings_allows_openai_when_opted_in(monkeypatch, tmp_path):
    dataset_path = _write_dataset(tmp_path)
    monkeypatch.setenv("FITD_ALLOW_OPENAI", "1")

    state = DashboardState(project_root=tmp_path)
    settings = state._normalize_settings(
        {
            "backend": "openai",
            "model": "gpt-4o-mini",
            "dataset_path": str(dataset_path),
            "max_examples": 2,
            "max_tokens": 64,
        }
    )

    assert settings["backend"] == "openai"


def test_run_trace_can_hide_or_reveal_responses(tmp_path):
    dataset_path = _write_dataset(tmp_path)
    state = DashboardState(project_root=tmp_path)

    output_dir = tmp_path / "results" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp_utc": "2026-02-17T00:00:00Z",
        "example_id": "1",
        "goal": "example goal",
        "turn_kind": "final",
        "turn_index": 1,
        "user_prompt": "test prompt",
        "assistant_response": "test response",
        "assistant_refusal": False,
        "assistant_success_heuristic": True,
    }
    (output_dir / "turn_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    run = RunState(
        run_id="run_test",
        batch_id="batch_test",
        label="FITD",
        attack="fitd",
        defense="none",
        fitd_variant="scaffold",
        author_prompt_file=None,
        author_prompt_track="prompts1",
        author_max_warmup_turns=None,
        backend="mock",
        model="mock-model",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        log_path=str(tmp_path / "logs" / "run_test.jsonl"),
        max_examples=1,
        max_tokens=64,
        temperature=0.0,
        sleep_seconds=0.0,
    )

    with state._lock:
        state._runs[run.run_id] = run

    hidden = state.run_trace(run_id="run_test", include_response=False)
    assert hidden["events"][0]["assistant_response"].startswith("[hidden")

    visible = state.run_trace(run_id="run_test", include_response=True)
    assert visible["events"][0]["assistant_response"] == "test response"


def test_handle_run_event_writes_log_file(tmp_path):
    dataset_path = _write_dataset(tmp_path)
    state = DashboardState(project_root=tmp_path)

    run = RunState(
        run_id="run_test",
        batch_id="batch_test",
        label="FITD",
        attack="fitd",
        defense="none",
        fitd_variant="scaffold",
        author_prompt_file=None,
        author_prompt_track="prompts1",
        author_max_warmup_turns=None,
        backend="mock",
        model="mock-model",
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "results" / "test_run"),
        log_path=str(tmp_path / "logs" / "run_test.jsonl"),
        max_examples=1,
        max_tokens=64,
        temperature=0.0,
        sleep_seconds=0.0,
    )

    with state._lock:
        state._runs[run.run_id] = run

    state._handle_run_event(
        run_id="run_test",
        event={
            "type": "phase",
            "phase": "running",
            "timestamp_utc": "2026-02-17T00:00:00Z",
        },
    )

    log_path = tmp_path / "logs" / "run_test.jsonl"
    assert log_path.exists()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "phase"
    assert payload["run_id"] == "run_test"
