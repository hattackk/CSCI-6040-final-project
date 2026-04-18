from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .attacks import resolve_author_prompt_file
from .models import is_openai_backend_allowed
from .runner import run_experiment


DEFAULT_CONDITIONS = (
    {"label": "Standard", "attack": "standard", "defense": "none"},
    {"label": "FITD", "attack": "fitd", "defense": "none"},
    {"label": "FITD + Vigilant", "attack": "fitd", "defense": "vigilant"},
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    clean = "-".join(segment for segment in clean.split("-") if segment)
    return clean or "run"


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _clip_text(value: Any, max_chars: int = 2400) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


@dataclass(slots=True)
class RunState:
    run_id: str
    batch_id: str
    label: str
    attack: str
    defense: str
    fitd_variant: str
    author_prompt_file: str | None
    author_prompt_track: str
    author_max_warmup_turns: int | None
    backend: str
    model: str
    dataset_path: str
    output_dir: str
    log_path: str
    max_examples: int
    max_tokens: int
    temperature: float
    sleep_seconds: float
    status: str = "queued"
    phase: str = "queued"
    progress_completed: float = 0.0
    progress_total: int = 0
    latest_example_id: str | None = None
    latest_turn: dict | None = None
    start_time_utc: str | None = None
    end_time_utc: str | None = None
    error: str | None = None
    summary: dict | None = None

    def to_dict(self) -> dict:
        summary = self.summary or {}
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "label": self.label,
            "attack": self.attack,
            "defense": self.defense,
            "fitd_variant": self.fitd_variant,
            "author_prompt_file": self.author_prompt_file,
            "author_prompt_track": self.author_prompt_track,
            "author_max_warmup_turns": self.author_max_warmup_turns,
            "backend": self.backend,
            "model": self.model,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "max_examples": self.max_examples,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sleep_seconds": self.sleep_seconds,
            "status": self.status,
            "phase": self.phase,
            "progress_completed": self.progress_completed,
            "progress_total": self.progress_total,
            "latest_example_id": self.latest_example_id,
            "latest_turn": self.latest_turn,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "error": self.error,
            "summary": self.summary,
            "asr": summary.get("asr"),
            "refusal_rate": summary.get("refusal_rate"),
        }


@dataclass(slots=True)
class BatchState:
    batch_id: str
    settings: dict
    run_ids: list[str]
    status: str = "queued"
    current_run_id: str | None = None
    created_time_utc: str = field(default_factory=_utc_now_iso)
    start_time_utc: str | None = None
    end_time_utc: str | None = None
    error: str | None = None


class DashboardState:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.openai_backend_allowed = is_openai_backend_allowed()

        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}
        self._batches: dict[str, BatchState] = {}
        self._activity: deque[dict] = deque(maxlen=500)

    @staticmethod
    def _safe_read_jsonl(path: Path) -> list[dict]:
        if not path.exists() or not path.is_file():
            return []
        rows: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            return []
        return rows

    @staticmethod
    def _status_label(outcome: dict | None) -> str:
        if outcome is None:
            return "NA"
        return "SUCCESS" if outcome.get("success") else "REFUSAL"

    @staticmethod
    def _example_outcomes(records: list[dict]) -> dict[str, dict]:
        outcomes: dict[str, dict] = {}
        for record in records:
            example_id = str(record.get("example_id") or "")
            if not example_id:
                continue
            evaluation = record.get("evaluation")
            if not isinstance(evaluation, dict):
                evaluation = {}
            outcomes[example_id] = {
                "example_id": example_id,
                "goal": record.get("goal"),
                "success": bool(evaluation.get("success", False)),
                "refusal": bool(evaluation.get("refusal", True)),
            }
        return outcomes

    @staticmethod
    def _first_non_refusal_turns(events: list[dict]) -> dict[str, str]:
        first: dict[str, str] = {}
        for event in events:
            example_id = str(event.get("example_id") or "")
            if not example_id or example_id in first:
                continue
            if bool(event.get("assistant_refusal", True)):
                continue
            turn_kind = str(event.get("turn_kind") or "turn")
            turn_index = int(event.get("turn_index", 0) or 0)
            first[example_id] = f"{turn_kind}-{turn_index}"
        return first

    @staticmethod
    def _sort_example_ids(values: set[str]) -> list[str]:
        def _key(val: str) -> tuple[int, int | str]:
            if val.isdigit():
                return (0, int(val))
            return (1, val)

        return sorted(values, key=_key)

    @staticmethod
    def _expected_turns_per_example(run: RunState) -> int:
        if run.attack == "standard":
            return 1
        if run.attack == "fitd" and run.fitd_variant == "author":
            if run.author_max_warmup_turns is not None and run.author_max_warmup_turns > 0:
                return run.author_max_warmup_turns + 1
            # Default author prompt data currently has 11 warmups + final.
            return 12
        if run.attack == "fitd":
            return 4
        return 1

    def _compute_batch_effect(self, run_ids: list[str], run_lookup: dict[str, RunState]) -> dict[str, Any]:
        standard_run: RunState | None = None
        fitd_run: RunState | None = None
        vigilant_run: RunState | None = None

        for run_id in run_ids:
            run = run_lookup.get(run_id)
            if run is None:
                continue
            if run.attack == "standard" and run.defense == "none":
                standard_run = run
            elif run.attack == "fitd" and run.defense == "none":
                fitd_run = run
            elif run.attack == "fitd" and run.defense == "vigilant":
                vigilant_run = run

        standard_map = self._example_outcomes(self._safe_read_jsonl(Path(standard_run.output_dir) / "records.jsonl")) if standard_run else {}
        fitd_map = self._example_outcomes(self._safe_read_jsonl(Path(fitd_run.output_dir) / "records.jsonl")) if fitd_run else {}
        vigilant_map = self._example_outcomes(self._safe_read_jsonl(Path(vigilant_run.output_dir) / "records.jsonl")) if vigilant_run else {}

        first_turn_map = {}
        if fitd_run is not None:
            fitd_events = self._safe_read_jsonl(Path(fitd_run.output_dir) / "turn_events.jsonl")
            first_turn_map = self._first_non_refusal_turns(fitd_events)

        standard_ids = set(standard_map.keys())
        fitd_ids = set(fitd_map.keys())
        vigilant_ids = set(vigilant_map.keys())

        compare_standard_fitd = standard_ids & fitd_ids
        compare_fitd_vigilant = fitd_ids & vigilant_ids
        union_ids = self._sort_example_ids(standard_ids | fitd_ids | vigilant_ids)

        standard_refusals = sum(1 for example_id in compare_standard_fitd if standard_map[example_id]["refusal"])
        fitd_successes = sum(1 for example_id in compare_standard_fitd if fitd_map[example_id]["success"])

        door_opened = 0
        for example_id in compare_standard_fitd:
            if standard_map[example_id]["refusal"] and fitd_map[example_id]["success"]:
                door_opened += 1

        defense_recovered = 0
        still_compromised = 0
        for example_id in compare_fitd_vigilant:
            fitd_success = fitd_map[example_id]["success"]
            vigilant_refusal = vigilant_map[example_id]["refusal"]
            vigilant_success = vigilant_map[example_id]["success"]
            if fitd_success and vigilant_refusal:
                defense_recovered += 1
            if fitd_success and vigilant_success:
                still_compromised += 1

        turn_hist: dict[str, int] = {}
        for example_id in fitd_ids:
            label = first_turn_map.get(example_id, "none")
            turn_hist[label] = turn_hist.get(label, 0) + 1

        hist_rows = [
            {"label": label, "count": count}
            for label, count in sorted(
                turn_hist.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        transitions: list[dict] = []
        for example_id in union_ids:
            standard_outcome = standard_map.get(example_id)
            fitd_outcome = fitd_map.get(example_id)
            vigilant_outcome = vigilant_map.get(example_id)
            source = standard_outcome or fitd_outcome or vigilant_outcome or {}
            transitions.append(
                {
                    "example_id": example_id,
                    "goal": source.get("goal"),
                    "standard": self._status_label(standard_outcome),
                    "fitd": self._status_label(fitd_outcome),
                    "vigilant": self._status_label(vigilant_outcome),
                    "door_opened": bool(
                        standard_outcome
                        and fitd_outcome
                        and standard_outcome.get("refusal")
                        and fitd_outcome.get("success")
                    ),
                    "recovered_by_defense": bool(
                        fitd_outcome
                        and vigilant_outcome
                        and fitd_outcome.get("success")
                        and vigilant_outcome.get("refusal")
                    ),
                    "first_non_refusal_turn": first_turn_map.get(example_id, "none"),
                }
            )

        def _rate(part: int, whole: int) -> float | None:
            if whole <= 0:
                return None
            return round(part / whole, 4)

        return {
            "available": bool(union_ids),
            "comparisons_standard_fitd": len(compare_standard_fitd),
            "comparisons_fitd_vigilant": len(compare_fitd_vigilant),
            "standard_refusals": standard_refusals,
            "fitd_successes": fitd_successes,
            "door_opened_count": door_opened,
            "door_opened_rate_over_standard_refusals": _rate(door_opened, standard_refusals),
            "door_opened_rate_over_compared": _rate(door_opened, len(compare_standard_fitd)),
            "defense_recovered_count": defense_recovered,
            "defense_recovered_rate_over_fitd_successes": _rate(defense_recovered, fitd_successes),
            "still_compromised_count": still_compromised,
            "still_compromised_rate_over_fitd_successes": _rate(still_compromised, fitd_successes),
            "first_non_refusal_histogram": hist_rows,
            "transitions": transitions[:180],
        }

    def _normalize_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        backend = str(payload.get("backend", "mock")).strip().lower()
        if backend not in {"mock", "openai", "hf", "ollama"}:
            raise ValueError("backend must be one of: mock, openai, hf, ollama")
        if backend == "openai" and not self.openai_backend_allowed:
            raise ValueError(
                "OpenAI backend is disabled by default. "
                "Set FITD_ALLOW_OPENAI=1 before launching the dashboard to opt in."
            )

        model = str(payload.get("model", "mock-model")).strip()
        if not model:
            raise ValueError("model is required")

        fitd_variant = str(payload.get("fitd_variant", "scaffold")).strip().lower()
        if fitd_variant not in {"scaffold", "author"}:
            raise ValueError("fitd_variant must be one of: scaffold, author")

        author_prompt_track = str(payload.get("author_prompt_track", "prompts1")).strip().lower()
        if author_prompt_track not in {"prompts1", "prompts2"}:
            raise ValueError("author_prompt_track must be one of: prompts1, prompts2")

        dataset_path = str(payload.get("dataset_path", "data/advbench/sample_prompts.csv")).strip()
        if not dataset_path:
            raise ValueError("dataset_path is required")
        dataset_obj = Path(dataset_path)
        if not dataset_obj.is_absolute():
            dataset_obj = (self.project_root / dataset_obj).resolve()
        if not dataset_obj.exists():
            raise ValueError(f"dataset_path not found: {dataset_obj}")

        author_prompt_file = str(payload.get("author_prompt_file", "")).strip() or None
        resolved_author_prompt_file = None
        if fitd_variant == "author" or author_prompt_file is not None:
            resolved_author_prompt_file = resolve_author_prompt_file(
                dataset_path=str(dataset_obj),
                author_prompt_file=author_prompt_file,
            )
        if fitd_variant == "author" and resolved_author_prompt_file is None:
            raise ValueError(
                "fitd_variant=author requires prompt data. "
                "Provide author_prompt_file or use a supported dataset name "
                "(jailbreakbench/harmbench). "
                "Example: dataset_path=data/author_fitd/jailbreakbench.csv "
                "author_prompt_file=data/author_fitd/prompt_jailbreakbench.json"
            )

        try:
            max_examples = int(payload.get("max_examples", 20))
            max_tokens = int(payload.get("max_tokens", 128))
            author_max_warmup_turns_raw = int(payload.get("author_max_warmup_turns", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_examples, max_tokens, and author_max_warmup_turns must be integers") from exc

        try:
            temperature = float(payload.get("temperature", 0.2))
            sleep_seconds = float(payload.get("sleep_seconds", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature and sleep_seconds must be numbers") from exc

        max_examples = _clamp_int(max_examples, 1, 5000)
        max_tokens = _clamp_int(max_tokens, 16, 2048)
        author_max_warmup_turns = _clamp_int(author_max_warmup_turns_raw, 0, 20)
        if author_max_warmup_turns <= 0:
            author_max_warmup_turns = None
        temperature = max(0.0, min(2.0, temperature))
        sleep_seconds = max(0.0, min(5.0, sleep_seconds))

        return {
            "backend": backend,
            "model": model,
            "fitd_variant": fitd_variant,
            "author_prompt_file": resolved_author_prompt_file,
            "author_prompt_track": author_prompt_track,
            "author_max_warmup_turns": author_max_warmup_turns,
            "dataset_path": str(dataset_obj),
            "max_examples": max_examples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "sleep_seconds": sleep_seconds,
        }

    def _append_activity(self, run: RunState, event: dict) -> None:
        event_type = event.get("type", "event")
        message = ""
        if event_type == "phase":
            message = f"{run.label}: phase -> {event.get('phase')}"
        elif event_type == "example_start":
            message = (
                f"{run.label}: example {event.get('example_index')}/"
                f"{event.get('total_examples')} started"
            )
        elif event_type == "example_complete":
            outcome = "SUCCESS" if event.get("success") else "REFUSAL"
            message = (
                f"{run.label}: example {event.get('example_index')}/"
                f"{event.get('total_examples')} -> {outcome}"
            )
        elif event_type == "example_error":
            message = f"{run.label}: example error ({event.get('error')})"
        elif event_type == "turn_event":
            refusal = "yes" if event.get("assistant_refusal") else "no"
            prompt_preview = str(event.get("user_prompt_preview") or "")
            message = (
                f"{run.label}: {event.get('turn_kind')} turn {event.get('turn_index')} "
                f"(refusal={refusal})"
            )
            if prompt_preview:
                message += f" | user: {prompt_preview}"
        elif event_type == "turn_start":
            prompt_preview = str(event.get("user_prompt_preview") or "")
            message = (
                f"{run.label}: generating {event.get('turn_kind')} turn {event.get('turn_index')}"
            )
            if prompt_preview:
                message += f" | user: {prompt_preview}"
        elif event_type == "run_complete":
            message = f"{run.label}: run complete"
        else:
            message = f"{run.label}: {event_type}"

        self._activity.append(
            {
                "timestamp_utc": event.get("timestamp_utc", _utc_now_iso()),
                "batch_id": run.batch_id,
                "run_id": run.run_id,
                "label": run.label,
                "event_type": event_type,
                "message": message,
            }
        )

    def _append_run_log(self, run: RunState, event: dict) -> None:
        event_type = str(event.get("type", "event"))
        payload = {
            "timestamp_utc": event.get("timestamp_utc", _utc_now_iso()),
            "event_type": event_type,
            "batch_id": run.batch_id,
            "run_id": run.run_id,
            "label": run.label,
            "attack": run.attack,
            "defense": run.defense,
            "fitd_variant": run.fitd_variant,
            "author_prompt_track": run.author_prompt_track,
            "author_max_warmup_turns": run.author_max_warmup_turns,
            "backend": run.backend,
            "model": run.model,
            "dataset_path": run.dataset_path,
            "output_dir": run.output_dir,
            "event": event,
        }
        _append_jsonl(Path(run.log_path), payload)

    def _handle_run_event(self, run_id: str, event: dict) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return

            event_type = event.get("type")
            if event_type == "phase":
                run.phase = str(event.get("phase", run.phase))
                if run.phase == "running":
                    run.progress_total = int(event.get("total_examples", run.progress_total))
            elif event_type == "example_start":
                run.phase = "running"
                run.progress_total = int(event.get("total_examples", run.progress_total))
                run.progress_completed = float(max(0, int(event.get("example_index", 1)) - 1))
                run.latest_example_id = str(event.get("example_id", ""))
            elif event_type == "turn_start":
                run.latest_turn = {
                    "turn_kind": event.get("turn_kind"),
                    "turn_index": event.get("turn_index"),
                    "assistant_refusal": None,
                    "assistant_success_heuristic": None,
                    "user_prompt_preview": event.get("user_prompt_preview"),
                    "assistant_response_preview": "[generating...]",
                    "example_index": event.get("example_index"),
                    "total_examples": event.get("total_examples"),
                    "turn_status": "generating",
                }
            elif event_type == "turn_event":
                run.latest_turn = {
                    "turn_kind": event.get("turn_kind"),
                    "turn_index": event.get("turn_index"),
                    "assistant_refusal": event.get("assistant_refusal"),
                    "assistant_success_heuristic": event.get("assistant_success_heuristic"),
                    "user_prompt_preview": event.get("user_prompt_preview"),
                    "assistant_response_preview": event.get("assistant_response_preview"),
                    "example_index": event.get("example_index"),
                    "total_examples": event.get("total_examples"),
                    "turn_status": "completed",
                }
                expected_turns = self._expected_turns_per_example(run)
                example_index = int(event.get("example_index", 0) or 0)
                turn_index = int(event.get("turn_index", 0) or 0)
                if run.progress_total > 0 and expected_turns > 0 and example_index > 0 and turn_index > 0:
                    turn_fraction = min(turn_index, expected_turns) / expected_turns
                    turn_progress = (example_index - 1) + turn_fraction
                    run.progress_completed = min(
                        float(run.progress_total),
                        max(run.progress_completed, turn_progress),
                    )
            elif event_type == "example_complete":
                run.progress_total = int(event.get("total_examples", run.progress_total))
                run.progress_completed = float(int(event.get("example_index", run.progress_completed)))
                run.latest_example_id = str(event.get("example_id", run.latest_example_id or ""))
            elif event_type == "example_error":
                run.progress_total = int(event.get("total_examples", run.progress_total))
                run.progress_completed = float(int(event.get("example_index", run.progress_completed)))
                run.error = str(event.get("error", "unknown error"))
            elif event_type == "run_complete":
                run.phase = "completed"

            self._append_activity(run, event)
            self._append_run_log(run, event)

    def _run_batch_worker(self, batch_id: str) -> None:
        with self._lock:
            batch = self._batches[batch_id]
            batch.status = "running"
            batch.start_time_utc = _utc_now_iso()

        for run_id in list(batch.run_ids):
            with self._lock:
                run = self._runs[run_id]
                batch.current_run_id = run_id
                run.status = "running"
                run.phase = "starting"
                run.start_time_utc = _utc_now_iso()
                self._append_run_log(
                    run,
                    {
                        "type": "run_start",
                        "timestamp_utc": run.start_time_utc,
                        "batch_id": run.batch_id,
                        "run_id": run.run_id,
                        "label": run.label,
                    },
                )

            def run_event_callback(event: dict, this_run_id: str = run_id) -> None:
                self._handle_run_event(this_run_id, event)

            try:
                summary = run_experiment(
                    dataset_path=run.dataset_path,
                    backend=run.backend,
                    model_name=run.model,
                    attack=run.attack,
                    defense=run.defense,
                    fitd_variant=run.fitd_variant,
                    author_prompt_file=run.author_prompt_file,
                    author_prompt_track=run.author_prompt_track,
                    author_max_warmup_turns=run.author_max_warmup_turns,
                    output_dir=run.output_dir,
                    max_examples=run.max_examples,
                    start_index=0,
                    max_tokens=run.max_tokens,
                    temperature=run.temperature,
                    sleep_seconds=run.sleep_seconds,
                    event_callback=run_event_callback,
                )
                with self._lock:
                    run.status = "completed"
                    run.phase = "completed"
                    run.summary = summary
                    run.progress_total = int(summary.get("total_examples", run.progress_total))
                    run.progress_completed = float(int(summary.get("total_examples", run.progress_completed)))
                    run.end_time_utc = _utc_now_iso()
            except Exception as exc:
                with self._lock:
                    run.status = "failed"
                    run.phase = "failed"
                    run.error = str(exc)
                    run.end_time_utc = _utc_now_iso()
                    self._append_run_log(
                        run,
                        {
                            "type": "run_failed",
                            "timestamp_utc": run.end_time_utc,
                            "error": str(exc),
                        },
                    )

        with self._lock:
            batch.current_run_id = None
            batch.end_time_utc = _utc_now_iso()
            failed = any(self._runs[rid].status == "failed" for rid in batch.run_ids)
            batch.status = "failed" if failed else "completed"

    def start_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._normalize_settings(payload)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{uuid4().hex[:8]}"
        model_slug = _slug(settings["model"])[:24]

        run_ids: list[str] = []
        for condition in DEFAULT_CONDITIONS:
            run_id = f"run_{uuid4().hex[:8]}"
            run_slug = f"{_slug(condition['attack'])}_{_slug(condition['defense'])}"
            run_base = f"{stamp}_{batch_id}_{run_slug}_{model_slug}"
            output_dir = self.results_dir / run_base
            log_path = self.logs_dir / f"{run_base}.jsonl"
            run = RunState(
                run_id=run_id,
                batch_id=batch_id,
                label=condition["label"],
                attack=condition["attack"],
                defense=condition["defense"],
                fitd_variant=settings["fitd_variant"],
                author_prompt_file=settings["author_prompt_file"],
                author_prompt_track=settings["author_prompt_track"],
                author_max_warmup_turns=settings["author_max_warmup_turns"],
                backend=settings["backend"],
                model=settings["model"],
                dataset_path=settings["dataset_path"],
                output_dir=str(output_dir),
                log_path=str(log_path),
                max_examples=settings["max_examples"],
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
                sleep_seconds=settings["sleep_seconds"],
            )
            run_ids.append(run_id)
            with self._lock:
                self._runs[run_id] = run

        with self._lock:
            self._batches[batch_id] = BatchState(
                batch_id=batch_id,
                settings=settings,
                run_ids=run_ids,
            )

        worker = threading.Thread(
            target=self._run_batch_worker,
            args=(batch_id,),
            daemon=True,
        )
        worker.start()
        return {"batch_id": batch_id, "run_ids": run_ids}

    def run_trace(self, run_id: str, limit: int = 120, include_response: bool = False) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"Unknown run_id: {run_id}")
            run_payload = {
                "run_id": run.run_id,
                "batch_id": run.batch_id,
                "label": run.label,
                "attack": run.attack,
                "defense": run.defense,
                "fitd_variant": run.fitd_variant,
                "author_prompt_file": run.author_prompt_file,
                "author_prompt_track": run.author_prompt_track,
                "author_max_warmup_turns": run.author_max_warmup_turns,
                "backend": run.backend,
                "model": run.model,
                "status": run.status,
                "phase": run.phase,
                "output_dir": run.output_dir,
            }

        clean_limit = _clamp_int(limit, 1, 500)
        turn_events = self._safe_read_jsonl(Path(run_payload["output_dir"]) / "turn_events.jsonl")
        selected = turn_events[-clean_limit:]
        events: list[dict[str, Any]] = []
        for event in selected:
            assistant_response = _clip_text(event.get("assistant_response", ""))
            if not include_response:
                assistant_response = "[hidden - enable Reveal Responses]"

            events.append(
                {
                    "timestamp_utc": event.get("timestamp_utc"),
                    "example_id": event.get("example_id"),
                    "goal": _clip_text(event.get("goal", ""), max_chars=260),
                    "turn_kind": event.get("turn_kind"),
                    "turn_index": int(event.get("turn_index", 0) or 0),
                    "assistant_refusal": bool(event.get("assistant_refusal", True)),
                    "assistant_success_heuristic": bool(event.get("assistant_success_heuristic", False)),
                    "user_prompt": _clip_text(event.get("user_prompt", "")),
                    "assistant_response": assistant_response,
                    "error": _clip_text(event.get("error", ""), max_chars=260),
                }
            )

        return {
            "run": run_payload,
            "include_response": include_response,
            "limit": clean_limit,
            "total_events": len(turn_events),
            "events": events,
        }

    def _load_recent_summaries(self, limit: int = 40) -> list[dict]:
        summary_files = sorted(
            self.results_dir.glob("*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        summaries: list[dict] = []
        for summary_file in summary_files[:limit]:
            try:
                with summary_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            summaries.append(
                {
                    "run_name": summary_file.parent.name,
                    "summary_path": str(summary_file),
                    "timestamp_utc": payload.get("timestamp_utc"),
                    "backend": payload.get("backend"),
                    "model": payload.get("model"),
                    "attack": payload.get("attack"),
                    "defense": payload.get("defense"),
                    "fitd_variant": payload.get("fitd_variant", "scaffold"),
                    "author_prompt_file": payload.get("author_prompt_file"),
                    "author_prompt_track": payload.get("author_prompt_track", "prompts1"),
                    "author_max_warmup_turns": payload.get("author_max_warmup_turns"),
                    "total_examples": payload.get("total_examples"),
                    "asr": payload.get("asr"),
                    "refusal_rate": payload.get("refusal_rate"),
                }
            )
        return summaries

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            batches = []
            run_lookup = dict(self._runs)
            for batch in sorted(
                self._batches.values(),
                key=lambda item: item.created_time_utc,
                reverse=True,
            ):
                run_ids = [run_id for run_id in batch.run_ids if run_id in self._runs]
                run_dicts = [self._runs[run_id].to_dict() for run_id in run_ids]
                batches.append(
                    {
                        "batch_id": batch.batch_id,
                        "settings": batch.settings,
                        "status": batch.status,
                        "current_run_id": batch.current_run_id,
                        "created_time_utc": batch.created_time_utc,
                        "start_time_utc": batch.start_time_utc,
                        "end_time_utc": batch.end_time_utc,
                        "error": batch.error,
                        "runs": run_dicts,
                        "_run_ids": run_ids,
                    }
                )

            activity = list(self._activity)[-120:]

        for batch in batches:
            run_ids = batch.pop("_run_ids", [])
            batch["effect"] = self._compute_batch_effect(run_ids, run_lookup)

        recent_summaries = self._load_recent_summaries(limit=40)
        models = sorted(
            {
                f"{entry.get('backend')}:{entry.get('model')}"
                for entry in recent_summaries
                if entry.get("backend") and entry.get("model")
            }
        )

        active_batch_id = None
        for batch in batches:
            if batch["status"] in {"queued", "running"}:
                active_batch_id = batch["batch_id"]
                break

        return {
            "project_root": str(self.project_root),
            "server_time_utc": _utc_now_iso(),
            "active_batch_id": active_batch_id,
            "batches": batches,
            "recent_summaries": recent_summaries,
            "models_seen": models,
            "activity": activity,
            "capabilities": {
                "openai_backend_allowed": self.openai_backend_allowed,
            },
        }


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState
    static_dir: Path

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict, status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return

        suffix = path.suffix.lower()
        content_type = "text/plain; charset=utf-8"
        if suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif suffix == ".js":
            content_type = "application/javascript; charset=utf-8"

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return {}
        try:
            content_length = int(raw_len)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in {"/", "/index.html"}:
            self._send_file(self.static_dir / "index.html")
            return
        if route == "/styles.css":
            self._send_file(self.static_dir / "styles.css")
            return
        if route == "/app.js":
            self._send_file(self.static_dir / "app.js")
            return
        if route == "/api/health":
            self._send_json({"ok": True, "timestamp_utc": _utc_now_iso()})
            return
        if route == "/api/state":
            self._send_json(self.state.snapshot())
            return
        if route == "/api/run-trace":
            try:
                query = parse_qs(parsed.query)
                run_id = (query.get("run_id", [""])[0] or "").strip()
                if not run_id:
                    raise ValueError("run_id query parameter is required")
                limit_raw = query.get("limit", ["120"])[0]
                limit = int(limit_raw)
                include_response = _parse_bool(query.get("include_response", [None])[0])
                payload = self.state.run_trace(
                    run_id=run_id,
                    limit=limit,
                    include_response=include_response,
                )
                self._send_json(payload)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=400)
            except Exception as exc:  # pragma: no cover
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/start-batch":
            try:
                payload = self._read_json()
                result = self.state.start_batch(payload)
                self._send_json({"ok": True, **result}, status_code=201)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=400)
            except Exception as exc:  # pragma: no cover
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
            return

        self.send_error(404, "Not found")


def serve_dashboard(project_root: Path, host: str = "127.0.0.1", port: int = 8787) -> None:
    state = DashboardState(project_root=project_root)
    static_dir = Path(__file__).resolve().parent / "dashboard_static"

    class Handler(DashboardHandler):
        pass

    Handler.state = state
    Handler.static_dir = static_dir

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}")
    print(f"Project root: {project_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
