from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .attacks import AttackType, AuthorPromptTrack, DefenseType, FitdVariant, build_attack, resolve_author_prompt_file
from .dataset import load_examples
from .evaluation import evaluate_response, summarize_records
from .models import build_model
from .types import Message, PromptExample

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _emit_event(event_callback: Callable[[dict], None] | None, payload: dict) -> None:
    if event_callback is None:
        return
    try:
        event_callback(payload)
    except Exception:
        # Dashboard callbacks must never interrupt experiment execution.
        return


def _preview_text(text: str, max_chars: int = 220) -> str:
    clean = text.replace("\r", " ").replace("\n", " ").strip()
    if len(clean) <= max_chars:
        return clean
    return f"{clean[: max_chars - 3]}..."


def _log_turn_event(
    events_path: Path,
    example: PromptExample,
    attack_name: AttackType,
    defense_name: DefenseType,
    turn_kind: str,
    turn_index: int,
    user_prompt: str,
    assistant_response: str,
) -> None:
    evaluation = evaluate_response(assistant_response)
    _append_jsonl(
        events_path,
        {
            "timestamp_utc": _utc_now_iso(),
            "example_id": example.example_id,
            "goal": example.goal,
            "attack": attack_name,
            "defense": defense_name,
            "turn_kind": turn_kind,
            "turn_index": turn_index,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "assistant_refusal": evaluation["refusal"],
            "assistant_success_heuristic": evaluation["success"],
        },
    )


def _run_single_example(
    model,
    example: PromptExample,
    attack_name: AttackType,
    defense_name: DefenseType,
    fitd_variant: FitdVariant,
    dataset_path: str,
    author_prompt_file: str | None,
    author_prompt_track: AuthorPromptTrack,
    author_max_warmup_turns: int | None,
    max_tokens: int,
    temperature: float,
    sleep_seconds: float,
    turn_start_logger: Callable[[str, int, str], None] | None = None,
    turn_logger: Callable[[str, int, str, str], None] | None = None,
) -> dict:
    attack = build_attack(
        goal=example.goal,
        attack=attack_name,
        defense=defense_name,
        fitd_variant=fitd_variant,
        example_id=example.example_id,
        dataset_path=dataset_path,
        author_prompt_file=author_prompt_file,
        author_prompt_track=author_prompt_track,
        author_max_warmup_turns=author_max_warmup_turns,
    )
    messages: list[Message] = []
    turn_trace: list[dict] = []

    if attack.system_prompt:
        messages.append({"role": "system", "content": attack.system_prompt})

    for warmup_index, warmup_prompt in enumerate(attack.warmup_user_turns, start=1):
        messages.append({"role": "user", "content": warmup_prompt})
        if turn_start_logger is not None:
            turn_start_logger("warmup", warmup_index, warmup_prompt)
        assistant_response = model.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        messages.append({"role": "assistant", "content": assistant_response})
        turn_trace.append(
            {
                "user": warmup_prompt,
                "assistant": assistant_response,
            }
        )
        if turn_logger is not None:
            turn_logger("warmup", warmup_index, warmup_prompt, assistant_response)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    final_turn_index = len(attack.warmup_user_turns) + 1
    messages.append({"role": "user", "content": attack.final_user_turn})
    if turn_start_logger is not None:
        turn_start_logger("final", final_turn_index, attack.final_user_turn)
    final_response = model.generate(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if turn_logger is not None:
        turn_logger("final", final_turn_index, attack.final_user_turn, final_response)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    evaluation = evaluate_response(final_response)
    return {
        "example_id": example.example_id,
        "goal": example.goal,
        "target": example.target,
        "attack": attack_name,
        "defense": defense_name,
        "warmup_trace": turn_trace,
        "final_prompt": attack.final_user_turn,
        "final_response": final_response,
        "evaluation": evaluation,
    }


def run_experiment(
    dataset_path: str,
    backend: str,
    model_name: str,
    attack: AttackType,
    defense: DefenseType,
    fitd_variant: FitdVariant,
    author_prompt_file: str | None,
    author_prompt_track: AuthorPromptTrack,
    author_max_warmup_turns: int | None,
    output_dir: str,
    max_examples: int | None,
    start_index: int,
    max_tokens: int,
    temperature: float,
    sleep_seconds: float,
    event_callback: Callable[[dict], None] | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_author_prompt_file = author_prompt_file
    if attack == "fitd" and fitd_variant == "author":
        resolved_author_prompt_file = resolve_author_prompt_file(
            dataset_path=dataset_path,
            author_prompt_file=author_prompt_file,
        )
        if resolved_author_prompt_file is None:
            raise ValueError(
                "fitd_variant=author requires author prompt data. "
                "Provide --author-prompt-file or use a supported dataset filename "
                "(jailbreakbench/harmbench). "
                "Example: --dataset-path data/author_fitd/jailbreakbench.csv "
                "--author-prompt-file data/author_fitd/prompt_jailbreakbench.json"
            )

    records_path = out_dir / "records.jsonl"
    summary_path = out_dir / "summary.json"
    errors_path = out_dir / "errors.jsonl"
    events_path = out_dir / "turn_events.jsonl"

    if records_path.exists():
        records_path.unlink()
    if errors_path.exists():
        errors_path.unlink()
    if events_path.exists():
        events_path.unlink()

    _emit_event(
        event_callback,
        {
            "type": "phase",
            "phase": "loading_model",
            "timestamp_utc": _utc_now_iso(),
            "backend": backend,
            "model": model_name,
        },
    )
    model = build_model(backend=backend, model_name=model_name)
    _emit_event(
        event_callback,
        {
            "type": "phase",
            "phase": "loading_dataset",
            "timestamp_utc": _utc_now_iso(),
            "dataset_path": dataset_path,
        },
    )
    examples = load_examples(dataset_path=dataset_path, max_examples=max_examples, start_index=start_index)
    total_examples = len(examples)
    _emit_event(
        event_callback,
        {
            "type": "phase",
            "phase": "running",
            "timestamp_utc": _utc_now_iso(),
            "total_examples": total_examples,
        },
    )

    records: list[dict] = []
    for example_index, example in enumerate(tqdm(examples, desc="Running prompts"), start=1):
        try:
            _emit_event(
                event_callback,
                {
                    "type": "example_start",
                    "timestamp_utc": _utc_now_iso(),
                    "example_index": example_index,
                    "total_examples": total_examples,
                    "example_id": example.example_id,
                    "goal": example.goal,
                },
            )

            def turn_logger(kind: str, idx: int, user_prompt: str, assistant_response: str) -> None:
                _log_turn_event(
                    events_path=events_path,
                    example=example,
                    attack_name=attack,
                    defense_name=defense,
                    turn_kind=kind,
                    turn_index=idx,
                    user_prompt=user_prompt,
                    assistant_response=assistant_response,
                )
                evaluation = evaluate_response(assistant_response)
                _emit_event(
                    event_callback,
                    {
                        "type": "turn_event",
                        "timestamp_utc": _utc_now_iso(),
                        "example_index": example_index,
                        "total_examples": total_examples,
                        "example_id": example.example_id,
                        "turn_kind": kind,
                        "turn_index": idx,
                        "assistant_refusal": evaluation["refusal"],
                        "assistant_success_heuristic": evaluation["success"],
                        "user_prompt_preview": _preview_text(user_prompt),
                        "assistant_response_preview": _preview_text(assistant_response),
                    },
                )

            def turn_start_logger(kind: str, idx: int, user_prompt: str) -> None:
                _emit_event(
                    event_callback,
                    {
                        "type": "turn_start",
                        "timestamp_utc": _utc_now_iso(),
                        "example_index": example_index,
                        "total_examples": total_examples,
                        "example_id": example.example_id,
                        "turn_kind": kind,
                        "turn_index": idx,
                        "user_prompt_preview": _preview_text(user_prompt),
                    },
                )

            record = _run_single_example(
                model=model,
                example=example,
                attack_name=attack,
                defense_name=defense,
                fitd_variant=fitd_variant,
                dataset_path=dataset_path,
                author_prompt_file=resolved_author_prompt_file,
                author_prompt_track=author_prompt_track,
                author_max_warmup_turns=author_max_warmup_turns,
                max_tokens=max_tokens,
                temperature=temperature,
                sleep_seconds=sleep_seconds,
                turn_start_logger=turn_start_logger,
                turn_logger=turn_logger,
            )
            records.append(record)
            _append_jsonl(records_path, record)
            _emit_event(
                event_callback,
                {
                    "type": "example_complete",
                    "timestamp_utc": _utc_now_iso(),
                    "example_index": example_index,
                    "total_examples": total_examples,
                    "example_id": example.example_id,
                    "success": record["evaluation"]["success"],
                    "refusal": record["evaluation"]["refusal"],
                },
            )
        except Exception as exc:  # pragma: no cover
            _append_jsonl(
                errors_path,
                {
                    "example_id": example.example_id,
                    "goal": example.goal,
                    "error": str(exc),
                    "timestamp_utc": _utc_now_iso(),
                },
            )
            _append_jsonl(
                events_path,
                {
                    "timestamp_utc": _utc_now_iso(),
                    "example_id": example.example_id,
                    "goal": example.goal,
                    "attack": attack,
                    "defense": defense,
                    "turn_kind": "error",
                    "turn_index": 0,
                    "user_prompt": "",
                    "assistant_response": "",
                    "assistant_refusal": True,
                    "assistant_success_heuristic": False,
                    "error": str(exc),
                },
            )
            _emit_event(
                event_callback,
                {
                    "type": "example_error",
                    "timestamp_utc": _utc_now_iso(),
                    "example_index": example_index,
                    "total_examples": total_examples,
                    "example_id": example.example_id,
                    "error": str(exc),
                },
            )

    summary = summarize_records(records)
    summary.update(
        {
            "timestamp_utc": _utc_now_iso(),
            "dataset_path": dataset_path,
            "backend": backend,
            "model": model_name,
            "attack": attack,
            "defense": defense,
            "fitd_variant": fitd_variant,
            "author_prompt_file": resolved_author_prompt_file,
            "author_prompt_track": author_prompt_track,
            "author_max_warmup_turns": author_max_warmup_turns,
            "max_examples": max_examples,
            "start_index": start_index,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "sleep_seconds": sleep_seconds,
            "records_path": str(records_path),
            "errors_path": str(errors_path),
            "turn_events_path": str(events_path),
        }
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    _emit_event(
        event_callback,
        {
            "type": "run_complete",
            "timestamp_utc": _utc_now_iso(),
            "summary": summary,
        },
    )

    return summary
