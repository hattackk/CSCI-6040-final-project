"""Retro-score a pre-judge run directory against the paper's §A.3.3 judge.

Usage:
    python -m fitd_repro.rescore <run-dir> [--judge-backend ...] [--judge-model ...]
                                           [--score-harmfulness]

Reads ``<run-dir>/records.jsonl``, calls the judge on every record's
``final_prompt`` / ``final_response``, rewrites the records with enriched
``evaluation`` blocks, and refreshes ``<run-dir>/summary.json`` so
dashboards and leaderboards pick up the judge-backed ASR. Idempotent:
records that already carry ``evaluation.judge_label`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import evaluate_response, summarize_records
from .judge import build_judge, judge_deviation_note, judge_paper_faithful


def _read_records(records_path: Path) -> list[dict]:
    records: list[dict] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_records(records_path: Path, records: list[dict]) -> None:
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitd_repro.rescore",
        description="Retroactively apply the §A.3.3 judge to an existing run directory.",
    )
    parser.add_argument("run_dir", type=str, help="Path to a run directory with records.jsonl + summary.json.")
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="openai",
        choices=["mock", "openai", "hf", "vllm", "ollama"],
        help="Judge backend (default: openai per paper §4.1).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model id (default: gpt-4o for openai, mock-judge for mock, else required).",
    )
    parser.add_argument(
        "--score-harmfulness",
        action="store_true",
        help="Also run the §A.3.6 harmfulness (1-5) judge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score records even if they already have judge_label.",
    )
    return parser


def _default_judge_model(backend: str) -> str | None:
    backend = backend.lower().strip()
    if backend == "openai":
        return "gpt-4o"
    if backend == "mock":
        return "mock-judge"
    return None


def rescore_run(
    run_dir: Path,
    *,
    judge_backend: str,
    judge_model: str | None,
    score_harmfulness: bool,
    force: bool,
) -> dict:
    run_dir = run_dir.resolve()
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found in {run_dir}")

    resolved_model = judge_model or _default_judge_model(judge_backend)
    if judge_backend != "mock" and not resolved_model:
        raise ValueError(
            f"--judge-model is required when --judge-backend={judge_backend!r}"
        )
    judge = build_judge(backend=judge_backend, model_name=resolved_model, role="judge")

    records = _read_records(records_path)
    rescored_count = 0
    skipped_count = 0
    for record in records:
        evaluation = record.get("evaluation") or {}
        if not force and "judge_label" in evaluation:
            skipped_count += 1
            continue
        goal = record.get("goal") or ""
        final_response = record.get("final_response") or ""
        updated = evaluate_response(
            final_response,
            goal=goal,
            judge=judge,
            score_harmfulness=score_harmfulness,
        )
        record["evaluation"] = updated
        rescored_count += 1

    _write_records(records_path, records)

    # Recompute summary with fresh aggregates, preserving the non-metric
    # configuration keys the runner originally wrote.
    previous_summary: dict = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
    total_examples = previous_summary.get("total_examples", len(records))
    refreshed_metrics = summarize_records(records, attempted_examples=total_examples)
    new_summary = dict(previous_summary)
    new_summary.update(refreshed_metrics)
    new_summary["judge_backend"] = judge_backend
    new_summary["judge_model"] = resolved_model
    new_summary["judge_paper_faithful"] = judge_paper_faithful(judge_backend, resolved_model)
    new_summary["judge_deviation"] = judge_deviation_note(judge_backend, resolved_model)
    new_summary["score_harmfulness"] = score_harmfulness
    new_summary["rescored"] = True
    new_summary["rescored_count"] = rescored_count
    new_summary["rescored_skipped"] = skipped_count

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(new_summary, handle, indent=2, ensure_ascii=True)

    return new_summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = rescore_run(
        run_dir=Path(args.run_dir),
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        score_harmfulness=args.score_harmfulness,
        force=args.force,
    )
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
