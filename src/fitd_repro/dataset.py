from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import PromptExample


PROMPT_COLUMNS = ("goal", "prompt", "instruction", "query", "request")
TARGET_COLUMNS = ("target", "expected", "reference")
ID_COLUMNS = ("id", "idx", "index", "example_id", "qid")


def _pick_column(candidates: list[str], preferred: tuple[str, ...]) -> str | None:
    lower_map = {name.lower(): name for name in candidates}
    for col in preferred:
        if col in lower_map:
            return lower_map[col]
    return None


def _normalize_examples(examples: list[PromptExample], max_examples: int | None) -> list[PromptExample]:
    if max_examples is None:
        return examples
    return examples[: max_examples]


def _load_csv(path: Path, start_index: int, max_examples: int | None) -> list[PromptExample]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")

        prompt_col = _pick_column(reader.fieldnames, PROMPT_COLUMNS)
        if prompt_col is None:
            raise ValueError(
                f"CSV at {path} is missing a usable prompt column. "
                f"Expected one of: {', '.join(PROMPT_COLUMNS)}"
            )

        target_col = _pick_column(reader.fieldnames, TARGET_COLUMNS)
        id_col = _pick_column(reader.fieldnames, ID_COLUMNS)

        all_rows: list[PromptExample] = []
        for row_idx, row in enumerate(reader):
            if row_idx < start_index:
                continue

            goal = (row.get(prompt_col) or "").strip()
            if not goal:
                continue

            target = (row.get(target_col) or "").strip() if target_col else None
            example_id = (row.get(id_col) or "").strip() if id_col else str(row_idx)
            all_rows.append(PromptExample(example_id=example_id, goal=goal, target=target or None))

    return _normalize_examples(all_rows, max_examples)


def _json_record_to_example(record: dict, fallback_id: int) -> PromptExample | None:
    goal = ""
    for col in PROMPT_COLUMNS:
        candidate = record.get(col)
        if isinstance(candidate, str) and candidate.strip():
            goal = candidate.strip()
            break
    if not goal:
        return None

    target = None
    for col in TARGET_COLUMNS:
        candidate = record.get(col)
        if isinstance(candidate, str) and candidate.strip():
            target = candidate.strip()
            break

    example_id = str(record.get("id") or record.get("idx") or record.get("example_id") or fallback_id)
    return PromptExample(example_id=example_id, goal=goal, target=target)


def _load_json(path: Path, start_index: int, max_examples: int | None) -> list[PromptExample]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            rows = payload["data"]
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    examples: list[PromptExample] = []
    for idx, row in enumerate(rows):
        if idx < start_index:
            continue
        if not isinstance(row, dict):
            continue
        example = _json_record_to_example(row, idx)
        if example:
            examples.append(example)

    return _normalize_examples(examples, max_examples)


def _load_jsonl(path: Path, start_index: int, max_examples: int | None) -> list[PromptExample]:
    examples: list[PromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            if idx < start_index:
                continue
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            example = _json_record_to_example(row, idx)
            if example:
                examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def load_examples(dataset_path: str | Path, max_examples: int | None = None, start_index: int = 0) -> list[PromptExample]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        examples = _load_csv(path, start_index=start_index, max_examples=max_examples)
    elif suffix == ".json":
        examples = _load_json(path, start_index=start_index, max_examples=max_examples)
    elif suffix == ".jsonl":
        examples = _load_jsonl(path, start_index=start_index, max_examples=max_examples)
    else:
        raise ValueError(f"Unsupported dataset format for {path}. Use CSV, JSON, or JSONL.")

    if not examples:
        raise ValueError(f"No usable examples found in {path}")
    return examples
