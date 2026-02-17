from __future__ import annotations

import argparse
import json

from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitd_repro",
        description="Run FITD jailbreak reproducibility experiments.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset file (CSV, JSON, or JSONL).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "openai", "hf"],
        help="Model backend to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-model",
        help="Model identifier for selected backend.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="fitd",
        choices=["standard", "fitd"],
        help="Attack strategy.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=["none", "vigilant"],
        help="Defense condition.",
    )
    parser.add_argument(
        "--fitd-variant",
        type=str,
        default="scaffold",
        choices=["scaffold", "author"],
        help="FITD prompt strategy: scaffold (local baseline) or author (pre-generated author prompts).",
    )
    parser.add_argument(
        "--author-prompt-file",
        type=str,
        default=None,
        help="Optional path to author pre-generated prompt JSON (prompt_jailbreakbench.json / prompt_harmbench.json).",
    )
    parser.add_argument(
        "--author-prompt-track",
        type=str,
        default="prompts1",
        choices=["prompts1", "prompts2"],
        help="Which author prompt sequence to use in author mode.",
    )
    parser.add_argument(
        "--author-max-warmup-turns",
        type=int,
        default=None,
        help=(
            "Optional cap for author warmup turns per example. "
            "Use smaller values (e.g., 3-6) for faster local tests. "
            "Unset or <=0 uses the full author chain."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max number of examples to run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in dataset.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens per model call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between requests to reduce rate-limit pressure.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_experiment(
        dataset_path=args.dataset_path,
        backend=args.backend,
        model_name=args.model,
        attack=args.attack,
        defense=args.defense,
        fitd_variant=args.fitd_variant,
        author_prompt_file=args.author_prompt_file,
        author_prompt_track=args.author_prompt_track,
        author_max_warmup_turns=args.author_max_warmup_turns,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        start_index=args.start_index,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, indent=2))
