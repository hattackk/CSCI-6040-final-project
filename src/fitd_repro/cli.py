from __future__ import annotations

import argparse
import json
import sys

from .runner import run_experiment


_BACKEND_CHOICES = ["mock", "openai", "hf", "vllm", "ollama"]


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
        choices=_BACKEND_CHOICES,
        help="Model backend to use for the target (T).",
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
        "--author-target-mode",
        type=str,
        default="softened",
        choices=["softened", "raw"],
        help=(
            "How to form the final author-mode target turn. "
            "'softened' preserves the project's historical local rewrite; "
            "'raw' preserves the original dataset goal exactly and is the closer paper-faithful option."
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
    parser.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        choices=_BACKEND_CHOICES,
        help="Backend for the JailbreakBench (§A.3.3) judge. Defaults to --backend.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Judge model id. Defaults: gpt-4o for --judge-backend openai "
            "(paper-faithful), --model for vllm/hf/ollama (target reused as judge), "
            "mock-judge for mock."
        ),
    )
    parser.add_argument(
        "--judge-scope",
        type=str,
        default="final",
        choices=["final", "all-turns"],
        help=(
            "Which turns the judge scores. 'final' (default) only scores the final "
            "response; 'all-turns' also scores every warmup turn (~13x more judge "
            "calls on a 12-turn FITD run)."
        ),
    )
    parser.add_argument(
        "--score-harmfulness",
        action="store_true",
        help="Also run the §A.3.6 harmfulness (1-5) judge on scored responses.",
    )
    parser.add_argument(
        "--assistant-backend",
        type=str,
        default=None,
        choices=_BACKEND_CHOICES,
        help="Backend for the FITD assistant model M (§A.3.4/A.3.5). Defaults to --backend.",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default=None,
        help=(
            "Assistant model id. Defaults: gpt-4o-mini for --assistant-backend openai "
            "(paper-faithful), --model for vllm/hf/ollama (target reused as M), "
            "mock-assistant for mock."
        ),
    )
    return parser


def _resolve_role_defaults(
    role: str,
    backend: str | None,
    model: str | None,
    target_backend: str,
    target_model: str,
    openai_model: str,
    mock_model: str,
) -> tuple[str, str | None]:
    """Fill in role defaults for judge / assistant.

    Rule:
      - If --{role}-backend is unset, inherit --backend.
      - If --{role}-model is unset:
          * openai -> paper-faithful default (``openai_model``)
          * mock   -> ``mock_model``
          * vllm/hf/ollama -> fall back to target's ``--model`` and warn on
            stderr (re-using the target as ``role`` is a deviation).
    """
    resolved_backend = (backend or target_backend).lower().strip()
    if model is None:
        if resolved_backend == "openai":
            resolved_model: str | None = openai_model
        elif resolved_backend == "mock":
            resolved_model = mock_model
        else:
            resolved_model = target_model
            print(
                f"[fitd_repro.cli] warning: --{role}-model not set; "
                f"re-using target model {target_model!r} for {role} role "
                f"(deviation from paper). Logged as {role}_deviation in summary.json.",
                file=sys.stderr,
            )
    else:
        resolved_model = model
    return resolved_backend, resolved_model


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    judge_backend, judge_model = _resolve_role_defaults(
        role="judge",
        backend=args.judge_backend,
        model=args.judge_model,
        target_backend=args.backend,
        target_model=args.model,
        openai_model="gpt-4o",
        mock_model="mock-judge",
    )
    assistant_backend, assistant_model = _resolve_role_defaults(
        role="assistant",
        backend=args.assistant_backend,
        model=args.assistant_model,
        target_backend=args.backend,
        target_model=args.model,
        openai_model="gpt-4o-mini",
        mock_model="mock-assistant",
    )

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
        author_target_mode=args.author_target_mode,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        start_index=args.start_index,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
        judge_backend=judge_backend,
        judge_model=judge_model,
        judge_scope=args.judge_scope,
        score_harmfulness=args.score_harmfulness,
        assistant_backend=assistant_backend,
        assistant_model=assistant_model,
    )
    print(json.dumps(summary, indent=2))
