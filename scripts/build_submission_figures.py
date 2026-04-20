#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS_FIG_DIR = ROOT / "docs" / "figures"
SLIDES_FIG_DIR = ROOT / "Final Project Presentation" / "figures"
OUT_DIRS = [DOCS_FIG_DIR, SLIDES_FIG_DIR]

COLORS = {
    "standard": "#7fe7ff",
    "fitd": "#ff9c49",
    "fitd_vigilant": "#5de2a5",
    "danger": "#ff5c6c",
    "ink": "#10141b",
    "ink_soft": "#1d2430",
    "muted": "#7b8a9a",
    "grid": "#d6dde6",
    "paper": "#f8fafc",
    "paper_dark": "#eef3f8",
}

VLLM_MATRIX_MODELS = [
    "Mistral 7B",
    "Llama 3.1 8B",
    "Qwen2 7B",
    "Qwen1.5 7B",
    "Llama 3 8B",
]

VLLM_MATRIX_STANDARD = [0.48, 0.20, 0.12, 0.12, 0.00]
VLLM_MATRIX_FITD = [0.72, 0.24, 0.12, 0.04, 0.04]
VLLM_MATRIX_FITD_VIGILANT = [0.52, 0.28, 0.08, 0.04, 0.00]

VLLM_QWEN_MODELS = ["Qwen2 7B", "Qwen1.5 7B"]
VLLM_QWEN_STANDARD = [0.12, 0.12]
VLLM_QWEN_FITD = [0.12, 0.04]
VLLM_QWEN_FITD_VIGILANT = [0.08, 0.04]


def read_summary(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text())


def ensure_dirs() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)


def write_all(name: str, contents: str) -> None:
    for out_dir in OUT_DIRS:
        (out_dir / name).write_text(contents, encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    )


def svg_text(x: float, y: float, text: str, size: int = 16, fill: str = COLORS["ink"], weight: int = 400,
             anchor: str = "start") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{safe}</text>'
    )


def bar_chart_svg(
    title: str,
    subtitle: str,
    y_label: str,
    models: list[str],
    series_labels: list[str],
    values: list[list[float]],
    series_colors: list[str],
    ymax: float,
    fmt,
) -> str:
    width = 1450
    height = 560
    margin_left = 110
    margin_right = 50
    margin_top = 102
    margin_bottom = 88
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    group_w = plot_w / len(models)
    bar_w = group_w * 0.18
    inner_gap = bar_w * 0.35
    grid_lines = 5

    parts = [
        svg_header(width, height),
        "<title>Figure</title>",
        "<desc>Submission figure</desc>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        svg_text(margin_left, 42, title, size=28, weight=700),
        svg_text(margin_left, 70, subtitle, size=15, fill=COLORS["muted"]),
    ]

    # Grid and y-axis labels.
    for idx in range(grid_lines + 1):
        ratio = idx / grid_lines
        y = margin_top + plot_h - ratio * plot_h
        value = ymax * ratio
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(svg_text(margin_left - 12, y + 5, fmt(value), size=13, fill=COLORS["muted"], anchor="end"))

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" '
        f'stroke="{COLORS["ink"]}" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" '
        f'stroke="{COLORS["ink"]}" stroke-width="2"/>'
    )

    # Y-axis title.
    parts.append(
        f'<text x="36" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 36 {margin_top + plot_h / 2:.1f})" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="15" fill="{COLORS["muted"]}" '
        f'font-weight="600" text-anchor="middle">{y_label}</text>'
    )

    # Bars.
    for model_idx, model in enumerate(models):
        group_x = margin_left + model_idx * group_w + group_w * 0.21
        for series_idx, series_values in enumerate(values):
            value = series_values[model_idx]
            bar_h = 0 if ymax == 0 else (value / ymax) * plot_h
            x = group_x + series_idx * (bar_w + inner_gap)
            y = margin_top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" '
                f'fill="{series_colors[series_idx]}"/>'
            )
            parts.append(svg_text(x + bar_w / 2, y - 10, fmt(value), size=12, fill=COLORS["ink"], weight=600, anchor="middle"))

        parts.append(svg_text(group_x + (len(values) * bar_w + (len(values) - 1) * inner_gap) / 2,
                              margin_top + plot_h + 30, model, size=14, weight=600, anchor="middle"))

    # Legend.
    legend_x = width - margin_right - 255
    legend_y = 28
    for idx, label in enumerate(series_labels):
        item_x = legend_x + idx * 82
        parts.append(
            f'<rect x="{item_x:.1f}" y="{legend_y:.1f}" width="18" height="18" rx="4" fill="{series_colors[idx]}"/>'
        )
        parts.append(svg_text(item_x + 26, legend_y + 14, label, size=12, fill=COLORS["ink"]))

    parts.append("</svg>")
    return "\n".join(parts)


def audit_flow_svg() -> str:
    width = 1100
    height = 420
    parts = [
        svg_header(width, height),
        "<title>Local to vLLM progression</title>",
        "<desc>Why we moved from the original local setup to the closer partner vLLM matrix</desc>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        svg_text(70, 54, "Figure 3. Why we moved beyond the original local all-zero story", size=28, weight=700),
        svg_text(70, 84, "The closer partner system finally produced harmful outputs, but the exact Qwen-family story stayed mixed.", size=16, fill=COLORS["muted"]),
    ]

    boxes = [
        (80, 160, 250, 120, COLORS["paper_dark"], COLORS["ink"], "Local runs looked too clean", "Mostly zero harmful outputs across tested slices"),
        (425, 160, 250, 120, "#fff4ec", COLORS["fitd"], "Partner vLLM runs changed that", "Clear harmful responses appeared on some models"),
        (770, 160, 250, 120, "#fff0f2", COLORS["danger"], "Exact Qwen still needs care", "Some judged Qwen positives were explicit refusals"),
    ]
    for x, y, w, h, fill, stroke, title, subtitle in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
        )
        parts.append(svg_text(x + w / 2, y + 46, title, size=24, weight=700, fill=stroke, anchor="middle"))
        parts.append(svg_text(x + w / 2, y + 78, subtitle, size=15, fill=COLORS["ink"], anchor="middle"))

    arrows = [(330, 220, 425, 220), (675, 220, 770, 220)]
    for x1, y1, x2, y2 in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["muted"]}" stroke-width="4"/>'
        )
        parts.append(
            f'<polygon points="{x2},{y2} {x2 - 18},{y2 - 10} {x2 - 18},{y2 + 10}" fill="{COLORS["muted"]}"/>'
        )

    parts.append(svg_text(550, 348, "Bottom line: the closer system is more believable, but it still does not give us a clean Qwen-family replication.",
                          size=18, fill=COLORS["ink"], weight=600, anchor="middle"))
    parts.append("</svg>")
    return "\n".join(parts)


def blocker_boxes_svg() -> str:
    width = 1100
    height = 520
    parts = [
        svg_header(width, height),
        "<title>Reproducibility blockers</title>",
        "<desc>Major reasons our results may differ from the paper</desc>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        svg_text(70, 54, "Figure 4. Why our reproduction can differ from the paper", size=28, weight=700),
        svg_text(70, 84, "The closer vLLM matrix reduced one gap, but several core paper-faithfulness gaps still remain.", size=16, fill=COLORS["muted"]),
    ]

    box_data = [
        ("Runtime got closer", "The partner GPU used vLLM, which is much closer to the paper than our earlier HF CPU path.", COLORS["standard"], 80, 150),
        ("Prompt path still differs", "The new matrix still used scaffold FITD on AdvBench, not the author raw-target path we wanted for the cleanest Qwen check.", COLORS["fitd"], 560, 150),
        ("Judge still differs", "The partner matrix used a local Qwen 2.5 judge, and some exact-Qwen positives were explicit refusals.", COLORS["fitd_vigilant"], 80, 310),
        ("Adaptive loop still missing", "We still did not reproduce the paper's full adaptive FITD.py pipeline with GPT-4o-mini assistant behavior.", COLORS["danger"], 560, 310),
    ]

    for title, body, color, x, y in box_data:
        parts.append(
            f'<rect x="{x}" y="{y}" width="420" height="120" rx="20" fill="#ffffff" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(svg_text(x + 24, y + 36, title, size=24, weight=700, fill=color))
        parts.append(svg_text(x + 24, y + 68, body, size=16, fill=COLORS["ink"]))

    parts.append("</svg>")
    return "\n".join(parts)


def build_figures() -> None:
    ensure_dirs()
    series_labels = ["Standard", "FITD", "FITD+V"]
    series_colors = [COLORS["standard"], COLORS["fitd"], COLORS["fitd_vigilant"]]

    write_all(
        "figure_asr_by_model.svg",
        bar_chart_svg(
            title="Figure 1. Judge-rescored ASR in the partner vLLM matrix",
            subtitle="The closer GPU/vLLM setup did produce harmful outputs, but the lift was highly model-dependent.",
            y_label="Judge ASR",
            models=VLLM_MATRIX_MODELS,
            series_labels=series_labels,
            values=[
                VLLM_MATRIX_STANDARD,
                VLLM_MATRIX_FITD,
                VLLM_MATRIX_FITD_VIGILANT,
            ],
            series_colors=series_colors,
            ymax=0.80,
            fmt=lambda v: f"{v:.2f}",
        ),
    )

    write_all(
        "figure_refusal_rate_by_model.svg",
        bar_chart_svg(
            title="Figure 2. Exact paper-family Qwen models in the partner vLLM matrix",
            subtitle="The closer runtime produced judged positives, but FITD did not show a strong Qwen-family lift.",
            y_label="Judge ASR",
            models=VLLM_QWEN_MODELS,
            series_labels=series_labels,
            values=[
                VLLM_QWEN_STANDARD,
                VLLM_QWEN_FITD,
                VLLM_QWEN_FITD_VIGILANT,
            ],
            series_colors=series_colors,
            ymax=0.16,
            fmt=lambda v: f"{v:.2f}",
        ),
    )

    write_all("figure_qwen_audit_flow.svg", audit_flow_svg())
    write_all("figure_reproduction_gaps.svg", blocker_boxes_svg())


if __name__ == "__main__":
    build_figures()
