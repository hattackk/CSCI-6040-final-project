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
    width = 1100
    height = 640
    margin_left = 110
    margin_right = 60
    margin_top = 120
    margin_bottom = 120
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
        svg_text(margin_left, 48, title, size=30, weight=700),
        svg_text(margin_left, 78, subtitle, size=16, fill=COLORS["muted"]),
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
        parts.append(svg_text(margin_left - 12, y + 5, fmt(value), size=14, fill=COLORS["muted"], anchor="end"))

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
        f'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{COLORS["muted"]}" '
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
            parts.append(svg_text(x + bar_w / 2, y - 10, fmt(value), size=13, fill=COLORS["ink"], weight=600, anchor="middle"))

        parts.append(svg_text(group_x + (len(values) * bar_w + (len(values) - 1) * inner_gap) / 2,
                              margin_top + plot_h + 32, model, size=15, weight=600, anchor="middle"))

    # Legend.
    legend_x = width - margin_right - 270
    legend_y = 44
    for idx, label in enumerate(series_labels):
        item_x = legend_x + idx * 86
        parts.append(
            f'<rect x="{item_x:.1f}" y="{legend_y:.1f}" width="18" height="18" rx="4" fill="{series_colors[idx]}"/>'
        )
        parts.append(svg_text(item_x + 26, legend_y + 14, label, size=13, fill=COLORS["ink"]))

    parts.append("</svg>")
    return "\n".join(parts)


def audit_flow_svg() -> str:
    width = 1100
    height = 420
    parts = [
        svg_header(width, height),
        "<title>Qwen audit flow</title>",
        "<desc>Qwen heuristic positives and audit outcome</desc>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        svg_text(70, 54, "Figure 3. Qwen manual audit removed the apparent FITD signal", size=28, weight=700),
        svg_text(70, 84, "The heuristic suggested weak success; manual review showed zero verified jailbreaks.", size=16, fill=COLORS["muted"]),
    ]

    boxes = [
        (80, 160, 250, 120, COLORS["paper_dark"], COLORS["ink"], "3 heuristic positives", "2 FITD + 1 FITD+Vigilant"),
        (425, 160, 250, 120, "#fff4ec", COLORS["fitd"], "3 outputs reviewed", "All flagged outputs inspected manually"),
        (770, 160, 250, 120, "#fff0f2", COLORS["danger"], "0 verified jailbreaks", "All were refusals or safety redirection"),
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

    parts.append(svg_text(550, 348, "Bottom line: Qwen did not produce any verified harmful completion in the audited slice.",
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
        svg_text(70, 84, "The failed reproduction is informative, but exact paper-match conditions were not fully met.", size=16, fill=COLORS["muted"]),
    ]

    box_data = [
        ("Model mismatch", "Qwen, Gemma 4, and local Llama 3 instead of the exact target stack.", COLORS["standard"], 80, 150),
        ("Pipeline mismatch", "Main runs used scaffold FITD, not guaranteed full adaptive FITD.py behavior.", COLORS["fitd"], 560, 150),
        ("Judge mismatch", "Qwen heuristic positives disappeared after manual audit.", COLORS["fitd_vigilant"], 80, 310),
        ("Compute/runtime", "CPU-only HF runs and mixed serving paths kept study small and not perfectly uniform.", COLORS["danger"], 560, 310),
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

    qwen_std = read_summary("results/20260411_qwen25-3b_advbench20_standard/summary.json")
    qwen_fitd = read_summary("results/20260411_qwen25-3b_advbench20_fitd/summary.json")
    qwen_vig = read_summary("results/20260411_qwen25-3b_advbench20_fitd_vigilant/summary.json")
    gemma_std = read_summary("results/20260415_gemma4-e4b_advbench10_standard/summary.json")
    gemma_fitd = read_summary("results/20260415_gemma4-e4b_advbench10_fitd/summary.json")
    gemma_vig = read_summary("results/20260415_gemma4-e4b_advbench10_fitd_vigilant/summary.json")
    llama_std = read_summary("results/20260417_llama3-8b-ollama_advbench10_standard/summary.json")
    llama_fitd = read_summary("results/20260417_llama3-8b-ollama_advbench10_fitd/summary.json")
    llama_vig = read_summary("results/20260417_llama3-8b-ollama_advbench10_fitd_vigilant/summary.json")

    models = ["Qwen 2.5 3B", "Gemma 4 E4B", "Llama 3 8B"]
    series_labels = ["Standard", "FITD", "FITD+V"]
    series_colors = [COLORS["standard"], COLORS["fitd"], COLORS["fitd_vigilant"]]

    asr_values = [
        [qwen_std["asr"], gemma_std["asr"], llama_std["asr"]],
        [qwen_fitd["asr"], gemma_fitd["asr"], llama_fitd["asr"]],
        [qwen_vig["asr"], gemma_vig["asr"], llama_vig["asr"]],
    ]
    refusal_values = [
        [qwen_std["refusal_rate"], gemma_std["refusal_rate"], llama_std["refusal_rate"]],
        [qwen_fitd["refusal_rate"], gemma_fitd["refusal_rate"], llama_fitd["refusal_rate"]],
        [qwen_vig["refusal_rate"], gemma_vig["refusal_rate"], llama_vig["refusal_rate"]],
    ]

    write_all(
        "figure_asr_by_model.svg",
        bar_chart_svg(
            title="Figure 1. Heuristic attack success rate by model and condition",
            subtitle="Only Qwen showed a small heuristic lift, and manual audit later removed it.",
            y_label="Heuristic ASR",
            models=models,
            series_labels=series_labels,
            values=asr_values,
            series_colors=series_colors,
            ymax=0.12,
            fmt=lambda v: f"{v:.2f}",
        ),
    )

    write_all(
        "figure_refusal_rate_by_model.svg",
        bar_chart_svg(
            title="Figure 2. Refusal rate by model and condition",
            subtitle="Gemma 4 and Llama 3 refused every tested prompt across all three settings.",
            y_label="Refusal rate",
            models=models,
            series_labels=series_labels,
            values=refusal_values,
            series_colors=series_colors,
            ymax=1.0,
            fmt=lambda v: f"{v:.2f}",
        ),
    )

    write_all("figure_qwen_audit_flow.svg", audit_flow_svg())
    write_all("figure_reproduction_gaps.svg", blocker_boxes_svg())


if __name__ == "__main__":
    build_figures()
