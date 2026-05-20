from __future__ import annotations

"""Read a benchmark summary.json and produce a PDF results table."""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


METHOD_PREFIXES = [
    "vanilla",
    "vanilla_short",
    "neural_ranknet",
    "neural_ranknet_short",
    "neural_no_ranknet",
    "neural_no_ranknet_short",
    "contact_implicit",
]


def _label(prefix: str, max_paths: int, max_paths_short: int) -> str:
    return {
        "vanilla": f"GCS ({max_paths} steps)",
        "vanilla_short": f"GCS ({max_paths_short} steps)",
        "neural_ranknet": f"Neural GCS w/ RankNet ({max_paths} paths)",
        "neural_ranknet_short": f"Neural GCS w/ RankNet ({max_paths_short} paths)",
        "neural_no_ranknet": f"Neural GCS w/o RankNet ({max_paths} paths)",
        "neural_no_ranknet_short": f"Neural GCS w/o RankNet ({max_paths_short} paths)",
        "contact_implicit": "Contact-Implicit",
    }.get(prefix, prefix)


def _fmt_mean_std(mean: Optional[float], std: Optional[float], unit: str = "s", scale: float = 1.0) -> str:
    if mean is None:
        return "—"
    m = mean * scale
    s = std * scale if std is not None else 0.0
    suffix = f" {unit}" if unit else ""
    return f"{m:.4f} ± {s:.4f}{suffix}"


def _fmt_success(rate: Optional[float], n: int) -> str:
    if rate is None:
        return "—"
    return f"{int(round(rate * n))}/{n}"


def _fmt_gap(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "N/A"
    s = std if std is not None else 0.0
    return f"{mean * 100:.1f} ± {s * 100:.1f}%"


def build_table(agg: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    n = int(agg.get("num_instances", 0))
    max_paths = int(agg.get("max_paths", 100))
    max_paths_short = int(agg.get("max_paths_short", 10))

    headers = [
        "Method",
        "Success",
        "Paths Used",
        "Relaxation / GNN (s)",
        "Rounding (s)",
        "Total (s)",
        "C_round ↓",
        "Opt. Gap",
    ]

    rows: list[list[str]] = []
    for prefix in METHOD_PREFIXES:
        label = _label(prefix, max_paths, max_paths_short)
        is_ci = prefix == "contact_implicit"

        if is_ci:
            success = _fmt_success(agg.get("contact_implicit_success_rate"), n)
            paths_used = "—"
            relax = "—"
            rounding = "—"
            total = _fmt_mean_std(agg.get("contact_implicit_total_s_mean"), agg.get("contact_implicit_total_s_std"))
            c_round = "N/A"
            gap = "N/A"
        else:
            success = _fmt_success(agg.get(f"{prefix}_success_rate"), n)
            pm = agg.get(f"{prefix}_num_paths_tried_mean")
            ps = agg.get(f"{prefix}_num_paths_tried_std")
            paths_used = f"{pm:.1f} ± {ps:.1f}" if pm is not None else "—"
            is_neural = prefix.startswith("neural")
            if is_neural:
                relax = _fmt_mean_std(
                    agg.get(f"{prefix}_gnn_s_mean"),
                    agg.get(f"{prefix}_gnn_s_std"),
                )
            else:
                relax = _fmt_mean_std(
                    agg.get(f"{prefix}_relaxation_wall_s_mean"),
                    agg.get(f"{prefix}_relaxation_wall_s_std"),
                )
            rounding = _fmt_mean_std(
                agg.get(f"{prefix}_rounding_solver_cumulative_s_mean"),
                agg.get(f"{prefix}_rounding_solver_cumulative_s_std"),
            )
            total = _fmt_mean_std(
                agg.get(f"{prefix}_total_s_mean"),
                agg.get(f"{prefix}_total_s_std"),
            )
            c_round = _fmt_mean_std(
                agg.get(f"{prefix}_cost_mean"),
                agg.get(f"{prefix}_cost_std"),
                unit="",
            )
            if is_neural:
                gap = _fmt_gap(
                    agg.get(f"{prefix}_gap_vs_gcs_relax_mean"),
                    agg.get(f"{prefix}_gap_vs_gcs_relax_std"),
                )
            else:
                gap = _fmt_gap(
                    agg.get(f"{prefix}_relative_gap_upper_bound_mean"),
                    agg.get(f"{prefix}_relative_gap_upper_bound_std"),
                )

        # Skip rows with no data at all
        if success == "—" and total == "—":
            continue

        rows.append([label, success, paths_used, relax, rounding, total, c_round, gap])

    return headers, rows


def render_pdf(headers: list[str], rows: list[list[str]], out_path: Path, title: str = "") -> None:
    n_rows = len(rows)
    n_cols = len(headers)

    col_widths = [3.2, 0.9, 1.2, 1.6, 1.6, 1.6, 1.4, 1.6]
    fig_w = sum(col_widths) + 0.4
    row_h = 0.45
    header_h = 0.55
    fig_h = header_h + n_rows * row_h + (0.6 if title else 0.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    top = 1.0
    if title:
        fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=11, fontweight="bold")
        top = 0.92

    # Normalise column positions
    total_w = sum(col_widths)
    x_starts = [sum(col_widths[:i]) / total_w for i in range(n_cols)]
    x_centers = [(x_starts[i] + col_widths[i] / 2 / total_w) for i in range(n_cols)]

    def _row_y(r: int) -> float:
        # r=0 → header; r>0 → data rows; top is header
        return top - header_h / fig_h - r * (row_h / fig_h)

    HEADER_BG = "#2c3e50"
    ALT_BG = "#ecf0f1"
    WHITE = "#ffffff"
    BORDER = "#7f8c8d"

    # Draw header
    for c, (hdr, xc) in enumerate(zip(headers, x_centers)):
        x0 = x_starts[c]
        w = col_widths[c] / total_w
        rect = mpatches.FancyBboxPatch(
            (x0, top - header_h / fig_h),
            w, header_h / fig_h,
            boxstyle="square,pad=0",
            linewidth=0.5, edgecolor=BORDER,
            facecolor=HEADER_BG,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            xc, top - (header_h / fig_h) / 2,
            hdr, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="white",
        )

    # Draw data rows
    for r, row_data in enumerate(rows):
        bg = ALT_BG if r % 2 == 0 else WHITE
        y_top = top - header_h / fig_h - r * (row_h / fig_h)
        for c, (cell, xc) in enumerate(zip(row_data, x_centers)):
            x0 = x_starts[c]
            w = col_widths[c] / total_w
            rect = mpatches.FancyBboxPatch(
                (x0, y_top - row_h / fig_h),
                w, row_h / fig_h,
                boxstyle="square,pad=0",
                linewidth=0.5, edgecolor=BORDER,
                facecolor=bg,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)
            bold = c == 0
            x_pos = x_starts[c] + 0.01 if c == 0 else xc
            ha = "left" if c == 0 else "center"
            ax.text(
                x_pos, y_top - (row_h / fig_h) / 2,
                cell, transform=ax.transAxes,
                ha=ha, va="center",
                fontsize=7.5,
                fontweight="bold" if bold else "normal",
            )

    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved results table → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=str, help="Path to summary.json from benchmark_planar_pushing.py")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path (default: same dir as summary.json)")
    parser.add_argument("--title", type=str, default="", help="Optional title for the table")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    data = json.loads(summary_path.read_text())
    agg = data["aggregate"]

    headers, rows = build_table(agg)

    out_path = Path(args.output) if args.output else summary_path.parent / "results_table.pdf"
    body = agg.get("body", "")
    title = args.title or f"Planar Pushing Benchmark — {body}" if body else args.title

    render_pdf(headers, rows, out_path, title=title)


if __name__ == "__main__":
    main()
