#!.venv/bin/python
"""Summarize derived scope CSV files by input_file."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group processed scope CSV rows by input_file and print summary stats.",
        add_help=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more processed CSV files from capture-power-analysis.py",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Print summaries as console tables",
    )
    parser.add_argument(
        "--md",
        action="store_true",
        help="Write summaries to markdown files",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Render summaries to image files (PNG)",
    )
    return parser.parse_args()


def stats(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan
    vmin = min(values)
    vmax = max(values)
    avg = sum(values) / n
    if n < 2:
        stddev = 0.0
    else:
        var = sum((v - avg) ** 2 for v in values) / (n - 1)
        stddev = math.sqrt(var)
    return vmin, vmax, avg, stddev


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.6g}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_line(vals: list[str]) -> str:
        return "| " + " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(render_line(headers))
    print(sep)
    for row in rows:
        print(render_line(row))
    print(sep)


def collect_summaries(path: Path) -> dict[str, list[list[str]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"chan2_voltage_v": [], "current_a": [], "power_w": []}
    )

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"input_file", "chan2_voltage_v", "current_a", "power_w"}
        headers = set(reader.fieldnames or [])
        missing = required - headers
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        for row in reader:
            key = (row.get("input_file") or "").strip()
            if not key:
                key = "<unknown>"
            groups[key]["chan2_voltage_v"].append(float(row["chan2_voltage_v"]))
            groups[key]["current_a"].append(float(row["current_a"]))
            groups[key]["power_w"].append(float(row["power_w"]))

    summaries: dict[str, list[list[str]]] = {}
    for input_file in sorted(groups):
        data = groups[input_file]
        rows: list[list[str]] = []
        for metric in ("chan2_voltage_v", "current_a", "power_w"):
            vmin, vmax, avg, stddev = stats(data[metric])
            rows.append([metric, fmt(vmin), fmt(vmax), fmt(avg), fmt(stddev)])
        summaries[f"{input_file} (n={len(data['power_w'])})"] = rows
    return summaries


def render_console(path: Path, summaries: dict[str, list[list[str]]]) -> None:
    print(f"\nFile: {path}")
    if not summaries:
        print("  No rows.")
        return
    for input_file, rows in summaries.items():
        print(f"  input_file: {input_file}")
        print_table(["metric", "min", "max", "avg", "stddev"], rows)


def render_markdown(path: Path, summaries: dict[str, list[list[str]]]) -> Path:
    md_path = path.with_name(f"{path.stem}-summary.md")
    lines: list[str] = [f"# Summary for `{path.name}`", ""]
    if not summaries:
        lines.extend(["No rows found.", ""])
    for input_file, rows in summaries.items():
        lines.extend([f"## {input_file}", "", "| metric | min | max | avg | stddev |", "|---|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def render_image(path: Path, summaries: dict[str, list[list[str]]]) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image output requires matplotlib. Install it in the active Python environment."
        ) from exc

    img_path = path.with_name(f"{path.stem}-summary.png")
    if not summaries:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No rows found for {path.name}", ha="center", va="center")
        fig.savefig(img_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return img_path

    n = len(summaries)
    fig_h = 2.1 * n + 0.8
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Summary: {path.name}", fontsize=12)

    headers = ["metric", "min", "max", "avg", "stddev"]
    for ax, (input_file, rows) in zip(axes, summaries.items()):
        ax.axis("off")
        ax.set_title(input_file, fontsize=10, pad=8)
        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

    fig.tight_layout()
    fig.savefig(img_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return img_path


def main() -> int:
    args = parse_args()
    if not (args.console or args.md or args.image):
        print("Error: choose at least one output mode: --console, --md, or --image")
        return 2

    for file_arg in args.csv_files:
        path = Path(file_arg)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        summaries = collect_summaries(path)
        if args.console:
            render_console(path, summaries)
        if args.md:
            out_md = render_markdown(path, summaries)
            print(f"Saved markdown: {out_md}")
        if args.image:
            out_img = render_image(path, summaries)
            print(f"Saved image: {out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
