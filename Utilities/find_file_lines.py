#!/usr/bin/env python3
"""List files under a directory with their line counts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def normalize_extension(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


def format_line_count(value: int, human_readable: bool) -> str:
    if not human_readable:
        return str(value)

    units = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    )
    for factor, suffix in units:
        if value >= factor:
            scaled = value / factor
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(value)


def iter_matching_files(
    root: Path, extension: str | None, no_extension: bool, include_hidden_dirs: bool
):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        if not include_hidden_dirs:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if no_extension:
                if path.suffix == "":
                    yield path
                continue
            if extension is None or path.suffix == extension:
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Find files recursively and print their line counts."
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="Show only the top N results.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        default=None,
        help="Only include files with this extension, e.g. .json or json.",
    )
    parser.add_argument(
        "-x",
        "--no-extension",
        action="store_true",
        help="Only include files that do not have an extension.",
    )
    parser.add_argument(
        "-h",
        "--include-hidden-dirs",
        action="store_true",
        help="Include hidden directories (e.g. .git) while scanning.",
    )
    parser.add_argument(
        "-H",
        "--human-readable",
        action="store_true",
        help="Display line counts using suffixes like K, M, and B.",
    )
    parser.add_argument(
        "-C",
        "--csv",
        action="store_true",
        help="Output results as CSV (columns: lines,file).",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be a positive integer")
    if args.extension is not None and args.no_extension:
        parser.error("--extension and --no-extension cannot be used together")

    root = Path(args.root).resolve()
    extension: str | None = None
    if args.extension is not None:
        raw_extension = args.extension.strip()
        if not raw_extension:
            parser.error("--extension cannot be empty")
        extension = normalize_extension(raw_extension)

    files = sorted(
        iter_matching_files(
            root,
            extension,
            args.no_extension,
            args.include_hidden_dirs,
        )
    )

    if not files:
        if args.no_extension:
            print("No files without an extension found.")
        elif extension is not None:
            print(f"No {extension} files found.")
        else:
            print("No files found.")
        return 0

    rows: list[tuple[str, int]] = []
    for path in files:
        rel = str(path.relative_to(root))
        try:
            lines = count_lines(path)
        except OSError:
            continue
        rows.append((rel, lines))

    if not rows:
        print("No readable files found.")
        return 0

    rows.sort(key=lambda item: (-item[1], item[0]))

    shown_rows = rows[: args.count] if args.count is not None else rows
    total_lines = sum(lines for _, lines in shown_rows)
    if args.csv:
        try:
            writer = csv.writer(sys.stdout)
            writer.writerow(["lines", "file"])
            for rel, lines in shown_rows:
                writer.writerow([lines, rel])
            writer.writerow([total_lines, "TOTAL"])
        except BrokenPipeError:
            return 0
        return 0

    formatted_rows = [
        (rel, format_line_count(lines, args.human_readable))
        for rel, lines in shown_rows
    ]
    formatted_total = format_line_count(total_lines, args.human_readable)
    line_width = max(
        len("Lines"),
        max(len(formatted) for _, formatted in formatted_rows),
        len(formatted_total),
    )
    try:
        print(f"{'Lines':>{line_width}}  File")
        print(f"{'-' * line_width}  ----")
        for rel, formatted_lines in formatted_rows:
            print(f"{formatted_lines:>{line_width}}  {rel}")
        print(f"{'-' * line_width}  ----")
        print(f"{formatted_total:>{line_width}}  TOTAL")
    except BrokenPipeError:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
