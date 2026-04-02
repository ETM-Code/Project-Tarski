#!.venv/bin/python
"""Integrate power over time from post-processed scope CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate power_w over time_from_start_s to compute energy usage.",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more post-processed CSV files from capture-power-analysis.py",
    )
    parser.add_argument(
        "-o",
        dest="output",
        help='Output CSV path (default: "output/<input>.energy.csv" for one input)',
    )
    if "-h" in sys.argv[1:]:
        parser.error("'-h' is disabled; use --help")
    return parser.parse_args()


def integrate_trapezoid(time_s: list[float], power_w: list[float]) -> float:
    if len(time_s) < 2:
        return 0.0
    energy_j = 0.0
    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            continue
        energy_j += 0.5 * (power_w[i - 1] + power_w[i]) * dt
    return energy_j


def load_groups(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"t": [], "p": []})
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"input_file", "time_from_start_s", "power_w"}
        headers = set(reader.fieldnames or [])
        missing = required - headers
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        for row in reader:
            key = (row.get("input_file") or "").strip() or "<unknown>"
            grouped[key]["t"].append(float(row["time_from_start_s"]))
            grouped[key]["p"].append(float(row["power_w"]))

    out: dict[str, tuple[list[float], list[float]]] = {}
    for key, v in grouped.items():
        pairs = sorted(zip(v["t"], v["p"]), key=lambda x: x[0])
        t = [x for x, _ in pairs]
        p = [y for _, y in pairs]
        out[key] = (t, p)
    return out


def default_output_path(csv_files: list[str]) -> Path:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(csv_files) == 1:
        in_name = Path(csv_files[0]).name
        if in_name.endswith(".proc.csv"):
            stem = in_name[: -len(".proc.csv")]
        elif in_name.endswith(".csv"):
            stem = in_name[: -len(".csv")]
        else:
            stem = Path(in_name).stem
        return out_dir / f"{stem}.energy.csv"
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return out_dir / f"energy-summary-{ts}.energy.csv"


def main() -> int:
    args = parse_args()
    grand_total_j = 0.0
    output_rows: list[list[str]] = []

    for file_arg in args.csv_files:
        path = Path(file_arg)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        print(f"\nFile: {path}")
        groups = load_groups(path)
        for input_file in sorted(groups):
            t, p = groups[input_file]
            energy_j = integrate_trapezoid(t, p)
            energy_wh = energy_j / 3600.0
            energy_mwh = energy_wh * 1000.0
            grand_total_j += energy_j
            span = (max(t) - min(t)) if t else 0.0
            output_rows.append(
                [
                    path.name,
                    input_file,
                    f"{span:.12g}",
                    f"{energy_j:.12g}",
                    f"{energy_wh:.12g}",
                    f"{energy_mwh:.12g}",
                ]
            )
            print(
                f"  {input_file}: duration={span:.6g}s  "
                f"energy={energy_j:.6g}J ({energy_wh:.6g}Wh, {energy_mwh:.6g}mWh)"
            )

    print(
        f"\nTotal energy across all inputs/files: {grand_total_j:.6g}J "
        f"({grand_total_j/3600.0:.6g}Wh, {(grand_total_j/3.6):.6g}mWh)"
    )

    output_path = Path(args.output) if args.output else default_output_path(args.csv_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "input_file", "duration_s", "energy_j", "energy_wh", "energy_mwh"])
        w.writerows(output_rows)
    print(f"Saved energy CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
