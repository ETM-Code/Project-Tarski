#!.venv/bin/python
"""Plot current and power derived from MATH voltage data in acquisition CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tomllib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute current/power from MATH voltage and plot results.",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument(
        "-C",
        dest="config_file",
        default="analysis-config.toml",
        help="Path to processing config TOML (default: analysis-config.toml)",
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more CSV paths from scope-capture.py",
    )
    parser.add_argument(
        "-r",
        "--resistance",
        type=float,
        help="Resistance in ohms used for Ohm's law (overrides config file)",
    )
    parser.add_argument(
        "-i",
        "--cap-idx",
        dest="capture_index",
        type=int,
        help="For duration/long-format CSV, select one capture index (default: stitch all captures)",
    )
    parser.add_argument(
        "-s",
        "--stitch-mode",
        choices=["mean", "raw"],
        default="mean",
        help="For stitched long-format data: mean per capture (default) or all raw points",
    )
    parser.add_argument(
        "-d",
        "--derive-diff",
        dest="manual_diff",
        action="store_true",
        help="Use CH1-CH2 for current calculation instead of MATH channel",
    )
    parser.add_argument(
        "-c",
        "--column",
        nargs=2,
        action="append",
        metavar=("TARGET", "NAME"),
        help=(
            "Override source column names by target: "
            "-c math MATH, -c ch1 CHAN1, -c ch2 CHAN2 (TARGET: ch1|ch2|math)"
        ),
    )
    parser.add_argument(
        "-o",
        dest="export_csv",
        help='Output path for derived CSV (default: "output/<input>.proc.csv" for one input)',
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Display interactive graphs (default: no graph display)",
    )
    if "-h" in sys.argv[1:]:
        parser.error("'-h' is disabled; use --help")
    return parser.parse_args()


def load_config(path: str) -> dict[str, object]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def read_math_wide_csv(reader: csv.DictReader, math_column: str) -> tuple[list[float], list[float]]:
    headers = reader.fieldnames or []
    candidates = [math_column, f"{math_column}_v", "MATH", "MATH_v"]
    selected = None
    for name in candidates:
        if name in headers:
            selected = name
            break
    if selected is None:
        raise ValueError(f"MATH column not found. Available columns: {headers}")

    time_s: list[float] = []
    volts: list[float] = []
    for row in reader:
        t_raw = (row.get("time_s") or "").strip()
        v_raw = (row.get(selected) or "").strip()
        if not t_raw or not v_raw:
            continue
        time_s.append(float(t_raw))
        volts.append(float(v_raw))
    return time_s, volts


def resolve_column(headers: list[str], base_name: str) -> str | None:
    candidates = [base_name, f"{base_name}_v"]
    for name in candidates:
        if name in headers:
            return name
    return None


def read_diff_wide_csv(
    reader: csv.DictReader, ch1_column: str, ch2_column: str
) -> tuple[list[float], list[float]]:
    headers = reader.fieldnames or []
    ch1_name = resolve_column(headers, ch1_column) or resolve_column(headers, "CHAN1")
    ch2_name = resolve_column(headers, ch2_column) or resolve_column(headers, "CHAN2")
    if ch1_name is None or ch2_name is None:
        raise ValueError(f"CH1/CH2 columns not found. Available columns: {headers}")

    time_s: list[float] = []
    diff_v: list[float] = []
    for row in reader:
        t_raw = (row.get("time_s") or "").strip()
        v1_raw = (row.get(ch1_name) or "").strip()
        v2_raw = (row.get(ch2_name) or "").strip()
        if not t_raw or not v1_raw or not v2_raw:
            continue
        v1 = float(v1_raw)
        v2 = float(v2_raw)
        time_s.append(float(t_raw))
        diff_v.append(v1 - v2)
    return time_s, diff_v


def read_current_and_power_wide(
    reader: csv.DictReader,
    manual_diff: bool,
    math_column: str,
    ch1_column: str,
    ch2_column: str,
) -> tuple[list[float], list[float], list[float]]:
    headers = reader.fieldnames or []
    ch2_name = resolve_column(headers, ch2_column) or resolve_column(headers, "CHAN2")
    if ch2_name is None:
        raise ValueError(f"CH2 column not found. Available columns: {headers}")

    if manual_diff:
        ch1_name = resolve_column(headers, ch1_column) or resolve_column(headers, "CHAN1")
        if ch1_name is None:
            raise ValueError(f"CH1 column not found. Available columns: {headers}")
        math_name = None
    else:
        math_name = resolve_column(headers, math_column) or resolve_column(headers, "MATH")
        if math_name is None:
            raise ValueError(f"MATH column not found. Available columns: {headers}")
        ch1_name = None

    time_s: list[float] = []
    current_v: list[float] = []
    power_v: list[float] = []
    for row in reader:
        t_raw = (row.get("time_s") or "").strip()
        v2_raw = (row.get(ch2_name) or "").strip()
        if not t_raw or not v2_raw:
            continue
        if manual_diff:
            v1_raw = (row.get(ch1_name) or "").strip() if ch1_name else ""
            if not v1_raw:
                continue
            cur_v = float(v1_raw) - float(v2_raw)
        else:
            vm_raw = (row.get(math_name) or "").strip() if math_name else ""
            if not vm_raw:
                continue
            cur_v = float(vm_raw)
        time_s.append(float(t_raw))
        current_v.append(cur_v)
        power_v.append(float(v2_raw))
    return time_s, current_v, power_v


def read_math_long_csv(
    reader: csv.DictReader, math_column: str, capture_index: int
) -> tuple[list[float], list[float]]:
    rows = [row for row in reader]
    captures = sorted(
        {int((row.get("capture_index") or "").strip()) for row in rows if (row.get("capture_index") or "").strip()}
    )
    if not captures:
        raise ValueError("No capture_index values in long-format CSV.")
    selected_capture = captures[-1] if capture_index < 0 else capture_index
    if selected_capture not in captures:
        raise ValueError(f"Capture index {selected_capture} not found. Available: {captures}")

    candidates = {math_column, f"{math_column}_v", "MATH", "MATH_v"}
    samples: list[tuple[int, float, float]] = []
    for row in rows:
        ci_raw = (row.get("capture_index") or "").strip()
        source = (row.get("source") or "").strip()
        if not ci_raw or int(ci_raw) != selected_capture or source not in candidates:
            continue
        si_raw = (row.get("sample_index") or "").strip()
        t_raw = (row.get("time_s") or "").strip()
        v_raw = (row.get("voltage_v") or "").strip()
        if not si_raw or not t_raw or not v_raw:
            continue
        samples.append((int(si_raw), float(t_raw), float(v_raw)))

    if not samples:
        raise ValueError(f"No MATH samples found for capture_index={selected_capture}.")

    samples.sort(key=lambda x: x[0])
    time_s = [t for _, t, _ in samples]
    volts = [v for _, _, v in samples]
    return time_s, volts


def read_diff_long_csv(
    reader: csv.DictReader, ch1_column: str, ch2_column: str, capture_index: int
) -> tuple[list[float], list[float]]:
    rows = [row for row in reader]
    captures = sorted(
        {int((row.get("capture_index") or "").strip()) for row in rows if (row.get("capture_index") or "").strip()}
    )
    if not captures:
        raise ValueError("No capture_index values in long-format CSV.")
    selected_capture = captures[-1] if capture_index < 0 else capture_index
    if selected_capture not in captures:
        raise ValueError(f"Capture index {selected_capture} not found. Available: {captures}")

    ch1_candidates = {ch1_column, f"{ch1_column}_v", "CHAN1", "CHAN1_v"}
    ch2_candidates = {ch2_column, f"{ch2_column}_v", "CHAN2", "CHAN2_v"}
    ch1_by_sample: dict[int, tuple[float, float]] = {}
    ch2_by_sample: dict[int, tuple[float, float]] = {}

    for row in rows:
        ci_raw = (row.get("capture_index") or "").strip()
        source = (row.get("source") or "").strip()
        if not ci_raw or int(ci_raw) != selected_capture:
            continue
        si_raw = (row.get("sample_index") or "").strip()
        t_raw = (row.get("time_s") or "").strip()
        v_raw = (row.get("voltage_v") or "").strip()
        if not si_raw or not t_raw or not v_raw:
            continue
        si = int(si_raw)
        sample = (float(t_raw), float(v_raw))
        if source in ch1_candidates:
            ch1_by_sample[si] = sample
        elif source in ch2_candidates:
            ch2_by_sample[si] = sample

    common_idx = sorted(set(ch1_by_sample.keys()) & set(ch2_by_sample.keys()))
    if not common_idx:
        raise ValueError(f"No overlapping CH1/CH2 samples found for capture_index={selected_capture}.")

    time_s: list[float] = []
    diff_v: list[float] = []
    for si in common_idx:
        t1, v1 = ch1_by_sample[si]
        _, v2 = ch2_by_sample[si]
        time_s.append(t1)
        diff_v.append(v1 - v2)
    return time_s, diff_v


def read_series_long(
    rows: list[dict[str, str]],
    capture_index: int,
    candidates: set[str],
) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for row in rows:
        ci_raw = (row.get("capture_index") or "").strip()
        source = (row.get("source") or "").strip()
        if not ci_raw or int(ci_raw) != capture_index or source not in candidates:
            continue
        si_raw = (row.get("sample_index") or "").strip()
        t_raw = (row.get("time_s") or "").strip()
        v_raw = (row.get("voltage_v") or "").strip()
        if not si_raw or not t_raw or not v_raw:
            continue
        out[int(si_raw)] = (float(t_raw), float(v_raw))
    return out


def read_current_and_power_long(
    rows: list[dict[str, str]],
    capture_index: int,
    manual_diff: bool,
    math_column: str,
    ch1_column: str,
    ch2_column: str,
) -> tuple[list[float], list[float], list[float]]:
    ch2_candidates = {ch2_column, f"{ch2_column}_v", "CHAN2", "CHAN2_v"}
    ch2_map = read_series_long(rows, capture_index, ch2_candidates)
    if not ch2_map:
        raise ValueError(f"No CH2 samples found for capture_index={capture_index}.")

    if manual_diff:
        ch1_candidates = {ch1_column, f"{ch1_column}_v", "CHAN1", "CHAN1_v"}
        ch1_map = read_series_long(rows, capture_index, ch1_candidates)
        common_idx = sorted(set(ch1_map.keys()) & set(ch2_map.keys()))
        if not common_idx:
            raise ValueError(f"No overlapping CH1/CH2 samples found for capture_index={capture_index}.")
        time_s = [ch2_map[i][0] for i in common_idx]
        current_v = [ch1_map[i][1] - ch2_map[i][1] for i in common_idx]
        power_v = [ch2_map[i][1] for i in common_idx]
        return time_s, current_v, power_v

    math_candidates = {math_column, f"{math_column}_v", "MATH", "MATH_v"}
    math_map = read_series_long(rows, capture_index, math_candidates)
    common_idx = sorted(set(math_map.keys()) & set(ch2_map.keys()))
    if not common_idx:
        raise ValueError(f"No overlapping MATH/CH2 samples found for capture_index={capture_index}.")
    time_s = [ch2_map[i][0] for i in common_idx]
    current_v = [math_map[i][1] for i in common_idx]
    power_v = [ch2_map[i][1] for i in common_idx]
    return time_s, current_v, power_v


def read_current_and_power_long_stitched(
    rows: list[dict[str, str]],
    manual_diff: bool,
    math_column: str,
    ch1_column: str,
    ch2_column: str,
    stitch_mode: str,
) -> tuple[list[float], list[float], list[float]]:
    captures = sorted(
        {
            int((row.get("capture_index") or "").strip())
            for row in rows
            if (row.get("capture_index") or "").strip()
        }
    )
    if not captures:
        raise ValueError("No capture_index values in long-format CSV.")

    elapsed_by_capture: dict[int, float] = {}
    for row in rows:
        ci_raw = (row.get("capture_index") or "").strip()
        ce_raw = (row.get("capture_elapsed_s") or "").strip()
        if not ci_raw or not ce_raw:
            continue
        ci = int(ci_raw)
        if ci not in elapsed_by_capture:
            elapsed_by_capture[ci] = float(ce_raw)

    out_t: list[float] = []
    out_current_v: list[float] = []
    out_power_v: list[float] = []
    for ci in captures:
        time_s, current_v, power_v = read_current_and_power_long(
            rows, ci, manual_diff, math_column, ch1_column, ch2_column
        )
        if not time_s:
            continue
        base_t = elapsed_by_capture.get(ci, 0.0)
        if stitch_mode == "mean":
            out_t.append(base_t)
            out_current_v.append(sum(current_v) / len(current_v))
            out_power_v.append(sum(power_v) / len(power_v))
        else:
            local_t0 = min(time_s)
            stitched_t = [base_t + (t - local_t0) for t in time_s]
            out_t.extend(stitched_t)
            out_current_v.extend(current_v)
            out_power_v.extend(power_v)

    if not out_t:
        raise ValueError("No valid stitched samples in long-format CSV.")
    return out_t, out_current_v, out_power_v


def read_current_and_power_series(
    path: Path,
    math_column: str,
    capture_index: int | None,
    manual_diff: bool,
    ch1_column: str,
    ch2_column: str,
    stitch_mode: str,
) -> tuple[list[float], list[float], list[float]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        if "time_s" not in headers:
            raise ValueError("CSV must contain 'time_s'.")
        is_long = {"capture_index", "source", "voltage_v"}.issubset(headers)
        if is_long:
            rows = [row for row in reader]
            captures = sorted(
                {
                    int((row.get("capture_index") or "").strip())
                    for row in rows
                    if (row.get("capture_index") or "").strip()
                }
            )
            if not captures:
                raise ValueError("No capture_index values in long-format CSV.")
            if capture_index is None:
                return read_current_and_power_long_stitched(
                    rows, manual_diff, math_column, ch1_column, ch2_column, stitch_mode
                )
            if capture_index not in captures:
                raise ValueError(f"Capture index {capture_index} not found. Available: {captures}")
            return read_current_and_power_long(
                rows, capture_index, manual_diff, math_column, ch1_column, ch2_column
            )
        return read_current_and_power_wide(reader, manual_diff, math_column, ch1_column, ch2_column)


def write_derived_csv(
    path: str,
    rows: list[tuple[str, int, float, float, float, float]],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input_file",
                "sample_index",
                "time_from_start_s",
                "chan2_voltage_v",
                "current_a",
                "power_w",
            ]
        )
        writer.writerows(rows)


def default_export_csv_path(csv_files: list[str]) -> str:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(csv_files) == 1:
        in_name = Path(csv_files[0]).name
        if in_name.endswith(".raw.csv"):
            stem = in_name[: -len(".raw.csv")]
        elif in_name.endswith(".csv"):
            stem = in_name[: -len(".csv")]
        else:
            stem = Path(in_name).stem
        return str(out_dir / f"{stem}.proc.csv")
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return str(out_dir / f"processed-scope-data-{ts}.proc.csv")


def resolve_column_overrides(
    column_args: list[list[str]] | None,
) -> tuple[str, str, str]:
    columns = {"math": "MATH", "ch1": "CHAN1", "ch2": "CHAN2"}
    if not column_args:
        return columns["math"], columns["ch1"], columns["ch2"]

    for target_raw, name_raw in column_args:
        target = target_raw.strip().lower()
        name = name_raw.strip()
        if target not in columns:
            raise ValueError("Invalid -c/--column target. Use ch1, ch2, or math.")
        if not name:
            raise ValueError("Column name cannot be empty.")
        columns[target] = name
    return columns["math"], columns["ch1"], columns["ch2"]


def main() -> int:
    args = parse_args()
    math_column, ch1_column, ch2_column = resolve_column_overrides(args.column)
    cfg = load_config(args.config_file)
    resistance = args.resistance if args.resistance is not None else float(cfg.get("resistance", 1.0))

    if resistance <= 0:
        raise ValueError("Resistance must be > 0 ohms.")

    render_plot = args.graph
    fig = None
    axes = None
    if render_plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plotted_any = False
    export_rows: list[tuple[str, int, float, float, float, float]] = []
    for file_arg in args.csv_files:
        csv_path = Path(file_arg)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        time_s, current_voltage_v, power_voltage_v = read_current_and_power_series(
            csv_path,
            math_column,
            args.capture_index,
            args.manual_diff,
            ch1_column,
            ch2_column,
            args.stitch_mode,
        )
        if not time_s:
            continue

        t0 = min(time_s)
        time_from_start = [t - t0 for t in time_s]
        current_a = [v / resistance for v in current_voltage_v]
        power_w = [v * i for v, i in zip(power_voltage_v, current_a)]
        label = csv_path.stem

        if render_plot and axes is not None:
            axes[0].plot(time_from_start, current_a, linewidth=1.0, label=label)
            axes[1].plot(time_from_start, power_w, linewidth=1.0, label=label)
        for idx, (t, v2, i, p) in enumerate(zip(time_from_start, power_voltage_v, current_a, power_w)):
            export_rows.append((csv_path.name, idx, t, v2, i, p))
        plotted_any = True

    if not plotted_any:
        raise ValueError("No valid current/power input samples found in provided CSV files.")

    if not args.graph:
        export_path = args.export_csv or default_export_csv_path(args.csv_files)
        write_derived_csv(export_path, export_rows)
        print(f"Saved derived CSV: {export_path}")

    if render_plot and axes is not None and fig is not None:
        axes[0].set_ylabel("Current (A)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_ylabel("Power (W)")
        axes[1].set_xlabel("Time from start (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        current_src = "CH1-CH2" if args.manual_diff else "MATH"
        fig.suptitle(
            f"Current from {current_src} via Ohm's law, Power = CHAN2 * I (R={resistance} ohm)"
        )
        fig.tight_layout()
        if args.graph:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
