#!.venv/bin/python
"""Acquire waveform data from a Keysight DSO-X 2012A over USBTMC using PyVISA.

Examples:
  python scope-capture.py --resource "USB0::0x0957::0x1798::MYXXXXXXXX::INSTR" \
      --output waveforms.csv
  python scope-capture.py --resource "USB0::...::INSTR" --include-math
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pyvisa
import tomllib


@dataclass
class Waveform:
    source: str
    time_s: List[float]
    volts: List[float]


def get_sudo_owner() -> tuple[int, int] | None:
    if os.geteuid() != 0:
        return None
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if not uid or not gid:
        return None
    try:
        return int(uid), int(gid)
    except ValueError:
        return None


def ensure_path_owner(path: Path) -> None:
    owner = get_sudo_owner()
    if owner is None or not path.exists():
        return
    uid, gid = owner
    try:
        os.chown(path, uid, gid)
    except PermissionError:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire CH1/CH2/(optional MATH) waveforms from DSO-X 2012A and save CSV.",
        add_help=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )
    parser.add_argument(
        "-C",
        dest="config_file",
        default="scope-config.toml",
        help="Path to TOML config file (default: scope-config.toml)",
    )
    parser.add_argument(
        "--resource",
        help='VISA resource string, e.g. "USB0::0x0957::0x1798::MYXXXXXXXX::INSTR"',
    )
    parser.add_argument(
        "-o",
        "--output",
        help='Output CSV path (default: "output/YYYY-MM-DD-hh-mm-ss.raw.csv")',
    )
    parser.add_argument("-T", dest="timeout_ms", type=int, default=10000, help="VISA timeout in ms")
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Requested waveform points (scope may quantize/limit)",
    )
    parser.add_argument(
        "-m",
        "--include-math",
        action="store_true",
        help="Also acquire MATH trace if enabled on the scope",
    )
    parser.add_argument(
        "-t",
        "--acquire-type",
        choices=["NORM", "AVER", "HRES"],
        default="NORM",
        help="Acquisition mode: NORM (default), AVER (averaging), HRES (high resolution)",
    )
    parser.add_argument(
        "-A",
        dest="average_count",
        type=int,
        default=16,
        help="Average count when --acquire-type AVER is used",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=0.0,
        help="Capture repeatedly for this many seconds (0 = single capture)",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Number of acquisition runs to perform; prompts for Enter between runs",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="High frame-rate mode: CH1/CH2 only, NORM acquisition, lower point count",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, object]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a TOML table: {path}")
    return data


def list_resources(rm: pyvisa.ResourceManager) -> None:
    try:
        resources = rm.list_resources()
    except Exception as exc:
        print(f"Resource discovery failed: {exc}", file=sys.stderr)
        print(
            "If this is a missing backend dependency, install extras in your venv:",
            file=sys.stderr,
        )
        print("  pip install pyusb psutil zeroconf", file=sys.stderr)
        print(
            "You can still run acquisition by supplying --resource explicitly.",
            file=sys.stderr,
        )
        return
    if not resources:
        print("No VISA resources found.")
        return
    print("Detected VISA resources:")
    for res in resources:
        print(f"  {res}")


def query_preamble(inst) -> Dict[str, float]:
    # PREamble? returns:
    # format,type,points,count,xincrement,xorigin,xreference,yincrement,yorigin,yreference
    values = inst.query_ascii_values(":WAVeform:PREamble?")
    if len(values) != 10:
        raise RuntimeError(f"Unexpected preamble length ({len(values)}): {values}")
    keys = [
        "format",
        "type",
        "points",
        "count",
        "xincrement",
        "xorigin",
        "xreference",
        "yincrement",
        "yorigin",
        "yreference",
    ]
    return dict(zip(keys, values))


def acquire_source(inst, source: str) -> Waveform:
    # Key SCPI sequence for each source:
    #   :WAVeform:SOURce CHANnel1|CHANnel2|MATH
    #   :WAVeform:PREamble?
    #   :WAVeform:DATA?
    inst.write(f":WAVeform:SOURce {source}")
    pre = query_preamble(inst)
    raw = inst.query_binary_values(
        ":WAVeform:DATA?",
        datatype="B",
        is_big_endian=True,
        container=list,
    )

    xincrement = float(pre["xincrement"])
    xorigin = float(pre["xorigin"])
    xreference = float(pre["xreference"])
    yincrement = float(pre["yincrement"])
    yorigin = float(pre["yorigin"])
    yreference = float(pre["yreference"])

    n = len(raw)
    time_s = [((i - xreference) * xincrement) + xorigin for i in range(n)]
    volts = [((sample - yreference) * yincrement) + yorigin for sample in raw]
    return Waveform(source=source, time_s=time_s, volts=volts)


def acquire_first_supported_source(inst, source_candidates: List[str]) -> Waveform:
    last_exc: Exception | None = None
    for source in source_candidates:
        try:
            waveform = acquire_source(inst, source)
            waveform.source = source_candidates[0]
            return waveform
        except Exception as exc:  # pragma: no cover - hardware dependent
            last_exc = exc
    if last_exc is None:
        raise RuntimeError("No source candidates were provided.")
    raise last_exc


def write_csv(path: str, waveforms: List[Waveform]) -> None:
    if not waveforms:
        raise RuntimeError("No waveforms to write.")

    # Use the first waveform's time axis as primary; align others by index.
    max_len = max(len(w.volts) for w in waveforms)
    base_time = waveforms[0].time_s
    waveform_map = {w.source: w for w in waveforms}
    sources = [w.source for w in waveforms]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "time_s", *[f"{s}_v" for s in sources]])
        for i in range(max_len):
            row = [i]
            row.append(base_time[i] if i < len(base_time) else "")
            for source in sources:
                values = waveform_map[source].volts
                row.append(values[i] if i < len(values) else "")
            writer.writerow(row)
    out_path = Path(path)
    ensure_path_owner(out_path)
    ensure_path_owner(out_path.parent)


def write_csv_series(path: str, captures: List[Tuple[int, float, List[Waveform]]]) -> None:
    if not captures:
        raise RuntimeError("No captures to write.")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["capture_index", "capture_elapsed_s", "sample_index", "time_s", "source", "voltage_v"]
        )
        for capture_index, elapsed_s, waveforms in captures:
            for waveform in waveforms:
                for sample_index, (t_val, v_val) in enumerate(zip(waveform.time_s, waveform.volts)):
                    writer.writerow(
                        [capture_index, f"{elapsed_s:.6f}", sample_index, t_val, waveform.source, v_val]
                    )
    out_path = Path(path)
    ensure_path_owner(out_path)
    ensure_path_owner(out_path.parent)


def default_output_path() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_path_owner(output_dir)
    return output_dir / f"{ts}.raw.csv"


def output_path_for_run(base_output: str | None, run_index: int, num_runs: int) -> Path:
    if base_output:
        path = Path(base_output)
        if num_runs > 1:
            suffix = path.suffix or ".csv"
            stem = path.stem if path.suffix else path.name
            path = path.with_name(f"{stem}-run{run_index + 1}{suffix}")
        return path
    path = default_output_path()
    if num_runs > 1:
        stem = path.name.removesuffix(".raw.csv")
        path = path.with_name(f"{stem}-run{run_index + 1}.raw.csv")
    return path


def capture_run(inst, sources: List[List[str]], duration_seconds: float, output: Path) -> None:
    if duration_seconds <= 0:
        waveforms = []
        for source_candidates in sources:
            label = source_candidates[0]
            try:
                waveforms.append(acquire_first_supported_source(inst, source_candidates))
                print(f"Captured {label}: {len(waveforms[-1].volts)} samples")
            except Exception as exc:  # Optional sources may be unavailable/disabled.
                print(f"Skipping {label}: {exc}", file=sys.stderr)

        if not waveforms:
            raise RuntimeError("No waveforms were captured.")
        write_csv(str(output), waveforms)
        return

    start = time.monotonic()
    captures: List[Tuple[int, float, List[Waveform]]] = []
    capture_index = 0
    while True:
        elapsed_s = time.monotonic() - start
        if elapsed_s >= duration_seconds:
            break
        waveforms = []
        for source_candidates in sources:
            label = source_candidates[0]
            try:
                waveforms.append(acquire_first_supported_source(inst, source_candidates))
            except Exception as exc:  # Optional sources may be unavailable/disabled.
                print(f"Skipping {label}: {exc}", file=sys.stderr)
        if waveforms:
            captures.append((capture_index, elapsed_s, waveforms))
            capture_index += 1
            print(f"Captured frame {capture_index} at t={elapsed_s:.2f}s")
    if not captures:
        raise RuntimeError("No waveforms were captured during duration window.")
    write_csv_series(str(output), captures)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config_file)

    resource = args.resource or cfg.get("resource") or cfg.get("usb_resource")
    output_arg = args.output or cfg.get("output")
    timeout_ms = args.timeout_ms if args.timeout_ms != 10000 else int(cfg.get("timeout_ms", args.timeout_ms))
    points = args.points if args.points != 1000 else int(cfg.get("points", args.points))
    acquire_type = args.acquire_type if args.acquire_type != "NORM" else str(cfg.get("acquire_type", args.acquire_type))
    average_count = args.average_count if args.average_count != 16 else int(cfg.get("average_count", args.average_count))
    include_math = args.include_math or bool(cfg.get("include_math", False))
    fast_mode = args.fast or bool(cfg.get("fast", False))
    if args.duration != 0.0:
        duration_seconds = args.duration
    else:
        duration_seconds = float(cfg.get("duration_seconds", args.duration))
    if args.num_runs < 1:
        print("Error: --num-runs must be >= 1", file=sys.stderr)
        return 2

    if fast_mode:
        include_math = False
        acquire_type = "NORM"
        if args.points == 1000 and "points" not in cfg:
            points = 200
        print("Fast mode enabled: CH1/CH2 only, NORM type, optimized points for frame rate.")

    rm = pyvisa.ResourceManager()

    if not resource:
        print(
            "Error: no resource set. Pass --resource or set 'resource' in scope-config.toml",
            file=sys.stderr,
        )
        return 2

    sources: List[List[str]] = [["CHAN1"], ["CHAN2"]]
    if include_math:
        sources.append(["MATH"])

    try:
        with rm.open_resource(str(resource)) as inst:
            inst.timeout = timeout_ms
            inst.write_termination = "\n"
            inst.read_termination = "\n"

            idn = inst.query("*IDN?").strip()
            print(f"Connected: {idn}")

            # Setup waveform transfer format.
            inst.write(":WAVeform:FORMat BYTE")
            inst.write(":WAVeform:POINts:MODE RAW")
            inst.write(f":WAVeform:POINts {points}")
            inst.write(f":ACQuire:TYPE {acquire_type}")
            if acquire_type == "AVER":
                inst.write(f":ACQuire:COUNt {average_count}")

            for run_index in range(args.num_runs):
                if args.num_runs > 1:
                    print(f"Starting run {run_index + 1}/{args.num_runs}")
                output = output_path_for_run(output_arg, run_index, args.num_runs)
                output.parent.mkdir(parents=True, exist_ok=True)
                ensure_path_owner(output.parent)
                capture_run(inst, sources, duration_seconds, output)
                print(f"Saved CSV: {output}")
                if run_index < args.num_runs - 1:
                    try:
                        input("Press Enter to continue to the next run...")
                    except EOFError:
                        print("No interactive input available; continuing to next run.")
            return 0
    except Exception as exc:
        print(f"Acquisition failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
