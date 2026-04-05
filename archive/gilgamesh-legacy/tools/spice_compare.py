#!/usr/bin/env python3
"""Run SPICE + Rust equivalent model comparison and produce plots.

This tool orchestrates the existing neuronSim SPICE generator and the Rust
comparator implemented in this crate. It executes four phases:

1. (Optional) Run the SPICE network generator to produce ngspice outputs.
2. Run the Rust comparator (`cargo run -- compare …`) to compute metrics.
3. Load the JSON report emitted by the comparator.
4. Emit summary plots and a metrics CSV inside the chosen output directory.

Example:
    python tools/spice_compare.py \
        --mode detailed \
        --output-dir comparison_runs/detailed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

METRIC_RE = re.compile(
    r"^(?P<signal>[^:]+): rms=(?P<rms>[-+0-9.eE]+) "
    r"mean_abs=(?P<mean>[-+0-9.eE]+) max_abs=(?P<max>[-+0-9.eE]+) "
    r"@ (?P<time>[-+0-9.eE]+)s \((?P<count>\d+) samples\)$"
)

TIMING_RE = re.compile(
    r"^\[timings\]\s+pre=(?P<pre>\d+)ns\s+sim=(?P<sim>\d+)ns\s+"
    r"analysis=(?P<analysis>\d+)ns\s+serialize=(?P<serialize>\d+)ns$"
)


def resolve_paths() -> tuple[Path, Path]:
    """Return (gilgamesh_root, spice_root) based on this script location."""
    this_file = Path(__file__).resolve()
    gilgamesh_root = this_file.parent.parent
    spice_root = gilgamesh_root.parent / "SPICE" / "neuronSim"
    return gilgamesh_root, spice_root


def run_spice(
    generator: Path,
    network: Path,
    neuron: Path,
    mode: str,
    runs: int = 1,
) -> List[Dict[str, float]]:
    base_cmd = [
        sys.executable,
        str(generator),
        "--network",
        str(network),
        "--neuron",
        str(neuron),
        "--mode",
        mode,
        "--yes",
        "--norun",
    ]
    print(f"[spice] Command: {' '.join(base_cmd)}")
    outputs_dir = generator.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ngspice = shutil.which("ngspice")
    if ngspice is None:
        raise SystemExit("ngspice not found on PATH; cannot benchmark SPICE phase.")

    durations: List[Dict[str, float]] = []
    for idx in range(runs):
        if runs > 1:
            print(f"[spice] Benchmark run {idx + 1}/{runs}")
        start = time.perf_counter()
        subprocess.run(base_cmd, cwd=str(generator.parent), check=True)
        preprocess = time.perf_counter() - start

        cir_path = outputs_dir / f"lif_{mode}.cir"
        log_path = outputs_dir / f"ngspice_{mode}.log"

        if not cir_path.exists():
            raise SystemExit(f"Expected netlist at {cir_path}; generator did not produce it")

        start = time.perf_counter()
        subprocess.run(
            [ngspice, "-b", "-o", str(log_path), str(cir_path)],
            cwd=str(outputs_dir),
            check=True,
        )
        simulate = time.perf_counter() - start

        durations.append({"preprocess": preprocess, "simulate": simulate, "post": 0.0})
        print(f"[spice] preprocess={preprocess:.3f}s simulate={simulate:.3f}s")
    return durations


def rust_binary_path(gilgamesh_root: Path, release: bool) -> Path:
    target_dir = "release" if release else "debug"
    suffix = ".exe" if sys.platform == "win32" else ""
    return gilgamesh_root / "target" / target_dir / f"gilgamesh{suffix}"


def run_rust_comparator(
    gilgamesh_root: Path,
    network: Path,
    neuron: Path,
    spice_csv: Path,
    json_out: Path,
    equivalent_csv: Path,
    release: bool,
) -> Dict:
    binary = rust_binary_path(gilgamesh_root, release)
    if not binary.exists():
        raise SystemExit(
            f"expected compiled binary at {binary}. Run once without --no-prebuild to build it."
        )

    cmd = [
        str(binary),
        "compare",
        "--network",
        str(network),
        "--neuron",
        str(neuron),
        "--spice-csv",
        str(spice_csv),
        "--json-out",
        str(json_out),
        "--equivalent-csv",
        str(equivalent_csv),
    ]
    print(f"[rust] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(gilgamesh_root), check=True)

    with open(json_out, "r", encoding="utf-8") as fh:
        return json.load(fh)


def benchmark_rust_comparator(
    gilgamesh_root: Path,
    network: Path,
    neuron: Path,
    spice_csv: Path,
    json_out: Path,
    equivalent_csv: Path,
    runs: int = 1,
    release: bool = False,
    prebuild: bool = True,
) -> tuple[Dict, List[float]]:
    durations: List[float] = []
    result: Optional[Dict] = None
    if prebuild:
        build_cmd = ["cargo", "build"]
        if release:
            build_cmd.append("--release")
        print(f"[rust] Pre-building comparator: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(gilgamesh_root), check=True)
        binary = rust_binary_path(gilgamesh_root, release)
        if not binary.exists():
            raise SystemExit(f"cargo build succeeded but {binary} was not found")
    for idx in range(runs):
        if runs > 1:
            print(f"[rust] Benchmark run {idx + 1}/{runs}")
        start = time.perf_counter()
        result = run_rust_comparator(
            gilgamesh_root,
            network,
            neuron,
            spice_csv,
            json_out,
            equivalent_csv,
            release,
        )
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"[rust] Completed in {elapsed:.3f}s")
    if result is None:
        raise RuntimeError("rust comparator did not produce a result")
    return result, durations


def parse_metrics_from_stdout(text: str) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        match = METRIC_RE.match(line)
        if not match:
            continue
        metrics.append(
            {
                "signal": match.group("signal"),
                "rms_error": float(match.group("rms")),
                "mean_abs_error": float(match.group("mean")),
                "max_abs_error": float(match.group("max")),
                "max_abs_error_time": float(match.group("time")),
                "sample_count": int(match.group("count")),
            }
        )
    return metrics


def parse_timings_from_stdout(text: str) -> Optional[Dict[str, int]]:
    for raw in text.splitlines():
        line = raw.strip()
        match = TIMING_RE.match(line)
        if match:
            return {
                "rust_preprocess_ns": int(match.group("pre")),
                "rust_simulate_ns": int(match.group("sim")),
                "rust_analysis_ns": int(match.group("analysis")),
                "rust_serialization_ns": int(match.group("serialize")),
            }
    return None


def benchmark_rust_core(
    gilgamesh_root: Path,
    network: Path,
    neuron: Path,
    spice_csv: Path,
    runs: int = 1,
    release: bool = False,
    prebuild: bool = True,
) -> tuple[List[Dict[str, float]], Optional[Dict[str, int]], List[float]]:
    durations: List[float] = []
    captured_metrics: List[Dict[str, float]] = []
    timings: Optional[Dict[str, int]] = None

    binary = rust_binary_path(gilgamesh_root, release)
    if prebuild:
        build_cmd = ["cargo", "build"]
        if release:
            build_cmd.append("--release")
        print(f"[rust] Pre-building comparator: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(gilgamesh_root), check=True)
        binary = rust_binary_path(gilgamesh_root, release)

    if not binary.exists():
        raise SystemExit(f"expected compiled binary at {binary}. Run without --core-only once to build it.")

    for idx in range(runs):
        if runs > 1:
            print(f"[rust] Benchmark run {idx + 1}/{runs}")
        cmd = [
            str(binary),
            "compare",
            "--network",
            str(network),
            "--neuron",
            str(neuron),
            "--spice-csv",
            str(spice_csv),
            "--no-output",
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(gilgamesh_root),
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.perf_counter() - start
        durations.append(elapsed)

        stdout = proc.stdout.strip()
        if stdout:
            print(stdout)
        if proc.stderr and proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)

        if not captured_metrics:
            captured_metrics = parse_metrics_from_stdout(proc.stdout)
        if timings is None:
            timings = parse_timings_from_stdout(proc.stdout)

    return captured_metrics, timings, durations


def write_metrics_csv(metrics: List[Dict], path: Path) -> None:
    if not metrics:
        return
    header = ["signal", "rms_error", "mean_abs_error", "max_abs_error", "max_abs_error_time", "sample_count"]
    lines = [",".join(header)]
    for m in metrics:
        lines.append(
            ",".join(
                [
                    m["signal"],
                    f"{m['rms_error']:.6e}",
                    f"{m['mean_abs_error']:.6e}",
                    f"{m['max_abs_error']:.6e}",
                    f"{m['max_abs_error_time']:.6e}",
                    str(m["sample_count"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_signal(signal: str, samples: List[Dict], outdir: Path) -> None:
    if not samples:
        return
    if plt is None:
        print(f"[plot] matplotlib not installed; skipping plot for {signal}")
        return
    times = [s["time_s"] for s in samples]
    eq_vals = [s["equivalent"] for s in samples]
    spice_vals = [s["spice"] for s in samples]
    deltas = [s["delta"] for s in samples]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top.plot(times, eq_vals, label="Equivalent", linewidth=1.6)
    ax_top.plot(times, spice_vals, label="SPICE", linewidth=1.0, linestyle="--")
    ax_top.set_ylabel("Voltage (V)")
    ax_top.set_title(f"{signal} — Equivalent vs SPICE")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bottom.plot(times, deltas, color="tab:red", linewidth=1.4)
    ax_bottom.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax_bottom.set_ylabel("Delta (V)")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    outfile = outdir / f"{signal}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def summarise_durations(
    durations: List[float], sample_count: Optional[int] = None
) -> Optional[Dict[str, float]]:
    if not durations:
        return None
    mean_s = statistics.fmean(durations)
    summary: Dict[str, float] = {
        "runs": float(len(durations)),
        "total_s": float(sum(durations)),
        "mean_s": float(mean_s),
        "median_s": float(statistics.median(durations)),
        "min_s": float(min(durations)),
        "max_s": float(max(durations)),
    }
    if sample_count and mean_s > 0:
        summary["mean_ms_per_sample"] = float((mean_s / sample_count) * 1e3)
        summary["samples_per_second"] = float(sample_count / mean_s)
        summary["sample_count"] = float(sample_count)
    return summary


def write_benchmark_report(data: Dict, path: Path) -> None:
    if not data:
        return
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    gilgamesh_root, default_spice_root = resolve_paths()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fast", "detailed"], default="detailed")
    parser.add_argument("--network", type=Path, help="Path to network JSON", default=None)
    parser.add_argument("--neuron", type=Path, help="Path to neuron JSON", default=None)
    parser.add_argument("--spice-csv", type=Path, default=None, help="Use an existing SPICE CSV")
    parser.add_argument("--output-dir", type=Path, default=gilgamesh_root / "comparison_outputs")
    parser.add_argument(
        "--spice-root",
        type=Path,
        default=default_spice_root,
        help="Root directory of neuronSim (contains lif_network_generator.py)",
    )
    parser.add_argument("--skip-spice", action="store_true", help="Do not re-run ngspice")
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="How many times to execute each simulator when measuring runtime.",
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--release",
        dest="release",
        action="store_true",
        help="Run the Rust comparator with cargo --release (default).",
    )
    profile_group.add_argument(
        "--debug",
        dest="release",
        action="store_false",
        help="Run the Rust comparator without optimizations (slower).",
    )
    parser.set_defaults(release=True)
    parser.add_argument(
        "--no-prebuild",
        action="store_true",
        help="Skip the initial cargo build step before benchmarking.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Benchmark Rust without writing artifacts (uses --no-output).",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Keep JSON/CSV artifacts even when running multiple benchmark iterations.",
    )

    args = parser.parse_args()

    if args.core_only and args.full_output:
        parser.error("--core-only and --full-output cannot be used together")

    core_mode = args.core_only or (args.benchmark_runs > 1 and not args.full_output)

    spice_root = args.spice_root.resolve()
    generator = spice_root / "lif_network_generator.py"
    if not generator.exists():
        raise SystemExit(f"Could not find lif_network_generator.py at {generator}")

    network = (args.network or (spice_root / "defaults" / "network_default.json")).resolve()
    neuron = (args.neuron or (spice_root / "defaults" / "neuron_default.json")).resolve()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bench_runs = max(1, args.benchmark_runs)
    spice_durations: List[Dict[str, float]] = []

    if not args.skip_spice and args.spice_csv is None:
        spice_durations = run_spice(generator, network, neuron, args.mode, runs=bench_runs)
    elif args.skip_spice:
        print("[spice] Skipping SPICE execution by request; no timing recorded")
    else:
        print("[spice] Using pre-generated CSV; no timing recorded")

    spice_csv = (
        args.spice_csv.resolve()
        if args.spice_csv is not None
        else (spice_root / "outputs" / f"lif_{args.mode}.csv").resolve()
    )
    if not spice_csv.exists():
        raise SystemExit(f"Expected SPICE CSV at {spice_csv}; rerun with --skip-spice disabled?")

    comparison_json = output_dir / f"comparison_{args.mode}.json"
    equivalent_csv = output_dir / f"equivalent_{args.mode}.csv"

    metrics: List[Dict[str, float]] = []
    rust_durations: List[float] = []
    rust_timings: Optional[Dict[str, int]] = None
    result: Optional[Dict] = None

    if core_mode:
        metrics, rust_timings, rust_durations = benchmark_rust_core(
            gilgamesh_root,
            network,
            neuron,
            spice_csv,
            runs=bench_runs,
            release=args.release,
            prebuild=not args.no_prebuild,
        )
        if metrics:
            print("\n=== Metrics (first run) ===")
            for metric in metrics:
                print(
                    f"{metric['signal']}: rms={metric['rms_error']:.6e}, "
                    f"mean_abs={metric['mean_abs_error']:.6e}, max_abs={metric['max_abs_error']:.6e} "
                    f"@ {metric['max_abs_error_time']:.6e}s (n={metric['sample_count']})"
                )
    else:
        result, rust_durations = benchmark_rust_comparator(
            gilgamesh_root,
            network,
            neuron,
            spice_csv,
            comparison_json,
            equivalent_csv,
            runs=bench_runs,
            release=args.release,
            prebuild=not args.no_prebuild,
        )
        metrics = result.get("metrics", [])
        rust_timings = result.get("timings", {})
        print("\n=== Metrics ===")
        for metric in metrics:
            print(
                f"{metric['signal']}: rms={metric['rms_error']:.6e}, "
                f"mean_abs={metric['mean_abs_error']:.6e}, max_abs={metric['max_abs_error']:.6e} "
                f"@ {metric['max_abs_error_time']:.6e}s (n={metric['sample_count']})"
            )
        write_metrics_csv(metrics, output_dir / f"metrics_{args.mode}.csv")

    benchmark: Dict[str, Dict[str, float]] = {}
    sample_count = metrics[0].get("sample_count") if metrics else None
    if sample_count is None and result and result.get("equivalent", {}).get("samples"):
        sample_count = len(result["equivalent"]["samples"])

    spice_pre = summarise_durations([d["preprocess"] for d in spice_durations])
    spice_sim = summarise_durations([d["simulate"] for d in spice_durations])
    if spice_pre or spice_sim:
        benchmark["spice"] = {}
        if spice_pre:
            benchmark["spice"]["preprocess"] = spice_pre
            print(
                f"\n[bench] SPICE preprocess mean {spice_pre['mean_s']:.3f}s over {int(spice_pre['runs'])} run(s)"
            )
        if spice_sim:
            benchmark["spice"]["simulate"] = spice_sim
            print(
                f"[bench] SPICE simulate mean {spice_sim['mean_s']:.3f}s over {int(spice_sim['runs'])} run(s)"
            )

    rust_summary = summarise_durations(rust_durations, sample_count=sample_count)
    if rust_summary:
        benchmark["rust"] = rust_summary
        print(
            f"[bench] Rust comparator mean {rust_summary['mean_s']:.3f}s over {int(rust_summary['runs'])} run(s)"
        )

    if rust_timings:
        def format_ns(value: Optional[int]) -> Optional[str]:
            if value is None:
                return None
            val = float(value)
            if val >= 1e9:
                return f"{val / 1e9:.3f}s"
            if val >= 1e6:
                return f"{val / 1e6:.3f}ms"
            if val >= 1e3:
                return f"{val / 1e3:.3f}µs"
            return f"{val:.0f}ns"

        preprocess = format_ns(rust_timings.get("rust_preprocess_ns"))
        simulate = format_ns(rust_timings.get("rust_simulate_ns"))
        analysis = format_ns(rust_timings.get("rust_analysis_ns"))
        serialization = format_ns(rust_timings.get("rust_serialization_ns"))
        joined = ", ".join(
            f"{label}={value}"
            for label, value in [
                ("pre", preprocess),
                ("sim", simulate),
                ("analysis", analysis),
                ("serialize", serialization),
            ]
            if value is not None
        )
        if joined:
            print(f"[bench] Rust timings: {joined}")
        benchmark["rust_timings"] = rust_timings

    if spice_pre or spice_sim:
        benchmark.setdefault("spice_timings", {})
        if spice_pre:
            mean_pre_ns = int(spice_pre["mean_s"] * 1e9)
            benchmark["spice_timings"]["preprocess_ns"] = mean_pre_ns
            if result is not None:
                result.setdefault("timings", {})["spice_preprocess_ns"] = mean_pre_ns
        if spice_sim:
            mean_sim_ns = int(spice_sim["mean_s"] * 1e9)
            benchmark["spice_timings"]["simulate_ns"] = mean_sim_ns
            if result is not None:
                result.setdefault("timings", {})["spice_simulate_ns"] = mean_sim_ns

        if result is not None:
            comparison_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if "spice" in benchmark:
        mean_spice_sim = benchmark["spice"].get("simulate", {}).get("mean_s")
        mean_rust_sim: Optional[float] = None

        if rust_timings and rust_timings.get("rust_simulate_ns"):
            mean_rust_sim = rust_timings["rust_simulate_ns"] / 1e9
        elif "rust" in benchmark:
            mean_rust_sim = benchmark["rust"].get("mean_s")

        if mean_spice_sim and mean_rust_sim and mean_rust_sim > 0:
            benchmark["rust_speedup_vs_spice_sim"] = mean_spice_sim / mean_rust_sim
            print(
                f"[bench] Rust speedup vs SPICE (simulate): {benchmark['rust_speedup_vs_spice_sim']:.2f}×"
            )

    write_benchmark_report(benchmark, output_dir / f"benchmark_{args.mode}.json")

    if result is not None:
        signal_series: Dict[str, List[Dict]] = result.get("signal_series", {})
        for signal, samples in signal_series.items():
            plot_signal(signal, samples, plots_dir)
        print(f"Outputs written to {output_dir}")
    else:
        print("Core-only mode: skipped writing comparison artifacts.")


if __name__ == "__main__":
    main()
