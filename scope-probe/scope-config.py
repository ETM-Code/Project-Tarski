#!.venv/bin/python
"""Configure basic DSO-X 2012A scale settings over SCPI/VISA."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pyvisa
import tomllib


_UNITED_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµ]+)\s*$"
)


def _parse_with_units(raw: str, unit_scale: dict[str, float], kind: str) -> float:
    m = _UNITED_NUMBER_RE.match(raw)
    if not m:
        example = "200mV" if kind == "voltage" else "100us or 0.1s"
        raise argparse.ArgumentTypeError(
            f"{kind} value must include units (for example: {example})"
        )
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit not in unit_scale:
        allowed = ", ".join(sorted(unit_scale.keys()))
        raise argparse.ArgumentTypeError(f"Invalid {kind} unit '{m.group(2)}'. Allowed: {allowed}")
    return value * unit_scale[unit]


def parse_time_value(raw: str) -> float:
    return _parse_with_units(
        raw,
        {
            "s": 1.0,
            "ms": 1e-3,
            "us": 1e-6,
            "µs": 1e-6,
            "ns": 1e-9,
        },
        "time",
    )


def parse_voltage_value(raw: str) -> float:
    return _parse_with_units(
        raw,
        {
            "v": 1.0,
            "mv": 1e-3,
            "uv": 1e-6,
            "µv": 1e-6,
        },
        "voltage",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set horizontal/vertical scale parameters on a Keysight DSO-X 2012A."
    )
    parser.add_argument(
        "-C",
        dest="config_file",
        default="scope-config.toml",
        help="TOML config file used to read default VISA resource",
    )
    parser.add_argument(
        "--resource",
        help='VISA resource, e.g. "USB0::0x0957::0x1798::MYXXXXXXXX::INSTR"',
    )
    parser.add_argument(
        "--set-resource",
        help='Persist resource to config file and exit, e.g. "USB0::...::INSTR"',
    )
    parser.add_argument(
        "--set-duration",
        type=parse_time_value,
        help="Persist default capture duration_seconds to config file and exit (e.g. 30s, 100ms)",
    )
    parser.add_argument(
        "-T",
        dest="timeout_ms",
        type=int,
        default=10000,
        help="VISA timeout in milliseconds",
    )
    parser.add_argument(
        "--time-scale",
        type=parse_time_value,
        default=1e-5,
        help="Horizontal scale with units (e.g. 10us, 0.1ms, 1s)",
    )
    parser.add_argument(
        "--time-position",
        type=parse_time_value,
        help="Optional horizontal position with units (e.g. 0s, 500us)",
    )
    parser.add_argument(
        "--ch-scale",
        nargs=2,
        action="append",
        metavar=("TARGET", "VALUE"),
        help=(
            "Set vertical scale with units by target channel: "
            "--ch-scale 1 500mV, --ch-scale 2 1V, --ch-scale all 500mV"
        ),
    )
    parser.add_argument(
        "--ch-offset",
        nargs=2,
        action="append",
        metavar=("TARGET", "VALUE"),
        help=(
            "Set vertical offset with units by target channel: "
            "--ch-offset 1 200mV, --ch-offset 2 -100mV, --ch-offset all 0V"
        ),
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search/list VISA resources and exit",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Read and print current scope settings without applying changes",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store current scope settings into the TOML config file and exit",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load saved scope settings from TOML config file and apply them",
    )
    parser.add_argument(
        "--math",
        choices=["off", "sub", "subtract"],
        help="Math mode: sub/subtract configures CH1-CH2, off disables math display",
    )
    return parser.parse_args()


def load_resource_from_config(path: str) -> str | None:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return None
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("resource")


def load_config(path: str) -> dict[str, object]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


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


def save_config(path: str, cfg: dict[str, object]) -> None:
    preferred_order = [
        "resource",
        "duration_seconds",
        "time_scale",
        "time_position",
        "ch1_scale",
        "ch2_scale",
        "ch1_offset",
        "ch2_offset",
        "ch1_display",
        "ch2_display",
    ]
    keys = [k for k in preferred_order if k in cfg] + [k for k in cfg if k not in preferred_order]
    lines = ["# Scope configuration defaults"]
    for key in keys:
        lines.append(f"{key} = {format_toml_value(cfg[key])}")
    cfg_path = Path(path)
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ensure_path_owner(cfg_path)


def list_resources(rm: pyvisa.ResourceManager) -> None:
    try:
        resources = rm.list_resources()
    except Exception as exc:
        print(f"Resource discovery failed: {exc}")
        print("If needed, run with sudo or fix USB/VISA permissions and try again.")
        return
    if not resources:
        print("No VISA resources found.")
        return
    print("Detected VISA resources:")
    for res in resources:
        print(f"  {res}")


def format_seconds(value: float) -> str:
    a = abs(value)
    if a >= 1:
        return f"{value:.6g} s"
    if a >= 1e-3:
        return f"{value * 1e3:.6g} ms"
    if a >= 1e-6:
        return f"{value * 1e6:.6g} us"
    if a >= 1e-9:
        return f"{value * 1e9:.6g} ns"
    return f"{value:.6g} s"


def format_volts(value: float) -> str:
    a = abs(value)
    if a >= 1:
        return f"{value:.6g} V"
    if a >= 1e-3:
        return f"{value * 1e3:.6g} mV"
    if a >= 1e-6:
        return f"{value * 1e6:.6g} uV"
    return f"{value:.6g} V"


def read_scope_settings(inst) -> dict[str, float | int]:
    return {
        "time_scale": float(inst.query(":TIMebase:SCALe?").strip()),
        "time_position": float(inst.query(":TIMebase:POSition?").strip()),
        "ch1_scale": float(inst.query(":CHANnel1:SCALe?").strip()),
        "ch2_scale": float(inst.query(":CHANnel2:SCALe?").strip()),
        "ch1_offset": float(inst.query(":CHANnel1:OFFSet?").strip()),
        "ch2_offset": float(inst.query(":CHANnel2:OFFSet?").strip()),
        "ch1_display": int(float(inst.query(":CHANnel1:DISPlay?").strip())),
        "ch2_display": int(float(inst.query(":CHANnel2:DISPlay?").strip())),
    }


def print_scope_settings(title: str, s: dict[str, float | int]) -> None:
    print(title)
    print(f"  TIME scale:         {format_seconds(float(s['time_scale']))}/div")
    print(f"  TIME position:      {format_seconds(float(s['time_position']))}")
    print(f"  CH1 scale:          {format_volts(float(s['ch1_scale']))}/div")
    print(f"  CH2 scale:          {format_volts(float(s['ch2_scale']))}/div")
    print(f"  CH1 offset:         {format_volts(float(s['ch1_offset']))}")
    print(f"  CH2 offset:         {format_volts(float(s['ch2_offset']))}")
    print(f"  CH1 display:        {'ON' if int(s['ch1_display']) else 'OFF'}")
    print(f"  CH2 display:        {'ON' if int(s['ch2_display']) else 'OFF'}")


def configure_math_ch1_minus_ch2(inst) -> None:
    # Keysight DSO-X math setup for CH1 - CH2
    inst.write(":MATH:OPERator SUBTract")
    inst.write(":MATH:SOURce1 CHANnel1")
    inst.write(":MATH:SOURce2 CHANnel2")
    inst.write(":MATH:DISPlay ON")


def configure_math_off(inst) -> None:
    inst.write(":MATH:DISPlay OFF")


def resolve_channel_scales(ch_scale_args: list[list[str]] | None) -> tuple[float, float]:
    # Keep previous defaults when no channel scale arguments are provided.
    ch1_scale = 0.5
    ch2_scale = 0.5
    if not ch_scale_args:
        return ch1_scale, ch2_scale

    for target_raw, value_raw in ch_scale_args:
        target = target_raw.strip().lower()
        try:
            value = parse_voltage_value(value_raw)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"Invalid --ch-scale value '{value_raw}': {exc}") from exc

        if target == "1":
            ch1_scale = value
        elif target == "2":
            ch2_scale = value
        elif target == "all":
            ch1_scale = value
            ch2_scale = value
        else:
            raise ValueError("Invalid --ch-scale target. Use 1, 2, or all.")

    return ch1_scale, ch2_scale


def resolve_channel_offsets(ch_offset_args: list[list[str]] | None) -> tuple[float | None, float | None]:
    ch1_offset: float | None = None
    ch2_offset: float | None = None
    if not ch_offset_args:
        return ch1_offset, ch2_offset

    for target_raw, value_raw in ch_offset_args:
        target = target_raw.strip().lower()
        try:
            value = parse_voltage_value(value_raw)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"Invalid --ch-offset value '{value_raw}': {exc}") from exc

        if target == "1":
            ch1_offset = value
        elif target == "2":
            ch2_offset = value
        elif target == "all":
            ch1_offset = value
            ch2_offset = value
        else:
            raise ValueError("Invalid --ch-offset target. Use 1, 2, or all.")

    return ch1_offset, ch2_offset


def main() -> int:
    args = parse_args()
    try:
        resolved_ch1_scale, resolved_ch2_scale = resolve_channel_scales(args.ch_scale)
        resolved_ch1_offset, resolved_ch2_offset = resolve_channel_offsets(args.ch_offset)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    if args.set_resource is not None or args.set_duration is not None:
        cfg = load_config(args.config_file)
        if args.set_resource is not None:
            cfg["resource"] = args.set_resource
        if args.set_duration is not None:
            cfg["duration_seconds"] = float(args.set_duration)
        save_config(args.config_file, cfg)
        print(f"Updated config: {args.config_file}")
        if args.set_resource is not None:
            print(f"  resource = {args.set_resource}")
        if args.set_duration is not None:
            print(f"  duration_seconds = {float(args.set_duration)}")
        return 0

    try:
        rm = pyvisa.ResourceManager()
    except Exception as exc:
        print(f"Failed to initialize VISA resource manager: {exc}")
        return 1

    if args.search:
        list_resources(rm)
        return 0

    resource = args.resource or load_resource_from_config(args.config_file)
    if not resource:
        print("Error: no VISA resource provided. Use --resource or set 'resource' in config.")
        return 2

    try:
        with rm.open_resource(resource) as inst:
            inst.timeout = args.timeout_ms
            inst.write_termination = "\n"
            inst.read_termination = "\n"

            idn = inst.query("*IDN?").strip()
            print(f"Connected: {idn}")
            current = read_scope_settings(inst)
            print_scope_settings("Current settings:", current)

            if args.store:
                cfg = load_config(args.config_file)
                cfg.update(
                    {
                        "time_scale": float(current["time_scale"]),
                        "time_position": float(current["time_position"]),
                        "ch1_scale": float(current["ch1_scale"]),
                        "ch2_scale": float(current["ch2_scale"]),
                        "ch1_offset": float(current["ch1_offset"]),
                        "ch2_offset": float(current["ch2_offset"]),
                        "ch1_display": bool(int(current["ch1_display"])),
                        "ch2_display": bool(int(current["ch2_display"])),
                    }
                )
                if resource:
                    cfg["resource"] = resource
                save_config(args.config_file, cfg)
                print(f"Stored current scope settings to: {args.config_file}")
                return 0

            if args.read_only:
                return 0

            if args.load:
                cfg = load_config(args.config_file)
                if not cfg:
                    print(f"No config data found in: {args.config_file}")
                    return 2

                time_scale = float(cfg.get("time_scale", current["time_scale"]))
                time_position = cfg.get("time_position", None)
                ch1_scale = float(cfg.get("ch1_scale", current["ch1_scale"]))
                ch2_scale = float(cfg.get("ch2_scale", current["ch2_scale"]))
                ch1_offset = cfg.get("ch1_offset", None)
                ch2_offset = cfg.get("ch2_offset", None)
                ch1_display = cfg.get("ch1_display", None)
                ch2_display = cfg.get("ch2_display", None)

                inst.write(f":TIMebase:SCALe {time_scale}")
                if time_position is not None:
                    inst.write(f":TIMebase:POSition {float(time_position)}")
                inst.write(f":CHANnel1:SCALe {ch1_scale}")
                inst.write(f":CHANnel2:SCALe {ch2_scale}")
                if ch1_offset is not None:
                    inst.write(f":CHANnel1:OFFSet {float(ch1_offset)}")
                if ch2_offset is not None:
                    inst.write(f":CHANnel2:OFFSet {float(ch2_offset)}")
                if ch1_display is not None:
                    inst.write(f":CHANnel1:DISPlay {'ON' if bool(ch1_display) else 'OFF'}")
                if ch2_display is not None:
                    inst.write(f":CHANnel2:DISPlay {'ON' if bool(ch2_display) else 'OFF'}")
                if args.math:
                    if args.math in ("sub", "subtract"):
                        configure_math_ch1_minus_ch2(inst)
                    elif args.math == "off":
                        configure_math_off(inst)

                print_scope_settings("Applied settings from config:", read_scope_settings(inst))
                return 0

            ch1_scale = resolved_ch1_scale
            ch2_scale = resolved_ch2_scale
            ch1_offset = resolved_ch1_offset
            ch2_offset = resolved_ch2_offset

            inst.write(f":TIMebase:SCALe {args.time_scale}")
            if args.time_position is not None:
                inst.write(f":TIMebase:POSition {args.time_position}")

            inst.write(":CHANnel1:DISPlay ON")
            inst.write(":CHANnel2:DISPlay ON")
            inst.write(f":CHANnel1:SCALe {ch1_scale}")
            inst.write(f":CHANnel2:SCALe {ch2_scale}")
            if ch1_offset is not None:
                inst.write(f":CHANnel1:OFFSet {ch1_offset}")
            if ch2_offset is not None:
                inst.write(f":CHANnel2:OFFSet {ch2_offset}")
            if args.math:
                if args.math in ("sub", "subtract"):
                    configure_math_ch1_minus_ch2(inst)
                elif args.math == "off":
                    configure_math_off(inst)

            print_scope_settings("Applied settings:", read_scope_settings(inst))
        return 0
    except PermissionError as exc:
        print(f"Scope communication failed (permission error): {exc}")
        print("Try running with sudo or fix USB device permissions/udev rules.")
        return 1
    except Exception as exc:
        print(f"Scope communication failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
