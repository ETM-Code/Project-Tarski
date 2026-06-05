#!/usr/bin/env python3
"""
CHARACTERIZATION (golden-master) build-smoke test for the T1-devboard AVR
firmware (src/device.cpp, src/main.cpp).

PURPOSE: lock in the CURRENT build behavior so any FUTURE change that breaks the
build or bloats the firmware is caught.

This is the hard gate that the pure-logic host test cannot provide: device.cpp
is hardware-bound (Arduino.h / Wire.h / AVR registers) and only the real AVR
toolchain can compile and link it. We therefore:

  1. build_smoke_pio_run_succeeds:
       `pio run` must exit 0 (catches any compile/link regression).
  2. build_smoke_flash_size_within_tolerance:
       avr-size text+data == Flash. Golden 7992 bytes, tolerance +/-64.
  3. build_smoke_ram_size_within_tolerance:
       avr-size data+bss == RAM. Golden 526 bytes, tolerance +/-16.

Golden values were obtained by RUNNING the current code:
    $ pio run        ->  RAM 526 bytes, Flash 7992 bytes
    $ avr-size firmware.elf
        text   data   bss   dec    hex
        7952     40   486   8478   211e
    Flash = text+data = 7992 ; RAM = data+bss = 526

Run with:  pytest  (this file is collected by pytest)
or standalone:  python3 run_build_smoke.py

Hermetic: no network, no hardware, no external services beyond the locally
installed `pio` and `avr-size` toolchain (PlatformIO downloads packages on first
run only). If those tools are missing, the relevant tests SKIP rather than fail,
so the suite stays honest about what it actually verified.
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

# interface/ project root (two levels up from this file).
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIRMWARE_ELF = os.path.join(
    PROJECT_DIR, ".pio", "build", "nanoatmega328new", "firmware.elf"
)

# --- Goldens (observed on current code) ---
GOLDEN_FLASH = 7992
# Tightened from 64 to 16: the observed Flash is a stable, exact 7992 on current
# code, so a wide +/-64 band would absorb a ~32-byte code regression. +/-16 keeps
# headroom for benign toolchain/linker jitter while actually catching small drift.
FLASH_TOL = 16
GOLDEN_RAM = 526
RAM_TOL = 16

PIO = shutil.which("pio")
AVR_SIZE = shutil.which("avr-size")


# Build once for the whole module; share the result across tests.
@pytest.fixture(scope="module")
def pio_build():
    if PIO is None:
        pytest.skip("pio (PlatformIO) not installed -- cannot run build smoke")
    proc = subprocess.run(
        [PIO, "run"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    return proc


def _parse_avr_size():
    """Return (flash, ram) parsed from Berkeley `avr-size` on the firmware elf."""
    assert AVR_SIZE is not None, "avr-size not found"
    assert os.path.exists(FIRMWARE_ELF), f"missing elf: {FIRMWARE_ELF}"
    out = subprocess.run(
        [AVR_SIZE, FIRMWARE_ELF], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, f"avr-size failed: {out.stderr}"
    # Second line: text  data  bss  dec  hex  filename
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert len(lines) >= 2, f"unexpected avr-size output:\n{out.stdout}"
    nums = re.findall(r"\d+", lines[1])
    text, data, bss = int(nums[0]), int(nums[1]), int(nums[2])
    flash = text + data
    ram = data + bss
    return flash, ram


def test_build_smoke_pio_run_succeeds(pio_build):
    """`pio run` must exit 0 -- the compile/link regression gate."""
    assert pio_build.returncode == 0, (
        "pio run failed (exit "
        f"{pio_build.returncode}).\nSTDOUT:\n{pio_build.stdout}\n"
        f"STDERR:\n{pio_build.stderr}"
    )
    assert "[SUCCESS]" in pio_build.stdout


def test_build_smoke_flash_size_within_tolerance(pio_build):
    """Flash (text+data) must stay near the golden -- catches accidental bloat."""
    assert pio_build.returncode == 0, "build must succeed before sizing"
    if AVR_SIZE is None:
        pytest.skip("avr-size not installed")
    flash, _ = _parse_avr_size()
    assert abs(flash - GOLDEN_FLASH) <= FLASH_TOL, (
        f"Flash size {flash} drifted from golden {GOLDEN_FLASH} "
        f"(tolerance +/-{FLASH_TOL}). CHARACTERIZATION: update the golden only "
        f"if the firmware change is intentional."
    )


def test_build_smoke_ram_size_within_tolerance(pio_build):
    """Static RAM (data+bss) must stay near the golden."""
    assert pio_build.returncode == 0, "build must succeed before sizing"
    if AVR_SIZE is None:
        pytest.skip("avr-size not installed")
    _, ram = _parse_avr_size()
    assert abs(ram - GOLDEN_RAM) <= RAM_TOL, (
        f"RAM size {ram} drifted from golden {GOLDEN_RAM} "
        f"(tolerance +/-{RAM_TOL}). CHARACTERIZATION: update the golden only if "
        f"the firmware change is intentional."
    )


# --- Host pure-logic test driver ------------------------------------------
# Compiles and runs test/host/test_device_logic.cpp so the whole component is
# covered by a single `pytest` invocation.
HOST_TEST_SRC = os.path.join(PROJECT_DIR, "test", "host", "test_device_logic.cpp")
INCLUDE_DIR = os.path.join(PROJECT_DIR, "include")
CXX = shutil.which("clang++") or shutil.which("g++")


def test_host_device_logic_compiles_and_passes(tmp_path):
    """Compile + run the host C++ characterization test; expect exit 0."""
    if CXX is None:
        pytest.skip("no C++ compiler (clang++/g++) available")
    binary = os.path.join(str(tmp_path), "test_device_logic")
    compile_proc = subprocess.run(
        [CXX, "-std=c++14", "-Wall", f"-I{INCLUDE_DIR}", HOST_TEST_SRC, "-o", binary],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compile_proc.returncode == 0, (
        f"host test failed to compile:\n{compile_proc.stderr}"
    )
    run_proc = subprocess.run([binary], capture_output=True, text=True, timeout=60)
    assert run_proc.returncode == 0, (
        f"host logic test FAILED:\n{run_proc.stdout}\n{run_proc.stderr}"
    )
    assert "ALL HOST LOGIC TESTS PASSED" in run_proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
