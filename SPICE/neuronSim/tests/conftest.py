"""
Pytest fixtures / path setup for the spice-python characterization suite.

These are CHARACTERIZATION (golden-master) tests: they lock in the CURRENT
behavior of the netlist/generator scripts so any future behavior change is
caught by a failing test. They never assert correctness, only "no change".

The generator scripts resolve default JSON config paths *relative to the
current working directory* (e.g. ``open("defaults/neuron_default.json")``),
so every test that loads a default config is run with cwd == the component
directory. We also put that directory on sys.path so the modules import.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Component directory == parent of this tests/ dir.
COMPONENT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Make the generator modules importable.
if str(COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENT_DIR))


@pytest.fixture(autouse=True)
def _chdir_component(monkeypatch):
    """Run every test with cwd == component dir so relative default JSON paths resolve."""
    monkeypatch.chdir(COMPONENT_DIR)
    yield


def golden(name: str) -> str:
    """Return the contents of a committed golden snapshot file (exact bytes)."""
    return (GOLDEN_DIR / name).read_text()
