"""Shared fixtures for the tarski_board characterization suite.

These tests LOCK IN the current behavior of tarski_board.py so that any
future change that alters behavior is caught by a failing test. They never
modify the production code and never touch real serial hardware: pure helpers
are tested directly, and methods that would need a serial port are exercised
on an instance built via ``__new__`` (bypassing ``__init__``/serial setup)
with the serial-touching collaborators monkeypatched.
"""

import os
import sys

import pytest

# Make tarski_board importable regardless of pytest's rootdir/cwd.
_COMPONENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPONENT_ROOT not in sys.path:
    sys.path.insert(0, _COMPONENT_ROOT)

import tarski_board as tb  # noqa: E402


@pytest.fixture
def board():
    """A TarskiBoard instance with NO serial port.

    Built via ``__new__`` so ``__init__`` (which opens a real serial port)
    never runs. Only pure-logic methods that don't touch ``self.ser`` are
    safe to call on it directly; callers that need serial collaborators
    must monkeypatch them.
    """
    return tb.TarskiBoard.__new__(tb.TarskiBoard)
