"""Test setup: make ``src/`` importable, and print a measurement report after the run.

Tests call the ``report`` fixture with lines of text (measured errors, voicing rates).
They are printed in a "Swara-Aware measurement report" section at the end of every
pytest run, so the actual numbers are visible even when all tests pass.
"""

import os
import sys
from pathlib import Path
from typing import Callable

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # plots in tests are saved, never shown
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPORT_LINES: list[str] = []


@pytest.fixture
def report() -> Callable[[str], None]:
    """Return a function that adds one line to the end-of-run report."""
    return _REPORT_LINES.append


def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001 (pytest hook)
    """Print the collected report lines after the test results."""
    if _REPORT_LINES:
        terminalreporter.section("Swara-Aware measurement report")
        for line in _REPORT_LINES:
            terminalreporter.write_line(line)
