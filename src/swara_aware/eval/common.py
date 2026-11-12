"""Shared helpers for the Stage 5 experiments: seeds, styles, CSV and markdown tables.

Audio concept: reproducible measurement. Every experiment draws its random numbers from
a generator seeded with one master seed plus the experiment's name, so each experiment
gives identical results on every run and does not depend on which others ran first.
"""

from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..pitch import METHOD_LABELS

MASTER_SEED = 2026
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
METHODS = list(METHOD_LABELS)  # fft_peak, fft_hps, autocorr, yin
GROSS_CENTS = 600.0            # error above this = octave / gross error
WITHIN_CENTS = 25.0            # "correct" for robustness tables

# Fixed colour per method (validated categorical palette, fixed order), plus a marker
# and line style so methods are distinguishable without colour.
METHOD_STYLE = {
    "fft_peak": ("#2a78d6", "o", "-"),
    "fft_hps": ("#eb6834", "s", "--"),
    "autocorr": ("#1baf7a", "^", "-."),
    "yin": ("#eda100", "D", ":"),
}


def rng_for(name: str, seed: int = MASTER_SEED) -> np.random.Generator:
    """Independent, reproducible random generator for one experiment."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def log_spaced(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` frequencies spaced evenly in log frequency (equal musical steps)."""
    return np.geomspace(lo, hi, n)


def cents_diff(estimate: Any, truth: Any) -> np.ndarray:
    """1200 * log2(estimate / truth); NaN where the estimate is missing/unvoiced."""
    est = np.asarray(estimate, dtype=float)
    return 1200.0 * np.log2(est / np.asarray(truth, dtype=float))


@dataclass
class ExperimentResult:
    """What one experiment hands to the report."""

    key: str                    # "E1" ...
    title: str
    markdown: str               # tables and notes (already formatted)
    finding: str                # one plain-English sentence or two
    files: list[str] = field(default_factory=list)  # CSV/PNG names written
    ran: bool = True


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a list of dicts as CSV (columns from the first row); floats to 4 dp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def fmt(value: Any, digits: int = 1) -> str:
    """Format a number for a markdown table ('–' for NaN/None)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]], digits: int = 1) -> str:
    """A GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def label(method: str) -> str:
    """Display name of a method."""
    return METHOD_LABELS.get(method, method)


def plot_style(ax) -> None:  # noqa: ANN001
    """Recessive grid and spines for every experiment figure."""
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
