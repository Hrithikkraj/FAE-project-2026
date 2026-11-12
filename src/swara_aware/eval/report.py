"""Write results/RESULTS.md from the experiment results.

Audio concept: honest reporting. Tables and findings are generated from the measured
numbers, so the report cannot drift from the data.
"""

from __future__ import annotations

import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from .common import MASTER_SEED, ExperimentResult


def write_report(results: list[ExperimentResult], cfg: Config, path: Path,
                 elapsed_s: float) -> None:
    """Assemble RESULTS.md: header, summary of findings, one section per experiment."""
    lines = [
        "# Swara-Aware: evaluation results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python scripts/run_all_experiments.py` "
        f"in {elapsed_s:.0f} s. Master random seed **{MASTER_SEED}**, so every number below "
        "is reproducible. The tanpura drone was **off** for every experiment. Python "
        f"{platform.python_version()}, numpy {np.__version__}, {platform.system()} "
        f"{platform.machine()}.",
        "",
        "Default analysis settings: " + ", ".join(
            f"{k}={v}" for k, v in asdict(cfg).items() if k in (
                "sample_rate", "frame_size", "hop_size", "fmin", "fmax",
                "silence_threshold_dbfs", "yin_threshold", "in_tune_cents", "close_cents",
                "stable_min_ms")) + ".",
        "",
        "## Key findings",
        "",
    ]
    for r in results:
        lines.append(f"- **{r.key}, {r.title}:** {r.finding}")
    for r in results:
        lines += ["", f"## {r.key}: {r.title}", "", r.markdown, "",
                  f"**Finding.** {r.finding}"]
        if r.files:
            lines += ["", "Files: " + ", ".join(f"`{f}`" for f in r.files)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
