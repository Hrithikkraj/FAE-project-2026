"""E1: pitch accuracy on synthetic tones with exactly known f0.

Audio concepts: pure tones vs harmonic tones vs the missing fundamental, error in cents,
and octave (gross) errors. Three signal types x four methods x 32 log-spaced
frequencies from 80 to 900 Hz, several random-phase frames each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import synth
from ..config import Config
from ..pitch import make_estimators
from .common import (GROSS_CENTS, METHOD_STYLE, METHODS, ExperimentResult, cents_diff,
                     label, log_spaced, md_table, plot_style, rng_for, write_csv)

SIGNALS = {"sine": "Pure sine", "harmonic": "Harmonic (voice-like)",
           "missing_f0": "Missing fundamental"}


def make_frame(kind: str, f0: float, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One analysis frame of the given signal type, starting at a random phase."""
    sr, n, extra = cfg.sample_rate, cfg.frame_size, 2048
    duration = (n + extra) / sr
    if kind == "sine":
        x = synth.sine_tone(f0, duration, sr, phase=rng.uniform(0, 2 * np.pi))
    else:
        x = synth.voice_like_tone(f0, duration, sr, missing_fundamental=(kind == "missing_f0"))
    start = int(rng.integers(0, extra))
    return x[start:start + n]


def summarise(errors: np.ndarray) -> dict[str, float]:
    """voiced %, MAE / median / max |error| over voiced frames, gross % over all frames."""
    voiced = ~np.isnan(errors)
    e = np.abs(errors[voiced])
    return {
        "voiced_pct": 100.0 * voiced.mean(),
        "mae_cents": float(e.mean()) if len(e) else np.nan,
        "median_cents": float(np.median(e)) if len(e) else np.nan,
        "max_cents": float(e.max()) if len(e) else np.nan,
        "gross_pct": 100.0 * float(np.sum(e > GROSS_CENTS)) / len(errors),
    }


def run(cfg: Config, out_dir: Path, n_freqs: int = 32, trials: int = 5) -> ExperimentResult:
    """Run E1 and write e1_accuracy_detail.csv, e1_accuracy.csv, e1_accuracy.png."""
    rng = rng_for("E1")
    estimators = {e.name: e for e in make_estimators(cfg)}
    freqs = log_spaced(80.0, 900.0, n_freqs)
    detail, errors = [], {(s, m): [] for s in SIGNALS for m in METHODS}
    for kind in SIGNALS:
        for f0 in freqs:
            for trial in range(trials):
                frame = make_frame(kind, f0, cfg, rng)
                for m in METHODS:
                    est = estimators[m].estimate(frame).f0_hz
                    err = float(cents_diff(est, f0)) if est is not None else np.nan
                    errors[(kind, m)].append(err)
                    detail.append({"signal": kind, "f0_hz": float(f0), "trial": trial,
                                   "method": m, "estimate_hz": np.nan if est is None else est,
                                   "error_cents": err})
    summary = []
    for kind in SIGNALS:
        for m in METHODS:
            summary.append({"signal": kind, "method": m,
                            **summarise(np.array(errors[(kind, m)]))})
    write_csv(out_dir / "e1_accuracy_detail.csv", detail)
    write_csv(out_dir / "e1_accuracy.csv", summary)
    plot(summary, out_dir / "e1_accuracy.png")

    rows = [(SIGNALS[r["signal"]], label(r["method"]), r["mae_cents"], r["median_cents"],
             r["max_cents"], r["gross_pct"], r["voiced_pct"]) for r in summary]
    table = md_table(["Signal", "Method", "MAE (c)", "Median (c)", "Max (c)",
                      "Gross >600 c (%)", "Voiced (%)"], rows, digits=1)
    md = (f"{n_freqs} frequencies, log-spaced from 80 to 900 Hz; {trials} random-phase "
          f"frames each (frame {cfg.frame_size}, {cfg.sample_rate} Hz). MAE, median and "
          "max are over voiced frames; the gross-error rate is over all frames.\n\n"
          + table + "\n\n![E1](e1_accuracy.png)")
    return ExperimentResult("E1", "Accuracy on synthetic tones", md, finding(summary),
                            ["e1_accuracy.csv", "e1_accuracy_detail.csv", "e1_accuracy.png"])


def finding(summary: list[dict]) -> str:
    """One data-driven paragraph."""
    get = {(r["signal"], r["method"]): r for r in summary}
    harm_worst = max(get[("harmonic", m)]["mae_cents"] for m in METHODS)
    miss = {m: get[("missing_f0", m)]["gross_pct"] for m in METHODS}
    failed = [label(m) for m in METHODS if miss[m] >= 50]
    ok = [f"{label(m)} {miss[m]:.0f}%" for m in METHODS if miss[m] < 50]
    return (f"On harmonic tones all four methods averaged at most {harm_worst:.1f} cents of "
            f"error. HPS fails on pure sines (gross errors on "
            f"{get[('sine', 'fft_hps')]['gross_pct']:.0f}% of frames) because a sine has no "
            "harmonics to multiply. With the fundamental removed, "
            f"{', '.join(failed) or 'no method'} reported the wrong octave on most frames, "
            f"while the time-domain methods kept their gross-error rate low "
            f"({', '.join(ok)}).")


def plot(summary: list[dict], path: Path) -> None:
    """Grouped bars: MAE (log scale) and gross-error rate per signal type and method."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(len(SIGNALS))
    width = 0.19
    for ax, key, title in [(axes[0], "mae_cents", "Mean absolute error (cents, log scale)"),
                           (axes[1], "gross_pct", "Gross / octave errors (% of frames)")]:
        for i, m in enumerate(METHODS):
            vals = [next(r[key] for r in summary if r["signal"] == s and r["method"] == m)
                    for s in SIGNALS]
            vals = [max(v, 0.01) if key == "mae_cents" else v for v in vals]
            ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=METHOD_STYLE[m][0],
                   label=label(m))
        ax.set_xticks(x, [SIGNALS[s] for s in SIGNALS])
        ax.set_title(title, fontsize=11)
        if key == "mae_cents":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 105)
        plot_style(ax)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("E1: accuracy on synthetic tones, 80–900 Hz", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
