"""E6: sensitivity to frame size and to the YIN threshold.

Audio concepts: the time-frequency trade-off (a longer frame holds more periods, so the
pitch is more precise and lower notes can be tracked, but the result arrives later),
and a detection threshold trading missed notes against false detections in noise.

Frame sizes 1024 / 2048 / 4096 with hop = frame / 4 (constant 75% overlap). A frame must
hold about two periods of the lowest note, so the search floor fmin is raised for short
frames: fmin = max(70 Hz, 2.05 * sample_rate / frame_size).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import dsp, synth
from ..config import Config
from ..pitch import YinEstimator, make_estimators
from .common import (METHOD_STYLE, METHODS, WITHIN_CENTS, ExperimentResult, cents_diff,
                     label, log_spaced, md_table, plot_style, rng_for, write_csv)

FRAME_SIZES = [1024, 2048, 4096]
YIN_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def frame_config(cfg: Config, n: int) -> Config:
    """Config for frame size ``n``: hop n/4 and a feasible fmin."""
    return replace(cfg, frame_size=n, hop_size=n // 4,
                   fmin=max(cfg.fmin, 2.05 * cfg.sample_rate / n))


WEAK_F0_AMPS = [0.15, 1.0, 0.8, 0.6, 0.4, 0.3]  # 2nd harmonic ~7x stronger than f0


def tone_frames(cfg: Config, rng: np.random.Generator, freqs: np.ndarray, trials: int,
                snr: float | None, amps: list[float] | None = None) -> list[tuple[float, np.ndarray]]:
    """(f0, frame) pairs of harmonic tones, optionally with white noise at ``snr`` dB."""
    sr, n = cfg.sample_rate, cfg.frame_size
    out = []
    for f0 in freqs:
        for _ in range(trials):
            start = int(rng.integers(0, 2048))
            x = synth.voice_like_tone(f0, (n + 2048) / sr, sr, amps=amps)[start:start + n]
            out.append((f0, x if snr is None else dsp.add_noise(x, snr, rng=rng)))
    return out


def evaluate(est, frames: list[tuple[float, np.ndarray]]) -> dict[str, float]:  # noqa: ANN001
    """% within 25 c, % voiced, median |error| and octave-error % (voiced), µs per frame."""
    errors, t_total = [], 0.0
    for f0, frame in frames:
        t0 = time.perf_counter()
        r = est.estimate(frame)
        t_total += time.perf_counter() - t0
        errors.append(abs(float(cents_diff(r.f0_hz, f0))) if r.f0_hz is not None else np.nan)
    e = np.array(errors)
    v = e[~np.isnan(e)]
    return {"within": 100.0 * float(np.mean(np.nan_to_num(e, nan=np.inf) <= WITHIN_CENTS)),
            "voiced": 100.0 * float(np.mean(~np.isnan(e))),
            "median": float(np.median(v)) if len(v) else np.nan,
            "octave": 100.0 * float(np.mean(v > 600)) if len(v) else np.nan,
            "us": 1e6 * t_total / len(frames)}


def run(cfg: Config, out_dir: Path, n_freqs: int = 12, trials: int = 8) -> ExperimentResult:
    """Run E6 and write e6_frame_size.csv, e6_yin_threshold.csv and e6_sensitivity.png."""
    rng = rng_for("E6")
    freqs_common = log_spaced(100.0, 900.0, n_freqs)  # trackable by every frame size
    low_freqs = log_spaced(70.0, 100.0, 4)            # where short frames must fail

    frame_rows = []
    for n in FRAME_SIZES:
        c = frame_config(cfg, n)
        clean = tone_frames(c, rng, freqs_common, trials, None)
        noisy = tone_frames(c, rng, freqs_common, trials, 10.0)
        low = tone_frames(c, rng, low_freqs, trials, None)
        for est in make_estimators(c):
            r_clean, r_noisy, r_low = (evaluate(est, clean), evaluate(est, noisy),
                                       evaluate(est, low))
            frame_ms = 1000.0 * n / c.sample_rate
            hop_ms = 1000.0 * c.hop_size / c.sample_rate
            frame_rows.append({
                "frame_size": n, "hop_size": c.hop_size, "fmin_hz": c.fmin, "method": est.name,
                "within_25c_clean_pct": r_clean["within"],
                "within_25c_10db_pct": r_noisy["within"],
                "median_err_10db_cents": r_noisy["median"],
                "within_25c_70_100hz_pct": r_low["within"], "proc_us": r_noisy["us"],
                "latency_ms": frame_ms + hop_ms + r_noisy["us"] / 1000.0})

    yin_rows = []
    noise_frames = [0.1 * rng.standard_normal(cfg.frame_size) for _ in range(60)]
    noise_frames += [synth.pink_noise(cfg.frame_size / cfg.sample_rate, cfg.sample_rate, 0.1, rng)
                     for _ in range(60)]
    tone_sets = {snr: tone_frames(cfg, rng, log_spaced(80, 900, n_freqs), trials, snr)
                 for snr in (20.0, 10.0, 5.0)}
    weak = (tone_frames(cfg, rng, log_spaced(80, 900, n_freqs), trials, None, WEAK_F0_AMPS)
            + tone_frames(cfg, rng, log_spaced(80, 900, n_freqs), trials, 10.0, WEAK_F0_AMPS))
    for th in YIN_THRESHOLDS:
        yin = YinEstimator(cfg.sample_rate, cfg.fmin, cfg.fmax, threshold=th,
                           silence_threshold_dbfs=None)  # gate off: test YIN's own decision
        row = {"threshold": th}
        for snr, frames in tone_sets.items():
            r = evaluate(yin, frames)
            row[f"within_25c_{snr:.0f}db_pct"] = r["within"]
            row[f"voiced_{snr:.0f}db_pct"] = r["voiced"]
        r = evaluate(yin, weak)
        row["within_25c_weak_f0_pct"] = r["within"]
        row["octave_err_weak_f0_pct"] = r["octave"]
        row["false_voiced_noise_pct"] = 100.0 * float(np.mean(
            [yin.estimate(f).f0_hz is not None for f in noise_frames]))
        yin_rows.append(row)

    write_csv(out_dir / "e6_frame_size.csv", frame_rows)
    write_csv(out_dir / "e6_yin_threshold.csv", yin_rows)
    plot(frame_rows, yin_rows, out_dir / "e6_sensitivity.png")

    t1 = md_table(["Frame", "Hop", "fmin (Hz)", "Method", "Clean (%)", "10 dB (%)",
                   "Median err @10 dB (c)", "70–100 Hz (%)", "µs/frame", "Latency (ms)"],
                  [(r["frame_size"], r["hop_size"], r["fmin_hz"], label(r["method"]),
                    r["within_25c_clean_pct"], r["within_25c_10db_pct"],
                    r["median_err_10db_cents"], r["within_25c_70_100hz_pct"], r["proc_us"],
                    r["latency_ms"]) for r in frame_rows], 1)
    t2 = md_table(["YIN threshold", "≤25 c @20 dB (%)", "≤25 c @10 dB (%)", "≤25 c @5 dB (%)",
                   "Voiced @5 dB (%)", "Weak-f0 ≤25 c (%)", "Weak-f0 octave err (%)",
                   "False voiced on loud noise (%)"],
                  [(f"{r['threshold']:.2f}", r["within_25c_20db_pct"], r["within_25c_10db_pct"],
                    r["within_25c_5db_pct"], r["voiced_5db_pct"], r["within_25c_weak_f0_pct"],
                    r["octave_err_weak_f0_pct"], r["false_voiced_noise_pct"])
                   for r in yin_rows], 1)
    md = ("**Frame size.** Harmonic tones 100–900 Hz (clean and with 10 dB white noise), "
          "plus low notes at 70–100 Hz. % = frames voiced and within 25 cents. Latency = "
          "frame + hop + processing.\n\n" + t1 +
          "\n\n**YIN threshold** (silence gate off). Harmonic tones 80–900 Hz in white noise; "
          "'weak-f0' = tones whose 2nd harmonic is ~7x stronger than the fundamental "
          "(clean and 10 dB), where a loose threshold could cause octave errors; "
          "'false voiced' = pitch reported on 120 frames of pure white or pink noise at "
          "-20 dBFS.\n\n" + t2 + "\n\n![E6](e6_sensitivity.png)")
    return ExperimentResult("E6", "Sensitivity: frame size and YIN threshold", md,
                            finding(frame_rows, yin_rows),
                            ["e6_frame_size.csv", "e6_yin_threshold.csv", "e6_sensitivity.png"])


def finding(frame_rows: list[dict], yin_rows: list[dict]) -> str:
    def get(n: int, m: str, key: str) -> float:
        return next(r[key] for r in frame_rows if r["frame_size"] == n and r["method"] == m)
    lat = {n: get(n, "yin", "latency_ms") for n in FRAME_SIZES}
    med = {n: get(n, "yin", "median_err_10db_cents") for n in FRAME_SIZES}
    low_1024 = get(1024, "yin", "within_25c_70_100hz_pct")
    default = next(r for r in yin_rows if abs(r["threshold"] - 0.15) < 1e-9)
    loose = yin_rows[-1]
    free = loose["false_voiced_noise_pct"] < 5 and loose["octave_err_weak_f0_pct"] < 5
    return (f"For YIN, going from 1024 to 2048 to 4096-sample frames raises latency "
            f"{lat[1024]:.0f} → {lat[2048]:.0f} → {lat[4096]:.0f} ms while the median error at "
            f"10 dB falls {med[1024]:.1f} → {med[2048]:.1f} → {med[4096]:.1f} cents; a 1024 frame "
            f"also cannot reach notes below ~88 Hz ({low_1024:.0f}% correct at 70–100 Hz). "
            f"YIN threshold: at our default 0.15, 5 dB accuracy was "
            f"{default['within_25c_5db_pct']:.0f}% (it answers 'unvoiced'); at "
            f"{loose['threshold']:.2f} it was {loose['within_25c_5db_pct']:.0f}% with "
            f"{loose['false_voiced_noise_pct']:.0f}% false voicing on pure noise and "
            f"{loose['octave_err_weak_f0_pct']:.0f}% octave errors on weak-fundamental tones. "
            f"So in these synthetic tests a looser threshold "
            f"{'cost nothing' if free else 'had a cost'}; we kept 0.15 because real voices "
            "add breath noise and consonants that synthetic noise does not, so the threshold "
            "should be re-checked on recordings (E4) before changing it.")


def plot(frame_rows: list[dict], yin_rows: list[dict], path: Path) -> None:
    """Accuracy vs latency per frame size; YIN threshold trade-off."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    ax = axes[0]
    for m in METHODS:
        color, marker, ls = METHOD_STYLE[m]
        pts = [(r["latency_ms"], r["median_err_10db_cents"], r["frame_size"])
               for r in frame_rows if r["method"] == m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, ls=ls,
                lw=2, ms=7, label=label(m))
    for r in frame_rows:
        if r["method"] == "yin":
            ax.annotate(f"{r['frame_size']}", (r["latency_ms"], r["median_err_10db_cents"]),
                        xytext=(4, -12), textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_yscale("log")
    ax.set(xlabel="theoretical latency (ms)", ylabel="median |error| at 10 dB (cents, log)",
           title="Frame size: precision vs latency (lower-left is better)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    plot_style(ax)

    ax = axes[1]
    th = [r["threshold"] for r in yin_rows]
    for key, lab, color, marker in [
            ("within_25c_10db_pct", "tones @10 dB within 25 c", "#2a78d6", "o"),
            ("within_25c_5db_pct", "tones @5 dB within 25 c", "#1baf7a", "^"),
            ("within_25c_weak_f0_pct", "weak-fundamental tones within 25 c", "#6250d6", "D"),
            ("false_voiced_noise_pct", "pitch reported on pure noise", "#e34948", "s")]:
        ax.plot(th, [r[key] for r in yin_rows], color=color, marker=marker, lw=2, ms=7, label=lab)
    ax.axvline(0.15, color="#52514e", ls="--", lw=1)
    ax.text(0.152, 4, "default 0.15", fontsize=8, color="#52514e")
    ax.set(xlabel="YIN threshold", ylabel="%", title="YIN threshold trade-off", ylim=(0, 103))
    ax.legend(fontsize=9, frameon=False, loc="center right")
    plot_style(ax)
    fig.suptitle("E6: sensitivity", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
