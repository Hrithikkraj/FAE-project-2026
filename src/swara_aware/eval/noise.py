"""E2: robustness to additive noise at controlled SNR.

Audio concepts: signal-to-noise ratio (SNR, in dB, from signal and noise power), white
vs pink noise, and recorded real-world noise. Harmonic voice-like tones; a frame counts
as correct if it is voiced and within 25 cents (an unvoiced answer counts as a miss).
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

from .. import dsp, synth
from ..audio_io import load_wav
from ..config import Config
from ..pitch import make_estimators
from .common import (DATA_DIR, METHOD_STYLE, METHODS, WITHIN_CENTS, ExperimentResult,
                     cents_diff, label, log_spaced, md_table, plot_style, rng_for, write_csv)

SNRS = [30.0, 20.0, 10.0, 5.0, 0.0]


def recorded_noises(cfg: Config) -> dict[str, np.ndarray]:
    """Noise recordings from data/noise/*.wav, resampled to the analysis rate."""
    from scipy.signal import resample_poly

    out = {}
    for path in sorted((DATA_DIR / "noise").glob("*.wav")):
        x, sr = load_wav(path)
        if sr != cfg.sample_rate:
            g = gcd(sr, cfg.sample_rate)
            x = resample_poly(x, cfg.sample_rate // g, sr // g)
        if len(x) >= 2 * cfg.frame_size:
            out[f"recorded:{path.stem}"] = x
    return out


def noise_frame(kind: str, n: int, sr: int, rng: np.random.Generator,
                recordings: dict[str, np.ndarray]) -> np.ndarray:
    """One frame of noise of the given kind (random segment for recordings)."""
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":
        return synth.pink_noise(n / sr, sr, rng=rng)
    rec = recordings[kind]
    start = int(rng.integers(0, len(rec) - n))
    return rec[start:start + n]


def run(cfg: Config, out_dir: Path, n_freqs: int = 12, trials: int = 8) -> ExperimentResult:
    """Run E2 and write e2_noise.csv and e2_noise.png."""
    rng = rng_for("E2")
    estimators = {e.name: e for e in make_estimators(cfg)}
    recordings = recorded_noises(cfg)
    kinds = ["white", "pink"] + list(recordings)
    freqs = log_spaced(80.0, 900.0, n_freqs)
    sr, n = cfg.sample_rate, cfg.frame_size
    rows = []
    for kind in kinds:
        for snr in [np.inf] + SNRS:
            hits = {m: [] for m in METHODS}
            for f0 in freqs:
                for _ in range(trials):
                    start = int(rng.integers(0, 2048))
                    clean = synth.voice_like_tone(f0, (n + 2048) / sr, sr)[start:start + n]
                    frame = clean if np.isinf(snr) else dsp.add_noise(
                        clean, snr, noise=noise_frame(kind, n, sr, rng, recordings))
                    for m in METHODS:
                        est = estimators[m].estimate(frame).f0_hz
                        err = abs(float(cents_diff(est, f0))) if est is not None else np.inf
                        hits[m].append(err)
            for m in METHODS:
                e = np.array(hits[m])
                rows.append({"noise": kind, "snr_db": "clean" if np.isinf(snr) else snr,
                             "method": m,
                             "within_25c_pct": 100.0 * float(np.mean(e <= WITHIN_CENTS)),
                             "voiced_pct": 100.0 * float(np.mean(np.isfinite(e))),
                             "median_err_cents": float(np.median(e[np.isfinite(e)]))
                             if np.isfinite(e).any() else np.nan})
    write_csv(out_dir / "e2_noise.csv", rows)
    plot(rows, kinds, out_dir / "e2_noise.png")

    header = ["Noise", "Method", "clean"] + [f"{s:.0f} dB" for s in SNRS]
    table_rows = []
    for kind in kinds:
        for m in METHODS:
            vals = [r["within_25c_pct"] for r in rows if r["noise"] == kind and r["method"] == m]
            table_rows.append([kind, label(m)] + vals)
    note = ("" if recordings else "\n\nNo recorded noise found in `data/noise/`; only "
            "synthetic white and pink noise were used. Add fan or traffic WAVs there and "
            "re-run to include them.")
    md = (f"Harmonic tones at {n_freqs} log-spaced frequencies (80–900 Hz), {trials} "
          f"frames each, per noise type and SNR. Values are the **% of frames voiced and "
          f"within {WITHIN_CENTS:.0f} cents**.\n\n" + md_table(header, table_rows, 0)
          + note + "\n\n![E2](e2_noise.png)")
    return ExperimentResult("E2", "Noise robustness", md, finding(rows),
                            ["e2_noise.csv", "e2_noise.png"])


def finding(rows: list[dict]) -> str:
    def val(kind: str, snr: float, m: str, key: str = "within_25c_pct") -> float:
        return next(r[key] for r in rows
                    if r["noise"] == kind and r["snr_db"] == snr and r["method"] == m)
    at10 = {m: val("white", 10.0, m) for m in METHODS}
    at0 = {m: val("white", 0.0, m) for m in METHODS}
    pink0 = {m: val("pink", 0.0, m) for m in METHODS}
    best0 = max(METHODS, key=lambda m: at0[m])
    yin_voiced0 = val("white", 0.0, "yin", "voiced_pct")
    yin_note = (f" YIN fell to {at0['yin']:.0f}%: it said 'unvoiced' on "
                f"{100 - yin_voiced0:.0f}% of those frames (its 0.15 threshold refuses to "
                "guess) rather than giving wrong pitches." if yin_voiced0 < 50 else "")
    return (f"At 10 dB white noise: {', '.join(f'{label(m)} {at10[m]:.0f}%' for m in METHODS)} "
            f"of frames within 25 cents. At 0 dB, {label(best0)} held up best "
            f"({at0[best0]:.0f}%).{yin_note} Pink noise at 0 dB: "
            f"{', '.join(f'{label(m)} {pink0[m]:.0f}%' for m in METHODS)}.")


def plot(rows: list[dict], kinds: list[str], path: Path) -> None:
    """% within 25 cents vs SNR, one panel per noise type, one line per method."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for m in METHODS:
            color, marker, ls = METHOD_STYLE[m]
            ys = [next(r["within_25c_pct"] for r in rows if r["noise"] == kind
                       and r["snr_db"] == s and r["method"] == m) for s in SNRS]
            ax.plot(SNRS, ys, color=color, marker=marker, ls=ls, lw=2, ms=7, label=label(m))
        ax.set_xticks(SNRS)
        ax.invert_xaxis()  # noisier to the right
        ax.set(title=f"{kind} noise", xlabel="SNR (dB)", ylim=(-3, 103))
        plot_style(ax)
    axes[0][0].set_ylabel(f"% frames within {WITHIN_CENTS:.0f} cents")
    axes[0][0].legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle("E2: harmonic tones in noise", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
