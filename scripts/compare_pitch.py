"""Stage 2: compare our pitch estimators with each other and with librosa's pYIN.

Audio concepts: pitch tracking over time (framing), error in cents against ground
truth, voicing decisions, and processing time per frame vs the real-time budget
(one hop = hop_size / sample_rate seconds).

Usage (from the swara-aware folder):
    python scripts/compare_pitch.py                    # synthetic glide only
    python scripts/compare_pitch.py --wav data/x.wav   # also a real recording
    python scripts/compare_pitch.py --no-reference     # skip librosa pYIN

Saves results/compare_pitch.png and prints error and timing tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swara_aware import audio_io, dsp, synth  # noqa: E402
from swara_aware.config import Config  # noqa: E402
from swara_aware.pitch import cents_error, estimate_track, make_estimators  # noqa: E402

# Fixed colour per method (validated categorical palette, fixed order) + line style,
# so identity never depends on colour alone.
STYLE = {
    "fft_peak": ("#2a78d6", "-"),
    "fft_hps": ("#eb6834", "--"),
    "autocorr": ("#1baf7a", "-."),
    "yin": ("#eda100", "-"),
    "pyin (ref)": ("#e87ba4", ":"),
}
TRUTH_COLOR = "#0b0b0b"
GROSS_CENTS = 50.0  # an error bigger than this is a "gross" error (wrong note)


@dataclass
class Track:
    """One method's pitch track over a signal."""

    name: str
    times: np.ndarray
    f0: np.ndarray
    seconds: float  # total processing time for the whole signal


def synthetic_glide(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """0.4 s silence, 3 s voice-like glide 150 -> 300 Hz, 0.4 s silence, 30 dB pink noise.

    Returns ``(signal, true_f0_per_sample)`` with NaN where the true signal is silent.
    """
    sr = cfg.sample_rate
    pad = np.zeros(int(0.4 * sr))
    voice = synth.voice_like_glide(150.0, 300.0, 3.0, sr, amplitude=0.4)
    truth_voice = synth.glide_frequencies(150.0, 300.0, 3.0, sr)
    x = np.concatenate([pad, voice, pad])
    truth = np.concatenate([np.full(len(pad), np.nan), truth_voice, np.full(len(pad), np.nan)])
    # Pink noise over the whole signal (also the "silent" parts), 30 dB below the voice.
    # SNR = 10*log10(P_voice / P_noise), so the noise gain is sqrt(P_voice / (P_noise * 10^3)).
    noise = synth.pink_noise(len(x) / sr, sr, rng=rng)
    gain = np.sqrt(dsp.signal_power(voice) / (dsp.signal_power(noise) * 10 ** (30.0 / 10)))
    return x + gain * noise, truth


def truth_at_frames(truth: np.ndarray, times: np.ndarray, cfg: Config) -> np.ndarray:
    """True f0 at each frame centre; NaN unless the WHOLE frame is inside the voiced part.

    Frames that straddle an onset/offset are excluded from scoring: they contain both
    silence and tone, so there is no single right answer.
    """
    half = cfg.frame_size // 2
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        c = int(round(t * cfg.sample_rate))
        lo, hi = c - half, c + half
        if lo >= 0 and hi <= len(truth) and not np.any(np.isnan(truth[lo:hi])):
            out[i] = truth[c]
    return out


def silent_frames(truth: np.ndarray, times: np.ndarray, cfg: Config) -> np.ndarray:
    """True for frames that lie completely in the silent (noise-only) parts."""
    half = cfg.frame_size // 2
    mask = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        c = int(round(t * cfg.sample_rate))
        lo, hi = max(c - half, 0), min(c + half, len(truth))
        mask[i] = np.all(np.isnan(truth[lo:hi]))
    return mask


def run_all(x: np.ndarray, cfg: Config, use_reference: bool) -> list[Track]:
    """Run every estimator (and optionally pYIN) over the signal, timing each."""
    tracks = []
    for est in make_estimators(cfg):
        start = time.perf_counter()
        times, f0, _ = estimate_track(x, est, cfg.frame_size, cfg.hop_size)
        tracks.append(Track(est.name, times, f0, time.perf_counter() - start))
    if use_reference:
        try:
            from swara_aware.pitch.reference import reference_track

            start = time.perf_counter()
            times, f0, _ = reference_track(x, cfg.sample_rate, cfg.fmin, cfg.fmax,
                                           cfg.frame_size, cfg.hop_size)
            tracks.append(Track("pyin (ref)", times, f0, time.perf_counter() - start))
        except ImportError as exc:
            print(f"Skipping reference tracker: {exc}")
    return tracks


def score(track: Track, truth_f: np.ndarray, silent: np.ndarray) -> dict[str, float]:
    """MAE in cents, gross error %, voicing recall %, false-voicing % against truth."""
    scored = ~np.isnan(truth_f)
    both = scored & ~np.isnan(track.f0)
    err = np.abs(cents_error(track.f0[both], truth_f[both]))
    return {
        "mae": float(np.mean(err)) if len(err) else np.nan,
        "median": float(np.median(err)) if len(err) else np.nan,
        "gross": 100.0 * float(np.mean(err > GROSS_CENTS)) if len(err) else np.nan,
        "recall": 100.0 * both.sum() / max(scored.sum(), 1),
        "false_voiced": 100.0 * float(np.mean(~np.isnan(track.f0[silent])))
        if silent.any() else np.nan,
    }


def print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    """Print one error table."""
    print(f"\n{title}")
    print(f"  {'method':12s} {'MAE (c)':>9s} {'median':>8s} {'gross%':>8s} "
          f"{'voiced%':>8s} {'false-voiced%':>14s}")
    for name, s in rows.items():
        print(f"  {name:12s} {s['mae']:9.2f} {s['median']:8.2f} {s['gross']:8.1f} "
              f"{s['recall']:8.1f} {s['false_voiced']:14.1f}")


def time_per_frame(x: np.ndarray, cfg: Config, n_frames: int = 500) -> dict[str, float]:
    """Mean processing time of ``estimate`` per frame in microseconds (voiced frames)."""
    frames = dsp.frame_signal(x, cfg.frame_size, cfg.hop_size)
    loud = frames[dsp.silence_gate(frames, cfg.silence_threshold_dbfs)]
    reps = int(np.ceil(n_frames / len(loud)))
    batch = np.tile(loud, (reps, 1))[:n_frames]
    out = {}
    for est in make_estimators(cfg):
        est.estimate(batch[0])  # warm-up
        start = time.perf_counter()
        for frame in batch:
            est.estimate(frame)
        out[est.name] = 1e6 * (time.perf_counter() - start) / len(batch)
    return out


def plot_panel(axes: np.ndarray, x: np.ndarray, sr: int, cfg: Config, tracks: list[Track],
               truth_f: np.ndarray | None, title: str, err_ref: str) -> None:
    """Spectrogram with f0 tracks (top) and error in cents vs time (bottom)."""
    ax_spec, ax_err = axes
    times, freqs, S = dsp.spectrogram(x, sr, 4096, cfg.hop_size)  # longer FFT: finer Hz
    keep = freqs <= 1000
    ax_spec.pcolormesh(times, freqs[keep], dsp.amplitude_to_db(S[keep], -90), shading="auto",
                       cmap="Greys", vmin=-90, vmax=-10)
    if truth_f is not None:
        ax_spec.plot(tracks[0].times, truth_f, color=TRUTH_COLOR, lw=3.0, alpha=0.35,
                     label="ground truth")
    for tr in tracks:
        color, ls = STYLE[tr.name]
        ax_spec.plot(tr.times, tr.f0, color=color, ls=ls, lw=1.6, label=tr.name)
    fmax_plot = 1.3 * np.nanmax([np.nanmax(t.f0) if np.any(~np.isnan(t.f0)) else 0
                                for t in tracks] + [100])
    ax_spec.set(title=title, ylabel="Frequency (Hz)", ylim=(0, min(fmax_plot, 1000)))
    ax_spec.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

    if truth_f is not None:
        ref_f = truth_f
    else:
        ref_tracks = [t for t in tracks if t.name == err_ref]
        ref_f = ref_tracks[0].f0 if ref_tracks else None
    if ref_f is None:
        ax_err.set_visible(False)
        return
    for tr in tracks:
        if tr.name == err_ref:
            continue
        n = min(len(tr.f0), len(ref_f))
        color, ls = STYLE[tr.name]
        ax_err.plot(tr.times[:n], cents_error(tr.f0[:n], ref_f[:n]), color=color, ls=ls,
                    lw=1.4, label=tr.name)
    ax_err.axhspan(-cfg.in_tune_cents, cfg.in_tune_cents, color="#d9d9d6", alpha=0.5, lw=0,
                   label=f"±{cfg.in_tune_cents:.0f} c (in tune)")
    ax_err.axhline(0, color="#52514e", lw=0.8)
    ax_err.set(ylabel=f"Error vs {err_ref} (cents)", xlabel="Time (s)", ylim=(-60, 60),
               xlim=ax_spec.get_xlim())
    ax_err.legend(loc="upper left", fontsize=8, ncol=5, framealpha=0.9)
    ax_err.grid(axis="y", color="#e6e6e3", lw=0.6)


def main() -> None:
    """Build the signals, run all methods, print tables, save the figure."""
    parser = argparse.ArgumentParser(description="Compare Swara-Aware pitch estimators.")
    parser.add_argument("--wav", help="also analyse this recording (pYIN used as reference)")
    parser.add_argument("--no-reference", action="store_true", help="skip librosa pYIN")
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(0)
    use_ref = not args.no_reference

    # --- synthetic glide with exact ground truth ---
    x, truth = synthetic_glide(cfg, rng)
    tracks = run_all(x, cfg, use_ref)
    truth_f = truth_at_frames(truth, tracks[0].times, cfg)
    silent = silent_frames(truth, tracks[0].times, cfg)
    print_table("Synthetic voice glide 150 -> 300 Hz (3 s, 30 dB pink noise) vs exact truth",
                {t.name: score(t, truth_f, silent) for t in tracks})
    panels = [(x, cfg.sample_rate, tracks, truth_f,
               "Synthetic voice glide 150→300 Hz, 30 dB pink noise", "ground truth")]

    # --- optional real recording, scored against pYIN ---
    if args.wav:
        y, sr = audio_io.load_wav(args.wav)
        if sr != cfg.sample_rate:
            from dataclasses import replace
            cfg_wav = replace(cfg, sample_rate=sr)
        else:
            cfg_wav = cfg
        wav_tracks = run_all(y, cfg_wav, use_ref)
        ref = [t for t in wav_tracks if t.name == "pyin (ref)"]
        if ref:
            ref_f = ref[0].f0
            no_silent = np.zeros(len(ref_f), dtype=bool)
            print_table(f"Recording {Path(args.wav).name} vs pYIN (approximate truth; "
                        "'voiced%' = share of pYIN-voiced frames we also voiced)",
                        {t.name: score(t, ref_f, no_silent) for t in wav_tracks
                         if t.name != "pyin (ref)"})
        panels.append((y, sr, wav_tracks, None, f"Recording: {Path(args.wav).name}",
                       "pyin (ref)"))

    # --- timing ---
    hop_us = 1e6 * cfg.hop_size / cfg.sample_rate
    print(f"\nProcessing time per frame (frame {cfg.frame_size}, mean of 500 voiced frames). "
          f"Real-time budget = one hop = {hop_us:.0f} us")
    for name, us in time_per_frame(x, cfg).items():
        print(f"  {name:12s} {us:9.1f} us   ({100 * us / hop_us:5.1f}% of budget)")
    for t in tracks:
        if t.name == "pyin (ref)":
            us = 1e6 * t.seconds / len(t.f0)
            print(f"  {t.name:12s} {us:9.1f} us   (batch, whole signal / frames; offline only)")

    # --- figure ---
    fig, axes = plt.subplots(2 * len(panels), 1, figsize=(12, 7.5 * len(panels)),
                             gridspec_kw={"height_ratios": [2, 1] * len(panels)},
                             constrained_layout=True)
    for i, (sig, sr, trs, tf, title, err_ref) in enumerate(panels):
        plot_panel(axes[2 * i : 2 * i + 2], sig, sr, cfg, trs, tf, title, err_ref)
    out = ROOT / "results" / "compare_pitch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
