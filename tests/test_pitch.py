"""Tests for the Stage 2 pitch estimators, on synthetic signals with known f0.

Concepts checked: pitch accuracy in cents, octave errors, the missing fundamental,
voicing decisions on silence and noise, and robustness to noise at a given SNR.

Measured numbers are printed in the "measurement report" at the end of the pytest
output. Known failures are reported, not hidden; only behaviour we rely on is asserted.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from swara_aware import dsp, synth
from swara_aware.config import Config
from swara_aware.pitch import PitchEstimator, cents_error, estimate_track, make_estimators

CFG = Config()
SR, N = CFG.sample_rate, CFG.frame_size
ESTIMATORS = {e.name: e for e in make_estimators(CFG)}
NAMES = list(ESTIMATORS)  # fft_peak, fft_hps, autocorr, yin
SINE_FREQS = [80, 100, 150, 220, 330, 440, 660, 900]
WEAK_F0_AMPS = [0.2, 1.0, 0.8, 0.6, 0.4, 0.3]  # 2nd harmonic 5x stronger than f0
Report = Callable[[str], None]


def one_frame(x: np.ndarray) -> np.ndarray:
    """A full frame taken from the steady middle of a test signal."""
    return x[1000 : 1000 + N]


def error_cents(est: PitchEstimator, frame: np.ndarray, f_true: float) -> float:
    """Cents error of one estimate; NaN if the estimator said unvoiced."""
    f0 = est.estimate(frame).f0_hz
    return float(cents_error(f0, f_true)) if f0 is not None else float("nan")


def fmt(e: float) -> str:
    return "  unvoiced" if np.isnan(e) else f"{e:+10.2f}"


def table(report: Report, title: str, freqs: list[float], errs: dict[str, list[float]]) -> None:
    """Add a cents-error table (rows = frequencies, columns = methods) to the report."""
    report(f"\n{title}  (error in cents)")
    report("  f0 (Hz) " + "".join(f"{n:>10s}" for n in errs))
    for i, f in enumerate(freqs):
        report(f"  {f:7.0f} " + "".join(fmt(errs[n][i]) for n in errs))


# ---------- 1. pure sine tones ----------

def test_sine_tones_within_5_cents(report: Report) -> None:
    errs = {n: [error_cents(ESTIMATORS[n], one_frame(synth.sine_tone(f, 0.2, SR)), f)
                for f in SINE_FREQS] for n in NAMES}
    table(report, "Sine tones", SINE_FREQS, errs)
    report("  NOTE: fft_hps fails on pure sines by design (no harmonics to multiply).")
    for name in ["fft_peak", "autocorr", "yin"]:
        worst = np.nanmax(np.abs(errs[name]))
        assert not np.any(np.isnan(errs[name])), f"{name} said unvoiced on a clean sine"
        assert worst < 5.0, f"{name}: worst error {worst:.2f} cents"


# ---------- 2. harmonic (voice-like) tones ----------

def test_harmonic_tones(report: Report) -> None:
    errs = {n: [error_cents(ESTIMATORS[n], one_frame(synth.voice_like_tone(f, 0.2, SR)), f)
                for f in SINE_FREQS] for n in NAMES}
    table(report, "Harmonic tones (decay 0.7, 8 harmonics)", SINE_FREQS, errs)
    for name in ["autocorr", "yin"]:
        worst = np.nanmax(np.abs(errs[name]))
        assert not np.any(np.isnan(errs[name])) and worst < 10.0, f"{name}: {worst:.2f} cents"


def test_weak_fundamental_hps_fixes_octave_errors(report: Report) -> None:
    freqs = [110, 150, 220, 330]
    errs = {n: [error_cents(ESTIMATORS[n],
                            one_frame(synth.voice_like_tone(f, 0.2, SR, amps=WEAK_F0_AMPS)), f)
                for f in freqs] for n in NAMES}
    table(report, f"Weak fundamental, harmonic amps {WEAK_F0_AMPS}", freqs, errs)
    octave = {n: int(np.sum(np.abs(np.abs(np.array(errs[n])) - 1200) < 50)) for n in NAMES}
    report(f"  octave errors: {octave}")
    # Plain peak picking jumps to the loudest harmonic (2 f0) ...
    assert octave["fft_peak"] == len(freqs)
    # ... HPS lines the harmonics up and recovers f0.
    assert octave["fft_hps"] == 0 and np.nanmax(np.abs(errs["fft_hps"])) < 10.0
    for name in ["autocorr", "yin"]:
        assert np.nanmax(np.abs(errs[name])) < 10.0


# ---------- 3. missing fundamental ----------

def test_missing_fundamental(report: Report) -> None:
    freqs = [110, 150, 220, 330, 440]
    errs = {n: [error_cents(ESTIMATORS[n], one_frame(
        synth.voice_like_tone(f, 0.2, SR, missing_fundamental=True)), f) for f in freqs]
        for n in NAMES}
    table(report, "Missing fundamental (harmonics 2-8 only)", freqs, errs)
    failed = [n for n in NAMES if np.any(np.isnan(errs[n])) or np.nanmax(np.abs(errs[n])) > 50]
    report(f"  methods that FAIL (report 2*f0 or unvoiced): {failed}")
    # Time-domain methods find the repetition period, which is still 1/f0.
    for name in ["autocorr", "yin"]:
        assert name not in failed


# ---------- 4. silence and noise must be unvoiced ----------

@pytest.mark.parametrize("label,frame", [
    ("digital silence", np.zeros(N)),
    ("white noise -60 dBFS", 0.001 * np.random.default_rng(1).standard_normal(N)),
    ("pink noise -65 dBFS", synth.pink_noise(N / SR, SR, 10 ** (-65 / 20),
                                             np.random.default_rng(2))),
])
def test_silence_and_quiet_noise_are_unvoiced(label: str, frame: np.ndarray) -> None:
    for name, est in ESTIMATORS.items():
        result = est.estimate(frame)
        assert result.f0_hz is None, f"{name} reported {result.f0_hz} Hz for {label}"


def test_loud_noise_without_gate(report: Report) -> None:
    """Switch the silence gate off: does each algorithm itself reject noise?"""
    rng = np.random.default_rng(3)
    trials = 50
    report(f"\nLoud noise (-20 dBFS RMS), silence gate OFF: % frames wrongly voiced "
           f"({trials} frames)")
    ungated_estimators = make_estimators(CFG)  # fresh copies, then switch the gate off
    for est in ungated_estimators:
        est.silence_threshold_dbfs = None
    for kind in ["white", "pink"]:
        voiced = {}
        for ungated in ungated_estimators:
            name = ungated.name
            hits = 0
            for _ in range(trials):
                if kind == "white":
                    x = 0.1 * rng.standard_normal(N)
                else:
                    x = synth.pink_noise(N / SR, SR, 0.1, rng)
                hits += ungated.estimate(x).f0_hz is not None
            voiced[name] = 100.0 * hits / trials
        report(f"  {kind:5s}: " + "  ".join(f"{n} {v:5.1f}%" for n, v in voiced.items()))
        assert voiced["autocorr"] <= 5.0 and voiced["yin"] <= 5.0


# ---------- 5. noisy tones ----------

def noisy_stats(est: PitchEstimator, f: float, snr: float, kind: str, trials: int,
                rng: np.random.Generator) -> tuple[float, float, float]:
    """(voiced %, median |error|, max |error|) over ``trials`` noisy frames."""
    errs = []
    for _ in range(trials):
        if kind == "sine":
            clean = synth.sine_tone(f, N / SR, SR, phase=rng.uniform(0, 2 * np.pi))
        else:
            clean = synth.voice_like_tone(f, N / SR, SR)
        errs.append(error_cents(est, dsp.add_noise(clean, snr, rng=rng), f))
    e = np.abs(np.array(errs))
    ok = ~np.isnan(e)
    if not ok.any():
        return 0.0, float("nan"), float("nan")
    return 100.0 * ok.mean(), float(np.median(e[ok])), float(np.max(e[ok]))


@pytest.mark.parametrize("snr", [20.0, 10.0])
@pytest.mark.parametrize("kind", ["harmonic", "sine"])
def test_noisy_tones(report: Report, snr: float, kind: str) -> None:
    rng = np.random.default_rng(int(snr) + len(kind))
    freqs, trials = [100, 220, 440], 30
    report(f"\n{kind.capitalize()} tones + white noise at {snr:.0f} dB SNR "
           f"({trials} frames each): voiced% / median|err| / max|err| (cents)")
    report("  f0 (Hz) " + "".join(f"{n:>22s}" for n in NAMES))
    stats = {}
    for f in freqs:
        row = []
        for name in NAMES:
            v, med, mx = noisy_stats(ESTIMATORS[name], f, snr, kind, trials, rng)
            stats[(name, f)] = (v, med, mx)
            row.append(f"{v:5.0f}% {med:6.1f} {mx:7.1f}")
        report(f"  {f:7.0f} " + "".join(f"{r:>22s}" for r in row))

    # Assert only what we rely on; everything else is in the report.
    for f in freqs:
        for name in ["autocorr", "yin"]:
            voiced, med, _ = stats[(name, f)]
            assert voiced >= 90.0, f"{name} at {f} Hz, {snr} dB: voiced {voiced:.0f}%"
            limit = 5.0 if (snr >= 20 or kind == "harmonic") else 15.0
            assert med < limit, f"{name} at {f} Hz, {snr} dB: median {med:.1f} cents"


# ---------- estimate_track and the reference wrapper ----------

def test_estimate_track_shapes_and_unvoiced_nan() -> None:
    x = np.concatenate([np.zeros(SR // 4), synth.voice_like_tone(220.0, 0.5, SR)])
    times, f0, conf = estimate_track(x, ESTIMATORS["yin"], N, CFG.hop_size)
    assert times.shape == f0.shape == conf.shape
    assert np.isnan(f0[0])  # first frame is pure silence
    voiced = f0[times > 0.25 + N / SR]
    assert np.all(np.abs(cents_error(voiced, 220.0)) < 5.0)


def test_reference_tracker_on_sine() -> None:
    pytest.importorskip("librosa")
    from swara_aware.pitch.reference import reference_track

    x = synth.voice_like_tone(220.0, 0.6, SR)
    times, f0, conf = reference_track(x, SR, CFG.fmin, CFG.fmax, N, CFG.hop_size)
    assert times.shape == f0.shape == conf.shape
    assert np.nanmedian(np.abs(cents_error(f0, 220.0))) < 10.0
