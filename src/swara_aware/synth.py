"""Synthetic test signals whose correct pitch is known exactly.

Audio concepts: sampling (a continuous formula evaluated at t = n / sample_rate),
harmonics and the missing fundamental, instantaneous frequency and phase (for glides),
and noise colour (white vs pink).

These signals are the "ground truth" for the tests and for the evaluation in Stage 5.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def time_axis(duration: float, sample_rate: int) -> np.ndarray:
    """Sample times t[n] = n / sample_rate for ``duration`` seconds (sampling)."""
    n_samples = int(round(duration * sample_rate))
    return np.arange(n_samples) / sample_rate


def sine_tone(
    freq: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    """Pure sine: x[n] = A * sin(2*pi*f*n/fs + phase)."""
    t = time_axis(duration, sample_rate)
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def harmonic_amplitudes(
    n_harmonics: int = 8, decay: float = 0.7, missing_fundamental: bool = False
) -> list[float]:
    """Relative amplitudes decay**(k-1) for harmonics k = 1..n (k=1 zeroed if missing)."""
    amps = [decay ** (k - 1) for k in range(1, n_harmonics + 1)]
    if missing_fundamental:
        amps[0] = 0.0
    return amps


def _harmonic_stack(
    phase: np.ndarray, max_f0: float, sample_rate: int, amps: Sequence[float], amplitude: float
) -> np.ndarray:
    """Sum amps[k-1] * sin(k * phase), skipping harmonics above Nyquist, peak-normalised."""
    x = np.zeros_like(phase)
    for k, a in enumerate(amps, start=1):
        if k * max_f0 >= sample_rate / 2:
            break  # this harmonic would alias
        if a != 0.0:
            x += a * np.sin(k * phase)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    return amplitude * x / peak if peak > 0 else x


def voice_like_tone(
    f0: float,
    duration: float,
    sample_rate: int,
    n_harmonics: int = 8,
    decay: float = 0.7,
    missing_fundamental: bool = False,
    amplitude: float = 0.5,
    amps: Sequence[float] | None = None,
) -> np.ndarray:
    """Harmonic-rich tone: the fundamental plus decaying harmonics (harmonics).

    Harmonic k (k = 1, 2, ...) is at k * f0 with relative amplitude decay**(k-1).
    Harmonics above Nyquist are skipped to avoid aliasing. With
    ``missing_fundamental=True`` the k=1 component is removed; we still *hear* f0, and
    a good pitch detector should still report f0, not 2*f0.

    Args:
        f0: Fundamental frequency in Hz.
        duration: Length in seconds.
        sample_rate: Sampling rate in Hz.
        n_harmonics: Number of harmonics including the fundamental.
        decay: Amplitude ratio between neighbouring harmonics (0 < decay <= 1).
        missing_fundamental: Drop the fundamental component.
        amplitude: Peak amplitude of the returned signal.
        amps: Explicit relative amplitudes for harmonics 1, 2, 3, ... (overrides
            ``n_harmonics``, ``decay`` and ``missing_fundamental``). Useful for a
            *weak* fundamental, e.g. ``[0.2, 1.0, 0.8, 0.5]``.

    Returns:
        The tone, normalised so its peak equals ``amplitude``.
    """
    if amps is None:
        amps = harmonic_amplitudes(n_harmonics, decay, missing_fundamental)
    phase = 2.0 * np.pi * f0 * time_axis(duration, sample_rate)
    return _harmonic_stack(phase, f0, sample_rate, amps, amplitude)


def voice_like_glide(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int,
    n_harmonics: int = 8,
    decay: float = 0.7,
    amplitude: float = 0.5,
    exponential: bool = True,
) -> np.ndarray:
    """Harmonic-rich glide (a sung meend): every harmonic follows k * f(t).

    The fundamental's phase is the running sum of its instantaneous frequency, and
    harmonic k uses k times that phase. True pitch: :func:`glide_frequencies`.
    """
    f = glide_frequencies(f_start, f_end, duration, sample_rate, exponential)
    phase = 2.0 * np.pi * np.cumsum(f) / sample_rate
    amps = harmonic_amplitudes(n_harmonics, decay)
    return _harmonic_stack(phase, max(f_start, f_end), sample_rate, amps, amplitude)


def glide(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
    exponential: bool = True,
) -> np.ndarray:
    """Tone sliding from ``f_start`` to ``f_end``, like a meend (instantaneous frequency).

    The phase is the running sum of the instantaneous frequency:
    phase[n] = 2*pi * sum(f[0..n]) / fs. Using sin(2*pi*f(t)*t) instead would be
    wrong, because it does not produce the intended frequency at each moment.

    With ``exponential=True`` the frequency changes by equal *cents* per second (how
    pitch is heard); otherwise it changes by equal Hz per second.

    Returns:
        The glide signal. Use :func:`glide_frequencies` for its true pitch curve.
    """
    f = glide_frequencies(f_start, f_end, duration, sample_rate, exponential)
    phase = 2.0 * np.pi * np.cumsum(f) / sample_rate
    return amplitude * np.sin(phase)


def glide_frequencies(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int,
    exponential: bool = True,
) -> np.ndarray:
    """Instantaneous frequency (Hz) of :func:`glide` at every sample (ground truth)."""
    n = len(time_axis(duration, sample_rate))
    ramp = np.linspace(0.0, 1.0, n)
    if exponential:
        return f_start * (f_end / f_start) ** ramp
    return f_start + (f_end - f_start) * ramp


def voice_from_contour(
    f0: np.ndarray,
    sample_rate: int,
    n_harmonics: int = 8,
    decay: float = 0.7,
    amplitude: float = 0.4,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Voice-like tone that follows any pitch curve, given per sample (NaN = silence).

    Phase is the running sum of the instantaneous frequency, so the curve can contain
    held notes, glides and gamakas without clicks. Voiced/silent boundaries get a short
    linear fade so they do not click either.
    """
    f0 = np.asarray(f0, dtype=float)
    voiced = ~np.isnan(f0)
    f_filled = np.where(voiced, f0, 0.0)
    phase = 2.0 * np.pi * np.cumsum(f_filled) / sample_rate
    max_f0 = float(np.nanmax(f0)) if voiced.any() else 1.0
    x = _harmonic_stack(phase, max_f0, sample_rate,
                        harmonic_amplitudes(n_harmonics, decay), amplitude)
    # Smooth on/off envelope: a moving average of the voiced mask.
    n_fade = max(1, int(fade_ms * sample_rate / 1000))
    env = np.convolve(voiced.astype(float), np.ones(n_fade) / n_fade, mode="same")
    return x * env * voiced


def white_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gaussian white noise (equal power at every frequency), with RMS ``amplitude``."""
    rng = rng if rng is not None else np.random.default_rng()
    n = len(time_axis(duration, sample_rate))
    return amplitude * rng.standard_normal(n)


def pink_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pink (1/f) noise with RMS ``amplitude`` (noise colour).

    Pink noise has equal power per octave, closer to real room noise than white noise.
    Method: take white noise to the frequency domain with the FFT, multiply bin k by
    1/sqrt(f_k) (power then falls as 1/f), and transform back.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = len(time_axis(duration, sample_rate))
    if n == 0:
        return np.zeros(0)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])  # bin 0 (DC) is set to zero
    x = np.fft.irfft(spectrum * scale, n=n)
    x_rms = np.sqrt(np.mean(x**2))
    return amplitude * x / x_rms if x_rms > 0 else x
