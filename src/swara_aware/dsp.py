"""Basic digital signal processing building blocks, written with numpy only.

Audio concepts: framing (short overlapping windows), windowing (Hann), RMS level and
dBFS, the FFT magnitude spectrum, a silence gate, the spectrogram (short-time Fourier
transform), and signal-to-noise ratio (SNR).

All signals are 1-D float arrays in [-1, 1]. Functions are pure: arrays in, arrays out.
"""

from __future__ import annotations

import numpy as np

# Smallest amplitude used before taking a log, so silence gives a finite dB value
# (1e-10 is -200 dBFS, far below anything audible).
_EPS = 1e-10


def frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Split a signal into overlapping frames (framing).

    Frame ``i`` holds samples ``x[i*hop_size : i*hop_size + frame_size]``. If the
    signal does not fill the last frame exactly, it is zero-padded at the end so that
    no samples are lost.

    Args:
        x: 1-D signal.
        frame_size: Samples per frame.
        hop_size: Samples between the starts of consecutive frames.

    Returns:
        2-D array of shape ``(n_frames, frame_size)``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("frame_signal expects a 1-D signal")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")

    # Number of frames needed so the last frame reaches the end of the signal.
    if len(x) <= frame_size:
        n_frames = 1
    else:
        n_frames = 1 + int(np.ceil((len(x) - frame_size) / hop_size))

    # Zero-pad so every frame is complete.
    needed = (n_frames - 1) * hop_size + frame_size
    padded = np.zeros(needed)
    padded[: len(x)] = x

    # Row i starts at i*hop; column j is the offset inside the frame.
    starts = hop_size * np.arange(n_frames)[:, None]
    offsets = np.arange(frame_size)[None, :]
    return padded[starts + offsets]


def hann_window(n: int) -> np.ndarray:
    """Return a periodic Hann window of length ``n`` (windowing).

    w[k] = 0.5 - 0.5 * cos(2*pi*k / n). It tapers the frame edges to zero, which
    reduces spectral leakage in the FFT. The "periodic" form (dividing by n, not n-1)
    is the usual choice for spectral analysis with overlapping frames.
    """
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def apply_window(frames: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    """Multiply each frame (last axis) by a window, Hann by default."""
    frames = np.asarray(frames, dtype=float)
    if window is None:
        window = hann_window(frames.shape[-1])
    return frames * window


def rms(frames: np.ndarray) -> np.ndarray:
    """Root-mean-square level along the last axis (RMS).

    RMS = sqrt(mean(x^2)). A sine of amplitude 1 has RMS 1/sqrt(2), about 0.707.
    Works on a single frame (returns a scalar) or a 2-D array of frames (one value
    per frame).
    """
    frames = np.asarray(frames, dtype=float)
    return np.sqrt(np.mean(frames**2, axis=-1))


def to_dbfs(level: np.ndarray | float) -> np.ndarray:
    """Convert a linear RMS level to decibels relative to full scale (dBFS).

    dBFS = 20 * log10(level). An RMS of 1.0 (a full-scale square wave) is 0 dBFS, so a
    full-scale sine is about -3 dBFS. Digital silence is clamped to -200 dBFS.
    """
    return 20.0 * np.log10(np.maximum(np.asarray(level, dtype=float), _EPS))


def frame_dbfs(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """RMS level in dBFS for every frame of a signal (framing + RMS)."""
    return to_dbfs(rms(frame_signal(x, frame_size, hop_size)))


def silence_gate(frames: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    """Mark which frames are loud enough to analyse (silence gate).

    Args:
        frames: 2-D array ``(n_frames, frame_size)`` or a single 1-D frame.
        threshold_dbfs: Frames with RMS below this level count as silence.

    Returns:
        Boolean array: True where the frame is *sound*, False where it is silence.
    """
    return to_dbfs(rms(frames)) >= threshold_dbfs


def fft_frequencies(n_fft: int, sample_rate: float) -> np.ndarray:
    """Centre frequency (Hz) of each bin of a real FFT of length ``n_fft``.

    Bin k is at k * sample_rate / n_fft, from 0 Hz up to Nyquist (sample_rate / 2).
    """
    return np.arange(n_fft // 2 + 1) * sample_rate / n_fft


def magnitude_spectrum(
    frame: np.ndarray, sample_rate: float, window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of one frame using the FFT.

    The magnitudes are scaled so a sine of amplitude A appears as a peak of height
    about A: we divide by the sum of the window (the coherent gain) and double every
    bin except DC and Nyquist, because a real signal splits its energy between
    positive and negative frequencies.

    Args:
        frame: 1-D frame of samples.
        sample_rate: Sampling rate in Hz.
        window: Apply a Hann window first (recommended).

    Returns:
        ``(freqs, magnitudes)``, both of length ``len(frame) // 2 + 1``.
    """
    frame = np.asarray(frame, dtype=float)
    n = len(frame)
    w = hann_window(n) if window else np.ones(n)
    spectrum = np.fft.rfft(frame * w)
    mags = np.abs(spectrum) / np.sum(w)
    mags[1:] *= 2.0
    if n % 2 == 0:
        mags[-1] /= 2.0  # the Nyquist bin has no negative-frequency twin
    return fft_frequencies(n, sample_rate), mags


def spectrogram(
    x: np.ndarray, sample_rate: float, frame_size: int, hop_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram from our own framing and FFT (short-time Fourier transform).

    Each column is the windowed magnitude spectrum of one frame, scaled the same way
    as :func:`magnitude_spectrum`.

    Returns:
        ``(times, freqs, S)`` where ``times`` (n_frames,) is the centre time of each
        frame in seconds, ``freqs`` (n_bins,) is in Hz, and ``S`` has shape
        ``(n_bins, n_frames)`` with ``n_bins = frame_size // 2 + 1``.
    """
    frames = frame_signal(x, frame_size, hop_size)
    w = hann_window(frame_size)
    spectra = np.fft.rfft(frames * w, axis=1)  # one FFT per row
    S = np.abs(spectra).T / np.sum(w)
    S[1:] *= 2.0
    if frame_size % 2 == 0:
        S[-1] /= 2.0
    times = (np.arange(frames.shape[0]) * hop_size + frame_size / 2) / sample_rate
    return times, fft_frequencies(frame_size, sample_rate), S


def amplitude_to_db(S: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """Convert linear magnitudes to dB (20*log10), clipped below at ``floor_db``."""
    return np.maximum(20.0 * np.log10(np.maximum(S, _EPS)), floor_db)


def signal_power(x: np.ndarray) -> float:
    """Average power of a signal: mean(x^2), the square of its RMS."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x**2))


def measure_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: 10 * log10(P_signal / P_noise) (SNR)."""
    return 10.0 * np.log10(signal_power(clean) / max(signal_power(noise), _EPS**2))


def add_noise(
    x: np.ndarray,
    snr_db: float,
    noise: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add noise to a signal at a requested signal-to-noise ratio (SNR).

    The noise is rescaled so that 10 * log10(P_signal / P_noise) equals ``snr_db``.
    Power, not amplitude, is used, so the scale factor is
    sqrt(P_signal / (P_noise * 10^(snr_db/10))).

    Args:
        x: Clean signal.
        snr_db: Desired SNR in dB (e.g. 20 = noise is 100x weaker in power).
        noise: Noise to add (same length as ``x``). Gaussian white noise if None.
        rng: Random generator for reproducible white noise.

    Returns:
        The noisy signal ``x + scaled_noise``. It is not clipped, so it may exceed
        [-1, 1] at very low SNR.
    """
    x = np.asarray(x, dtype=float)
    if noise is None:
        rng = rng if rng is not None else np.random.default_rng()
        noise = rng.standard_normal(len(x))
    noise = np.asarray(noise, dtype=float)
    if noise.shape != x.shape:
        raise ValueError("noise must have the same shape as the signal")

    p_signal = signal_power(x)
    p_noise = signal_power(noise)
    if p_noise == 0.0:
        raise ValueError("noise has zero power")
    target_noise_power = p_signal / (10.0 ** (snr_db / 10.0))
    return x + noise * np.sqrt(target_noise_power / p_noise)
