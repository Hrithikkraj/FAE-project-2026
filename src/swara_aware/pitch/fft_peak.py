"""Pitch from the FFT magnitude spectrum: peak picking, optionally with HPS.

Audio concepts: windowing, zero-padding, FFT bins, harmonics, parabolic interpolation
on the log spectrum, and the Harmonic Product Spectrum (HPS).

Why plain peak picking fails: a sung note is a stack of harmonics at f0, 2f0, 3f0, ...
and the *loudest* one is often not the fundamental (vowel formants boost harmonics
2-4, and small laptop mics roll off low frequencies). Plain peak picking then reports
2f0 or 3f0: an octave (or twelfth) error. If the fundamental is missing entirely, it
can never be found this way. HPS multiplies the spectrum by copies of itself
compressed by 2, 3, ... so that only f0, where all harmonics line up, stays large.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..dsp import hann_window
from .base import PitchEstimator, PitchResult, parabolic_interpolation

_EPS = 1e-12


class FFTPeakEstimator(PitchEstimator):
    """Strongest spectral peak in [fmin, fmax], or the HPS peak if ``use_hps``."""

    def __init__(
        self,
        sample_rate: int,
        fmin: float,
        fmax: float,
        use_hps: bool = False,
        hps_harmonics: int = Config.hps_harmonics,
        zero_pad_factor: int = Config.fft_zero_pad_factor,
        voicing_threshold: float = Config.fft_voicing_threshold,
        silence_threshold_dbfs: float | None = Config.silence_threshold_dbfs,
    ) -> None:
        super().__init__(sample_rate, fmin, fmax, silence_threshold_dbfs)
        self.use_hps = use_hps
        self.hps_harmonics = hps_harmonics
        self.zero_pad_factor = zero_pad_factor
        self.voicing_threshold = voicing_threshold
        self.name = "fft_hps" if use_hps else "fft_peak"

    def _estimate_voiced(self, frame: np.ndarray) -> PitchResult:
        n = len(frame)
        # 1. Window (reduces leakage) and zero-pad. Zero-padding does not add real
        #    resolution, but samples the spectrum more finely, so the peak shape is
        #    smoother and the parabola fit below is more accurate.
        n_fft = int(2 ** np.ceil(np.log2(n * self.zero_pad_factor)))
        mag = np.abs(np.fft.rfft((frame - frame.mean()) * hann_window(n), n_fft))
        bin_hz = self.sample_rate / n_fft

        # 2. Work in log magnitude (dB-like). A Hann main lobe is close to a Gaussian,
        #    and the log of a Gaussian is exactly a parabola, so interpolation is accurate.
        log_mag = np.log(mag + _EPS)
        if self.use_hps:
            # HPS in the log domain: log S(k) + log S(2k) + ... + log S(Hk).
            # Bins beyond n_fft/(2H) have no H-th harmonic, so the result is shorter.
            length = len(log_mag) // self.hps_harmonics
            score = np.zeros(length)
            for h in range(1, self.hps_harmonics + 1):
                score += log_mag[::h][:length]
        else:
            score = log_mag

        # 3. Strongest peak inside the allowed f0 range.
        lo = int(np.ceil(self.fmin / bin_hz))
        hi = min(int(np.floor(self.fmax / bin_hz)), len(score) - 2)
        if hi <= lo:
            return PitchResult(None, 0.0)
        k = lo + int(np.argmax(score[lo : hi + 1]))

        # 4. Parabolic interpolation -> fractional bin -> Hz.
        offset, _ = parabolic_interpolation(score, k)
        f0 = (k + offset) * bin_hz

        # 5. Confidence = peak prominence over the band: 1 - median/peak magnitude.
        #    A clear harmonic peak gives ~1; for white noise the biggest bin is only
        #    ~3x the median, giving ~0.7, below the voicing threshold.
        band = mag[lo : hi + 1]
        confidence = float(1.0 - np.median(band) / (mag[k] + _EPS))
        if self.use_hps:  # the HPS peak may sit on a weak f0 bin; use the band max
            confidence = float(1.0 - np.median(band) / (band.max() + _EPS))
        debug = {"bin": k, "offset": offset, "bin_hz": bin_hz}
        if confidence < self.voicing_threshold:
            return PitchResult(None, confidence, debug)
        return PitchResult(f0, confidence, debug)
