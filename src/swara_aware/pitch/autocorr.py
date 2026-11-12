"""Pitch by normalized autocorrelation, computed with the FFT.

Audio concepts: autocorrelation (similarity of a signal with a delayed copy of
itself), periodicity, the Wiener-Khinchin theorem, and parabolic interpolation.

A periodic signal lines up with itself when delayed by one period T, so the
autocorrelation r(tau) has a peak at tau = T (and at 2T, 3T, ...). f0 = fs / T.

Wiener-Khinchin: the autocorrelation is the inverse FFT of the power spectrum |X|^2.
That costs O(N log N) instead of O(N^2) for the direct double loop.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .base import PitchEstimator, PitchResult, parabolic_interpolation

_EPS = 1e-12


def normalized_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation for lags 0..max_lag (values in about -1..1).

    r(tau) = sum_j x[j] x[j+tau] is divided by sqrt(E_head(tau) * E_tail(tau)), the
    energies of the two overlapping parts. This removes the natural fall-off of r(tau)
    with lag (fewer samples overlap), so a perfectly periodic signal scores 1 at every
    multiple of its period, and the value is a correlation coefficient we can threshold.
    """
    n = len(x)
    # Zero-pad to >= 2n so the FFT's circular correlation does not wrap around.
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(x, n_fft)
    r = np.fft.irfft(np.abs(spectrum) ** 2, n_fft)[: max_lag + 1]  # Wiener-Khinchin

    energy = np.concatenate([[0.0], np.cumsum(x**2)])  # energy[m] = sum of x[:m]**2
    lags = np.arange(max_lag + 1)
    e_head = energy[n - lags]                 # sum of x[0 : n-tau]**2
    e_tail = energy[n] - energy[lags]         # sum of x[tau : n]**2
    return r / (np.sqrt(e_head * e_tail) + _EPS)


def lobe_peaks(nacf: np.ndarray) -> list[int]:
    """Index of the maximum of every positive lobe (run of nacf > 0), except lag 0's.

    Taking one peak per lobe, rather than every tiny local maximum, matters with
    noise: noise adds small ripples on the shoulder of a broad peak (low notes), and a
    "first local maximum" rule would stop on a ripple tens of cents before the period.
    """
    positive = np.concatenate([[False], nacf > 0, [False]]).astype(int)
    starts = np.nonzero(np.diff(positive) == 1)[0]  # first index of each run
    ends = np.nonzero(np.diff(positive) == -1)[0]   # one past the last index
    return [int(s + np.argmax(nacf[s:e])) for s, e in zip(starts, ends) if s > 0]


class AutocorrEstimator(PitchEstimator):
    """Period = first strong peak of the normalized autocorrelation."""

    name = "autocorr"

    def __init__(
        self,
        sample_rate: int,
        fmin: float,
        fmax: float,
        voicing_threshold: float = Config.acf_voicing_threshold,
        peak_ratio: float = Config.acf_peak_ratio,
        silence_threshold_dbfs: float | None = Config.silence_threshold_dbfs,
    ) -> None:
        super().__init__(sample_rate, fmin, fmax, silence_threshold_dbfs)
        self.voicing_threshold = voicing_threshold
        self.peak_ratio = peak_ratio
        # Allowed periods in samples: 1/fmax .. 1/fmin seconds.
        self.min_lag = max(1, int(np.floor(sample_rate / fmax)))
        self.max_lag = int(np.ceil(sample_rate / fmin))

    def _estimate_voiced(self, frame: np.ndarray) -> PitchResult:
        if len(frame) <= 2 * self.max_lag:
            raise ValueError("frame too short: need more than two periods of fmin")
        x = frame - frame.mean()  # remove DC, which would add a constant to r(tau)
        nacf = normalized_autocorrelation(x, self.max_lag + 1)

        # Candidate peaks: the highest point of each positive lobe, inside the lag range.
        peaks = np.array([p for p in lobe_peaks(nacf) if self.min_lag <= p <= self.max_lag])
        if len(peaks) == 0:
            return PitchResult(None, 0.0)

        # Pick the FIRST peak that is nearly as high as the best one. The peaks at 2T,
        # 3T are about as high as at T; taking the highest could pick 2T (octave low).
        best = float(nacf[peaks].max())
        chosen = int(peaks[np.argmax(nacf[peaks] >= self.peak_ratio * best)])

        # Parabolic interpolation: the true period is rarely a whole number of samples
        # (at 900 Hz, T = 49.0 samples, but 1 sample of error = 35 cents).
        offset, height = parabolic_interpolation(nacf, chosen)
        period = chosen + offset
        confidence = float(np.clip(height, 0.0, 1.0))  # peak height = periodicity
        debug = {"lag": period}
        if confidence < self.voicing_threshold:
            return PitchResult(None, confidence, debug)
        return PitchResult(self.sample_rate / period, confidence, debug)
