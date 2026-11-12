"""YIN pitch estimator (de Cheveigne and Kawahara, 2002), steps 1-5 of the paper.

Audio concepts: the difference function (how different a signal is from itself delayed
by tau), the cumulative mean normalized difference (CMNDF), absolute thresholding,
and parabolic interpolation.

Autocorrelation looks for a *peak* of similarity; YIN looks for a *dip* of difference
d(tau) = sum_j (x[j] - x[j+tau])^2, which is 0 for a perfect period. Raw d(tau) has
two problems: d(0) = 0 always, and d(tau) at 2T, 3T is as low as at T, so "lowest dip"
causes octave-low errors. The CMNDF fixes both (see step 3).

Known limitation: d(tau) compares x[0:W] with x[tau:tau+W], so the analysis is centred
about (W + T) / 2 samples into the frame, a few ms before the frame centre. On a fast
glide this reads the pitch slightly early (about 2 cents on a 400 cents/s meend).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .base import PitchEstimator, PitchResult, parabolic_interpolation

_EPS = 1e-12


def difference_function(x: np.ndarray, max_lag: int, window: int) -> np.ndarray:
    """Step 2: d(tau) = sum_{j<W} (x[j] - x[j+tau])^2 for tau = 0..max_lag.

    Expanding the square: d(tau) = E(x[0:W]) + E(x[tau:tau+W]) - 2 * r(tau), where E
    is energy and r the cross-correlation of x[0:W] with x. r for all lags comes from
    one FFT product (Wiener-Khinchin), so this is O(N log N), not O(N * max_lag).
    """
    n_fft = int(2 ** np.ceil(np.log2(window + max_lag + window)))
    a = np.fft.rfft(x[:window], n_fft)
    b = np.fft.rfft(x[: window + max_lag], n_fft)
    # Cross-correlation r[tau] = sum_j x[j] * x[j+tau] = IFFT(conj(A) * B).
    r = np.fft.irfft(np.conj(a) * b, n_fft)[: max_lag + 1]

    energy = np.concatenate([[0.0], np.cumsum(x**2)])
    lags = np.arange(max_lag + 1)
    e_ref = energy[window]                          # energy of x[0:W] (constant)
    e_shift = energy[lags + window] - energy[lags]  # energy of x[tau:tau+W]
    return np.maximum(e_ref + e_shift - 2.0 * r, 0.0)  # clip tiny negative rounding


def cmndf(d: np.ndarray) -> np.ndarray:
    """Step 3: cumulative mean normalized difference, d'(tau).

    d'(0) = 1 and d'(tau) = d(tau) / ((1/tau) * sum_{j=1..tau} d(j)): each value is
    divided by the average of all smaller lags. Effects:
      * the trivial dip at tau = 0 disappears (d' starts at 1, no need for fmax limit);
      * d' is ~1 for non-periodic signals and near 0 at a true period, so one fixed
        threshold works for loud and quiet notes alike;
      * because the running mean grows with tau, dips are judged relative to the
        signal's own scale, and we can take the FIRST good dip (the true period T)
        rather than the deepest one, which might be at 2T (octave error).
    """
    out = np.ones_like(d)
    cumulative = np.cumsum(d[1:])
    tau = np.arange(1, len(d))
    out[1:] = d[1:] * tau / (cumulative + _EPS)
    return out


class YinEstimator(PitchEstimator):
    """YIN: first CMNDF dip below an absolute threshold gives the period."""

    name = "yin"

    def __init__(
        self,
        sample_rate: int,
        fmin: float,
        fmax: float,
        threshold: float = Config.yin_threshold,
        silence_threshold_dbfs: float | None = Config.silence_threshold_dbfs,
    ) -> None:
        super().__init__(sample_rate, fmin, fmax, silence_threshold_dbfs)
        self.threshold = threshold
        self.min_lag = max(2, int(np.floor(sample_rate / fmax)))
        self.max_lag = int(np.ceil(sample_rate / fmin))

    def _estimate_voiced(self, frame: np.ndarray) -> PitchResult:
        window = len(frame) - self.max_lag - 1  # integration window W
        if window < self.max_lag:
            raise ValueError("frame too short: need about two periods of fmin")
        x = frame - frame.mean()

        d = difference_function(x, self.max_lag + 1, window)   # step 2
        d_norm = cmndf(d)                                      # step 3

        # Step 4, absolute threshold: the first lag in range whose d' is below the
        # threshold marks the first good dip, i.e. the shortest plausible period.
        search = d_norm[self.min_lag : self.max_lag + 1]
        below = np.nonzero(search < self.threshold)[0]
        if len(below) == 0:  # no dip deep enough: aperiodic (noise) -> unvoiced
            best = self.min_lag + int(np.argmin(search))
            return PitchResult(None, float(np.clip(1 - d_norm[best], 0, 1)), {"lag": best})
        first = self.min_lag + int(below[0])
        # Find the bottom of that dip. The paper slides downhill to the first local
        # minimum, but noise ripples can stop that early on a broad (low-note) dip.
        # We take the minimum of the whole dip: the run of lags where d' stays below 1.
        end = first
        while end + 1 <= self.max_lag and d_norm[end + 1] < 1.0:
            end += 1
        tau = first + int(np.argmin(d_norm[first : end + 1]))

        # Step 5, parabolic interpolation on the raw difference function (as in the
        # paper), giving a fractional period and so a precise f0.
        offset, _ = parabolic_interpolation(d, tau)
        period = tau + offset
        confidence = float(np.clip(1.0 - d_norm[tau], 0.0, 1.0))
        return PitchResult(self.sample_rate / period, confidence, {"lag": period})
