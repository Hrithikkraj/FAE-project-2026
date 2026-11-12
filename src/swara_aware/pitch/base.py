"""Common interface for pitch estimators, plus shared helpers.

Audio concepts: fundamental frequency (f0), voiced vs unvoiced frames, framing a whole
signal into a pitch track, and parabolic interpolation (sub-sample peak refinement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..dsp import frame_signal, rms, to_dbfs


@dataclass
class PitchResult:
    """Output of one pitch estimate.

    Attributes:
        f0_hz: Fundamental frequency in Hz, or None if the frame is unvoiced.
        confidence: How sure the method is that the frame is periodic, 0 to 1.
        debug: Optional method-specific values (e.g. the chosen lag or bin).
    """

    f0_hz: float | None
    confidence: float
    debug: dict[str, Any] = field(default_factory=dict)


class PitchEstimator(ABC):
    """Base class: one frame of audio in, one :class:`PitchResult` out.

    ``estimate`` first applies a silence gate (frames quieter than
    ``silence_threshold_dbfs`` are unvoiced, so we never report a pitch for room hiss),
    then calls the subclass's ``_estimate_voiced``. Pass ``silence_threshold_dbfs=None``
    to switch the gate off and test the algorithm's own voicing decision.
    """

    name: str = "base"

    def __init__(
        self,
        sample_rate: int,
        fmin: float,
        fmax: float,
        silence_threshold_dbfs: float | None = Config.silence_threshold_dbfs,
    ) -> None:
        """Store the search range. ``fmin``/``fmax`` bound the f0 search in Hz."""
        if not 0 < fmin < fmax < sample_rate / 2:
            raise ValueError("need 0 < fmin < fmax < sample_rate / 2")
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.silence_threshold_dbfs = silence_threshold_dbfs

    def estimate(self, frame: np.ndarray) -> PitchResult:
        """Estimate the pitch of one frame (silence gate, then the method itself)."""
        frame = np.asarray(frame, dtype=float)
        # Digital silence (or a constant) has no period at all. Without this check,
        # 0/0 in the normalisations can look like a "perfect" match.
        if np.ptp(frame) < 1e-9:
            return PitchResult(None, 0.0, {"gated": True, "reason": "no signal"})
        if self.silence_threshold_dbfs is not None:
            level = float(to_dbfs(rms(frame)))
            if level < self.silence_threshold_dbfs:
                return PitchResult(None, 0.0, {"gated": True, "level_dbfs": level})
        return self._estimate_voiced(frame)

    @abstractmethod
    def _estimate_voiced(self, frame: np.ndarray) -> PitchResult:
        """Method-specific estimate for a frame that passed the silence gate."""


def parabolic_interpolation(y: np.ndarray, i: int) -> tuple[float, float]:
    """Refine a peak (or dip) at index ``i`` to sub-sample precision.

    A sampled peak almost never falls exactly on a sample/bin. Fitting a parabola
    through the three points (i-1, i, i+1) and taking its vertex gives the true peak
    position far more precisely than the sample grid. E.g. one FFT bin of 21.5 Hz is
    ~170 cents at 220 Hz; without interpolation we could never reach 5-cent accuracy.

    Returns:
        ``(offset, value)``: vertex position relative to ``i`` (in -0.5..0.5 for a true
        local extremum) and the parabola's value there.

    The offset is clamped to -0.5..0.5. If ``i`` is not a true extremum (e.g. the best
    value sits at the edge of the search range), the unclamped vertex can land many
    samples away and even give a negative frequency.
    """
    if i <= 0 or i >= len(y) - 1:
        return 0.0, float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return 0.0, b
    offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return offset, b - 0.25 * (a - c) * offset


def estimate_track(
    signal: np.ndarray, estimator: PitchEstimator, frame_size: int, hop_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an estimator over a whole signal, frame by frame (framing).

    Returns:
        ``(times, f0, confidence)``: frame-centre times in seconds, f0 in Hz with NaN
        for unvoiced frames, and confidence values.
    """
    frames = frame_signal(signal, frame_size, hop_size)
    f0 = np.full(len(frames), np.nan)
    conf = np.zeros(len(frames))
    for i, frame in enumerate(frames):
        result = estimator.estimate(frame)
        conf[i] = result.confidence
        if result.f0_hz is not None:
            f0[i] = result.f0_hz
    times = (np.arange(len(frames)) * hop_size + frame_size / 2) / estimator.sample_rate
    return times, f0, conf


def cents_error(estimate_hz: np.ndarray | float, true_hz: np.ndarray | float) -> np.ndarray:
    """Pitch error in cents: 1200 * log2(estimate / truth). NaN where either is NaN."""
    return 1200.0 * np.log2(np.asarray(estimate_hz, float) / np.asarray(true_hz, float))
