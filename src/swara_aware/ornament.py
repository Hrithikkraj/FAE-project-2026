"""Ornament-aware labelling of a pitch track: silent, stable, glide or ornament.

Audio concepts: pitch contour in cents, median filtering (removes single-frame
outliers), local slope in cents per second (how fast the pitch moves), and local range
(how widely it swings). In Hindustani music a meend (slide) or gamaka (oscillation) is
deliberate, so only held ("stable") notes are judged for intonation.

Rules, applied to each voiced frame using only the past (so it runs live). Over the
last ``ornament_window_ms`` the pitch *oscillates* if it travels back at least
``ornament_turn_fraction`` of its range (see :func:`back_travel`) with low net drift.
  1. ornament: oscillating with range >= ``ornament_min_range_cents`` (e.g. a gamaka).
     Checked first: the slope inside a gamaka is steep and would look like a glide.
  2. stable: over ``slope_window_ms`` the range is <= ``stable_max_range_cents`` and the
     net drift is <= ``glide_threshold_cents_per_s``; OR oscillating with a range below
     the ornament threshold (vibrato: judged by its mean pitch).
  3. glide: everything else, i.e. the pitch is moving between notes (a meend). The
     reported slope is the fitted slope over ``slope_window_ms``.
A stable run only becomes a judged *held note* once it lasts ``stable_min_ms``;
``FrameState.held_ms`` reports how long the current stable run has lasted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import Config
from .swara import freq_to_cents

STATES = ("silent", "stable", "glide", "ornament")


@dataclass(frozen=True)
class FrameState:
    """Label for one analysis frame.

    Attributes:
        time: Frame time in seconds.
        f0_hz: Raw pitch from the estimator (NaN if unvoiced).
        cents: Median-filtered pitch in cents relative to Sa (NaN if silent).
        state: "silent", "stable", "glide" or "ornament".
        slope: Local slope in cents per second (NaN if not enough frames).
        range_cents: Pitch range over the ornament window (NaN if silent).
        held_ms: How long the current stable run has lasted (0 if not stable).
    """

    time: float
    f0_hz: float
    cents: float
    state: str
    slope: float
    range_cents: float
    held_ms: float


def fitted_slope(times: np.ndarray, cents: np.ndarray) -> float:
    """Least-squares slope (cents per second) of a straight line through the points.

    A fitted line is less sensitive to jitter than (last - first) / duration.
    """
    t = times - times.mean()
    denom = float(np.sum(t * t))
    return float(np.sum(t * (cents - cents.mean())) / denom) if denom > 0 else 0.0


def back_travel(cents: np.ndarray) -> float:
    """How far the pitch travelled *backwards*: (total path - net change) / 2.

    A monotonic glide has back-travel ~0 (path = net change). One full gamaka cycle of
    range R travels 2R in total but ends near where it started, so back-travel >= R/2.
    """
    total_path = float(np.sum(np.abs(np.diff(cents))))
    return (total_path - abs(float(cents[-1] - cents[0]))) / 2.0


class OrnamentTracker:
    """Frame-by-frame state machine; feed it (time, f0, confidence) one frame at a time.

    Use :meth:`update` for live audio and :meth:`process` for a whole array; both give
    identical results because :meth:`process` simply calls :meth:`update` in a loop.
    """

    def __init__(self, cfg: Config, sa_hz: float) -> None:
        """Set up empty history buffers for a singer whose Sa is ``sa_hz``."""
        self.cfg = cfg
        self.sa_hz = sa_hz
        self.reset()

    def reset(self) -> None:
        """Forget all history (call when the singer stops or Sa changes)."""
        self._raw: deque[float] = deque(maxlen=self.cfg.median_filter_frames)
        self._history: deque[tuple[float, float]] = deque()  # (time, smoothed cents)
        self._stable_since: float | None = None

    def update(self, time: float, f0_hz: float | None, confidence: float) -> FrameState:
        """Label one frame. ``f0_hz`` None/NaN or low confidence means unvoiced."""
        cfg = self.cfg
        f0 = np.nan if f0_hz is None else float(f0_hz)
        if np.isnan(f0) or f0 <= 0 or confidence < cfg.min_confidence:
            self.reset()  # a note boundary: windows must not mix two notes
            return FrameState(time, f0, np.nan, "silent", np.nan, np.nan, 0.0)

        # 1. Causal median filter over the last few frames removes one-frame outliers
        #    (e.g. an octave jump), at the cost of ~2 frames of delay.
        self._raw.append(float(freq_to_cents(f0, self.sa_hz)))
        cents = float(np.median(self._raw))

        # 2. Keep (time, cents) for the longer ornament window; drop older frames.
        self._history.append((time, cents))
        while self._history[0][0] < time - cfg.ornament_window_ms / 1000.0:
            self._history.popleft()
        hist = np.array(self._history)
        t_all, c_all = hist[:, 0], hist[:, 1]
        short = t_all >= time - cfg.slope_window_ms / 1000.0
        t_short, c_short = t_all[short], c_all[short]

        long_range = float(c_all.max() - c_all.min())
        if len(t_short) < 3:  # note onset: not enough frames to measure motion yet
            return self._label(time, f0, cents, "glide", np.nan, long_range)

        slope = fitted_slope(t_short, c_short)
        short_range = float(c_short.max() - c_short.min())
        drift = abs(fitted_slope(t_all, c_all))
        oscillating = (long_range > 0
                       and back_travel(c_all) >= cfg.ornament_turn_fraction * long_range
                       and drift < cfg.ornament_max_drift_cents_per_s)

        # 3. Apply the rules in order (see module docstring).
        if oscillating and long_range >= cfg.ornament_min_range_cents:
            state = "ornament"
        elif ((short_range <= cfg.stable_max_range_cents
               and abs(slope) <= cfg.glide_threshold_cents_per_s)
              or oscillating):  # small oscillation = vibrato, judged on its mean
            state = "stable"
        else:
            state = "glide"
        return self._label(time, f0, cents, state, slope, long_range)

    def _label(self, time: float, f0: float, cents: float, state: str, slope: float,
               range_cents: float) -> FrameState:
        """Build the FrameState and track how long the current stable run has lasted."""
        if state == "stable":
            if self._stable_since is None:
                self._stable_since = time
            held_ms = 1000.0 * (time - self._stable_since) + self.cfg.hop_duration_ms
        else:
            self._stable_since = None
            held_ms = 0.0
        return FrameState(time, f0, cents, state, slope, range_cents, held_ms)

    def process(self, times: np.ndarray, f0: np.ndarray,
                confidence: np.ndarray | None = None) -> list[FrameState]:
        """Label a whole track (for tests and offline analysis). Resets first."""
        self.reset()
        if confidence is None:
            confidence = np.ones(len(times))
        return [self.update(float(t), float(f), float(c))
                for t, f, c in zip(times, f0, confidence)]
