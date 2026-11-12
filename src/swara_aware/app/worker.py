"""Processing worker: audio blocks in, small per-frame result objects out.

Audio concepts: a rolling analysis buffer (each new hop of samples slides the
frame_size window forward, i.e. overlapping frames in real time), the silence gate,
pitch estimation, swara mapping and ornament-aware labelling, and processing time
against the real-time budget (one hop).

Threads: the audio source (microphone callback or file thread) only puts blocks into a
queue. This worker thread does all the computation and puts results into another
queue. The UI only reads that queue and draws. No Qt here, so it is testable headless.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import Config
from ..ornament import OrnamentTracker
from ..pitch import PitchEstimator, make_estimators
from ..session import Session
from ..swara import cents_to_freq, estimate_sa, nearest_swara


class BlockSource(Protocol):
    """Anything with ``read(timeout)`` returning a block of samples or None."""

    def read(self, timeout: float | None = None) -> np.ndarray | None: ...


@dataclass(frozen=True)
class FrameResult:
    """Everything the UI needs to draw one frame (small and immutable)."""

    time: float            # seconds since listening started (frame centre)
    f0_hz: float           # raw pitch, NaN if unvoiced
    confidence: float
    cents: float           # smoothed cents from Sa, NaN if silent
    state: str             # silent / stable / glide / ornament
    swara: str             # nearest swara label, "" if silent
    target_cents: float    # that swara's position in cents from Sa (NaN if silent)
    cents_error: float     # signed error vs that swara (NaN if silent)
    status: str            # in_tune / close / sharp / flat / gliding / ornament / silent
    held_ms: float         # how long the current stable run has lasted
    proc_ms: float         # processing time for this frame
    estimator: str         # estimator name used


@dataclass(frozen=True)
class SaEstimate:
    """Result of "Find my Sa": the estimated Sa in Hz, or None if no steady hum."""

    sa_hz: float | None
    n_voiced: int


@dataclass(frozen=True)
class EndOfStream:
    """The source (e.g. a WAV file) has no more audio."""


class ProcessingWorker(threading.Thread):
    """Background thread running the full analysis pipeline frame by frame."""

    def __init__(self, source: BlockSource, cfg: Config, sa_hz: float,
                 estimator: str = "yin", mode: str | None = None,
                 out_maxsize: int = 4000) -> None:
        super().__init__(name="ProcessingWorker", daemon=True)
        self.source = source
        self.cfg = cfg
        self.sa_hz = sa_hz
        self.mode = mode or cfg.tuning_mode
        self.results: queue.Queue[Any] = queue.Queue(maxsize=out_maxsize)
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._estimators = {e.name: e for e in make_estimators(cfg)}
        self.estimator: PitchEstimator = self._estimators[estimator]
        self.tracker = OrnamentTracker(cfg, sa_hz)
        self.session = Session(cfg, sa_hz, self.mode)
        self._buffer = np.zeros(cfg.frame_size)
        self._pending = np.zeros(0)
        self._samples_in = 0
        self._recording: list[np.ndarray] | None = None
        self._find_sa: dict[str, Any] | None = None
        self.dropped_results = 0

    # ---------- thread-safe controls (called from the UI thread) ----------

    def set_estimator(self, name: str) -> None:
        """Switch pitch method live."""
        self._commands.put(("estimator", name))

    def set_sa(self, sa_hz: float) -> None:
        """Change Sa; this starts a new session (cents are relative to Sa)."""
        self._commands.put(("sa", float(sa_hz)))

    def set_mode(self, mode: str) -> None:
        """Change tuning mode ("equal" or "just")."""
        self._commands.put(("mode", mode))

    def find_sa(self, seconds: float = 3.0) -> None:
        """Collect pitch for ``seconds`` of steady humming, then emit an SaEstimate."""
        self._commands.put(("find_sa", float(seconds)))

    def new_session(self) -> None:
        """Discard the current session statistics and start fresh."""
        self._commands.put(("new_session", None))

    def start_recording(self) -> None:
        """Start keeping a copy of the raw audio (for saving a WAV)."""
        self._commands.put(("record", True))

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the audio captured so far (thread-safe enough:
        the list is only appended to by the worker)."""
        blocks = self._recording or []
        self._commands.put(("record", False))
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def snapshot_session(self) -> Session:
        """A copy of the current session that the UI can analyse safely."""
        snap = Session(self.cfg, self.session.sa_hz, self.session.mode)
        snap.frames = list(self.session.frames)
        return snap

    def stop(self) -> None:
        """Ask the thread to finish (returns immediately; use join() to wait)."""
        self._stop_event.set()

    # ---------- the worker loop ----------

    def run(self) -> None:
        """Read blocks, process every complete hop, until stopped."""
        hop = self.cfg.hop_size
        finished = getattr(self.source, "finished", None)
        while not self._stop_event.is_set():
            self._apply_commands()
            block = self.source.read(timeout=0.05)
            if block is None:
                if finished is not None and finished.is_set():
                    self._emit(EndOfStream())
                    break
                continue
            if self._recording is not None:
                self._recording.append(block)
            self._pending = np.concatenate([self._pending, block])
            while len(self._pending) >= hop:  # blocks may not be exactly one hop
                self._process_hop(self._pending[:hop])
                self._pending = self._pending[hop:]

    def _apply_commands(self) -> None:
        """Apply settings changes sent by the UI (only this thread touches state)."""
        while True:
            try:
                cmd, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if cmd == "estimator" and value in self._estimators:
                self.estimator = self._estimators[value]
            elif cmd == "sa":
                self.sa_hz = value
                self.tracker = OrnamentTracker(self.cfg, value)
                self.session = Session(self.cfg, value, self.mode)
            elif cmd == "mode":
                self.mode = value
                self.session.mode = value
            elif cmd == "new_session":
                self.tracker.reset()
                self.session = Session(self.cfg, self.sa_hz, self.mode)
            elif cmd == "record":
                self._recording = [] if value else None
            elif cmd == "find_sa":
                self._find_sa = {"until": self._samples_in + value * self.cfg.sample_rate,
                                 "f0": []}

    def _process_hop(self, hop_samples: np.ndarray) -> None:
        """Slide the frame by one hop and analyse it."""
        cfg = self.cfg
        start = time.perf_counter()
        self._buffer = np.concatenate([self._buffer[len(hop_samples):], hop_samples])
        self._samples_in += len(hop_samples)
        t = (self._samples_in - cfg.frame_size / 2) / cfg.sample_rate  # frame centre

        result = self.estimator.estimate(self._buffer)
        state = self.tracker.update(t, result.f0_hz, result.confidence)
        self.session.add(state)

        if np.isnan(state.cents):
            swara, target, error, status = "", np.nan, np.nan, "silent"
        else:
            sw = nearest_swara(float(cents_to_freq(state.cents, self.sa_hz)), self.sa_hz,
                               self.mode, cfg.in_tune_cents, cfg.close_cents)
            swara, target, error = sw.label, sw.cents - sw.cents_error, sw.cents_error
            status = {"glide": "gliding", "ornament": "ornament"}.get(state.state, sw.status)

        if self._find_sa is not None:
            self._find_sa["f0"].append(np.nan if result.f0_hz is None else result.f0_hz)
            if self._samples_in >= self._find_sa["until"]:
                f0 = np.array(self._find_sa["f0"])
                self._emit(SaEstimate(estimate_sa(f0, cfg.close_cents),
                                      int(np.sum(~np.isnan(f0)))))
                self._find_sa = None

        proc_ms = 1000.0 * (time.perf_counter() - start)
        self._emit(FrameResult(
            time=t, f0_hz=state.f0_hz, confidence=result.confidence, cents=state.cents,
            state=state.state, swara=swara, target_cents=target, cents_error=error,
            status=status, held_ms=state.held_ms, proc_ms=proc_ms,
            estimator=self.estimator.name))

    def _emit(self, item: Any) -> None:
        """Queue a result for the UI; if the UI is not keeping up, drop the oldest."""
        try:
            self.results.put_nowait(item)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.dropped_results += 1
            self.results.put_nowait(item)
