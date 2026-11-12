"""Smoke test for the live-app processing worker (no GUI): synthetic glide in, results out.

Concepts checked: the rolling buffer + estimator + ornament pipeline works on streamed
blocks, results arrive in order, and a held note / glide / held note is recognised.
"""

import numpy as np
import pytest

from swara_aware import synth
from swara_aware.app.sources import ArraySource
from swara_aware.app.worker import EndOfStream, FrameResult, ProcessingWorker, SaEstimate
from swara_aware.config import Config
from swara_aware.swara import cents_to_freq

CFG = Config()
SA = 220.0


def run_worker(signal: np.ndarray, estimator: str = "yin", find_sa: float | None = None):
    source = ArraySource(signal, CFG.sample_rate, CFG.hop_size, realtime=False)
    worker = ProcessingWorker(source, CFG, SA, estimator=estimator)
    if find_sa is not None:
        worker.find_sa(find_sa)
    source.start()
    worker.start()
    worker.join(timeout=30)
    source.stop()
    assert not worker.is_alive(), "worker did not finish"
    items = []
    while not worker.results.empty():
        items.append(worker.results.get())
    return worker, items


def glide_signal() -> tuple[np.ndarray, np.ndarray]:
    """Sa 0.8 s, glide up to Pa (700 cents) over 0.6 s, Pa 0.8 s. Returns (x, true_f0)."""
    sr = CFG.sample_rate
    t = np.arange(int(2.2 * sr)) / sr
    cents = np.clip((t - 0.8) / 0.6, 0, 1) * 700
    f0 = cents_to_freq(cents, SA)
    return synth.voice_from_contour(f0, sr), f0


def test_worker_produces_results_for_gliding_tone() -> None:
    x, _ = glide_signal()
    worker, items = run_worker(x)
    frames = [i for i in items if isinstance(i, FrameResult)]
    assert isinstance(items[-1], EndOfStream)
    assert len(frames) >= len(x) // CFG.hop_size - 1
    assert all(b.time > a.time for a, b in zip(frames, frames[1:]))
    states = {f.state for f in frames}
    assert {"stable", "glide"} <= states
    # Held Sa at 0.5 s and held Pa at 1.9 s are identified and in tune.
    at = lambda sec: min(frames, key=lambda f: abs(f.time - sec))  # noqa: E731
    assert (at(0.5).swara, at(0.5).status) == ("Sa", "in_tune")
    assert (at(1.9).swara, at(1.9).status) == ("Pa", "in_tune")
    assert at(1.1).status == "gliding"
    assert max(f.proc_ms for f in frames) < 1000 * CFG.hop_size / CFG.sample_rate * 10
    notes = worker.snapshot_session().held_notes()
    assert [n.label for n in notes] == ["Sa", "Pa"]


@pytest.mark.parametrize("estimator", ["fft_peak", "fft_hps", "autocorr", "yin"])
def test_every_estimator_runs_in_worker(estimator: str) -> None:
    x, _ = glide_signal()
    _, items = run_worker(x, estimator)
    frames = [i for i in items if isinstance(i, FrameResult)]
    assert frames and all(f.estimator == estimator for f in frames)


def test_find_my_sa_in_worker() -> None:
    hum = synth.voice_like_tone(196.0, 3.5, CFG.sample_rate)
    _, items = run_worker(hum, find_sa=3.0)
    estimates = [i for i in items if isinstance(i, SaEstimate)]
    assert len(estimates) == 1 and estimates[0].sa_hz == pytest.approx(196.0, rel=0.003)
