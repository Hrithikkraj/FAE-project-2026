"""Tests for the ornament-aware logic and session statistics.

Concepts checked: stable / glide / ornament labelling from slope and range, that only
held notes are judged, and that glides and gamakas are never counted as errors.
Most tests feed exact pitch tracks (fast, deterministic); the integration tests run
synthetic audio through YIN, like the real app.
"""

from __future__ import annotations

import csv
from typing import Callable

import numpy as np
import pytest

from swara_aware import synth
from swara_aware.config import Config
from swara_aware.ornament import OrnamentTracker, back_travel
from swara_aware.pitch import YinEstimator, estimate_track
from swara_aware.session import Session, analyse_track
from swara_aware.swara import cents_to_freq

CFG = Config()
SA = 220.0
HOP_S = CFG.hop_size / CFG.sample_rate


def track_from_cents(fn: Callable[[np.ndarray], np.ndarray], duration: float):
    """Exact pitch track sampled at frame times: (times, f0_hz)."""
    t = np.arange(0.0, duration, HOP_S)
    return t, cents_to_freq(fn(t), SA)


def segments(states: list[str], times: np.ndarray, min_s: float = 0.06) -> list[str]:
    """Collapse per-frame states into runs, dropping runs shorter than ``min_s``."""
    runs: list[list] = []
    for s, t in zip(states, times):
        if runs and runs[-1][0] == s:
            runs[-1][2] = t
        else:
            runs.append([s, t, t])
    kept = [r[0] for r in runs if r[2] - r[1] + HOP_S >= min_s]
    return [s for i, s in enumerate(kept) if i == 0 or s != kept[i - 1]]


def hold_meend_hold(t: np.ndarray) -> np.ndarray:
    """Sa held 0.7 s, meend up to Ma (500 cents) over 0.4 s, Ma held 0.7 s."""
    return np.clip((t - 0.7) / 0.4, 0.0, 1.0) * 500.0


def gamaka(t: np.ndarray) -> np.ndarray:
    """Re held 0.6 s, then +-50 cents at 5 Hz for 1.2 s, then Re held 0.6 s."""
    osc = (t >= 0.6) & (t < 1.8)
    return 200.0 + np.where(osc, 50.0 * np.sin(2 * np.pi * 5.0 * (t - 0.6)), 0.0)


# ---------- logic on exact tracks ----------

def test_held_meend_held_is_stable_glide_stable() -> None:
    t, f0 = track_from_cents(hold_meend_hold, 1.8)
    states = [fr.state for fr in OrnamentTracker(CFG, SA).process(t, f0)]
    assert segments(states, t) == ["stable", "glide", "stable"]
    # Middle of each part has the right label.
    at = lambda sec: states[int(sec / HOP_S)]  # noqa: E731
    assert (at(0.4), at(0.9), at(1.5)) == ("stable", "glide", "stable")


def test_meend_is_never_judged() -> None:
    t, f0 = track_from_cents(hold_meend_hold, 1.8)
    session = analyse_track(t, f0, np.ones(len(t)), CFG, SA)
    notes = session.held_notes()
    assert [n.label for n in notes] == ["Sa", "Ma"]
    assert all(n.status == "in_tune" for n in notes)
    assert session.summary()["in_tune_pct"] == pytest.approx(100.0)
    # No judged frame lies inside the meend (0.7 .. 1.1 s).
    assert all(n.end < 0.75 or n.start > 1.1 for n in notes)


def test_gamaka_is_ornament_and_never_an_error() -> None:
    t, f0 = track_from_cents(gamaka, 2.4)
    frames = OrnamentTracker(CFG, SA).process(t, f0)
    inside = [fr.state for fr in frames if 0.6 + CFG.ornament_window_ms / 1000 <= fr.time < 1.8]
    assert np.mean([s == "ornament" for s in inside]) > 0.95

    session = analyse_track(t, f0, np.ones(len(t)), CFG, SA)
    notes = session.held_notes()
    assert all(n.end < 0.65 or n.start > 1.8 for n in notes), "gamaka judged as a held note"
    assert all(n.label == "Re" and n.status == "in_tune" for n in notes)
    assert session.summary()["in_tune_pct"] == pytest.approx(100.0)


def test_held_note_40_cents_flat_is_judged_flat() -> None:
    t, f0 = track_from_cents(lambda t: np.full_like(t, 700.0 - 40.0), 1.0)
    notes = analyse_track(t, f0, np.ones(len(t)), CFG, SA).held_notes()
    assert len(notes) == 1
    assert notes[0].label == "Pa"
    assert notes[0].mean_error == pytest.approx(-40.0, abs=0.01)
    assert notes[0].status == "flat"


def test_small_vibrato_counts_as_stable_and_is_judged_on_its_mean() -> None:
    t, f0 = track_from_cents(lambda t: 700.0 + 10.0 * np.sin(2 * np.pi * 5.5 * t), 1.5)
    session = analyse_track(t, f0, np.ones(len(t)), CFG, SA)
    states = [fr.state for fr in session.frames]
    assert np.mean([s == "stable" for s in states]) > 0.95
    assert abs(session.held_notes()[0].mean_error) < 3.0


def test_short_stable_run_is_not_judged() -> None:
    # Pa held for only 150 ms, shorter than stable_min_ms (250 ms).
    t, f0 = track_from_cents(lambda t: np.full_like(t, 700.0), 0.15)
    assert analyse_track(t, f0, np.ones(len(t)), CFG, SA).held_notes() == []


def test_silence_and_low_confidence_are_silent() -> None:
    t = np.arange(0, 0.3, HOP_S)
    f0 = np.full(len(t), np.nan)
    f0[10:15] = 300.0  # voiced, but with low confidence below
    conf = np.where(np.isnan(f0), 0.0, 0.1)
    states = {fr.state for fr in OrnamentTracker(CFG, SA).process(t, f0, conf)}
    assert states == {"silent"}


def test_live_update_matches_array_processing() -> None:
    t, f0 = track_from_cents(gamaka, 2.4)
    array_states = [fr.state for fr in OrnamentTracker(CFG, SA).process(t, f0)]
    live = OrnamentTracker(CFG, SA)
    live_states = [live.update(ti, fi, 1.0).state for ti, fi in zip(t, f0)]
    assert live_states == array_states


def test_back_travel() -> None:
    assert back_travel(np.linspace(0, 300, 30)) == pytest.approx(0.0)
    assert back_travel(np.array([0, 50, 0, -50, 0.0])) == pytest.approx(100.0)


# ---------- integration: synthetic audio -> YIN -> logic ----------

def run_yin(cents_fn: Callable[[np.ndarray], np.ndarray], duration: float) -> Session:
    sr = CFG.sample_rate
    t = np.arange(int(duration * sr)) / sr
    x = synth.voice_from_contour(cents_to_freq(cents_fn(t), SA), sr)
    yin = YinEstimator(sr, CFG.fmin, CFG.fmax)
    times, f0, conf = estimate_track(x, yin, CFG.frame_size, CFG.hop_size)
    return analyse_track(times, f0, conf, CFG, SA)


def test_audio_meend_through_yin() -> None:
    session = run_yin(hold_meend_hold, 1.8)
    times = np.array([fr.time for fr in session.frames])
    assert segments([fr.state for fr in session.frames], times) == ["stable", "glide", "stable"]
    assert [n.label for n in session.held_notes()] == ["Sa", "Ma"]
    assert session.summary()["in_tune_pct"] == pytest.approx(100.0)


def test_audio_gamaka_through_yin_is_not_an_error() -> None:
    session = run_yin(gamaka, 2.4)
    summary = session.summary()
    assert summary["seconds_by_state"]["ornament"] > 0.7
    assert summary["in_tune_pct"] == pytest.approx(100.0)


def test_audio_flat_pa_through_yin() -> None:
    notes = run_yin(lambda t: np.full_like(t, 660.0), 1.0).held_notes()
    assert len(notes) == 1 and notes[0].label == "Pa" and notes[0].status == "flat"
    assert notes[0].mean_error == pytest.approx(-40.0, abs=1.0)


# ---------- session output ----------

def test_summary_csv_and_plots(tmp_path) -> None:
    t, f0 = track_from_cents(hold_meend_hold, 1.8)
    session = analyse_track(t, f0, np.ones(len(t)), CFG, SA)
    summary = session.summary()
    assert set(summary["per_swara"]) == {"Sa", "Ma"}
    assert summary["per_swara"]["Sa"]["seconds"] > 0.4

    path = tmp_path / "track.csv"
    session.export_csv(path)
    with open(path) as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["time", "f0", "cents", "swara", "state"]
    assert len(rows) == len(t) + 1
    assert rows[int(0.4 / HOP_S) + 1][3:] == ["Sa", "stable"]

    session.plot_trace(tmp_path / "trace.png")
    session.plot_accuracy(tmp_path / "acc.png")
    assert (tmp_path / "trace.png").stat().st_size > 0
    assert (tmp_path / "acc.png").stat().st_size > 0
