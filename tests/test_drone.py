"""Tests for the tanpura drone (rendered offline; no sound is played).

Concepts checked: additive synthesis puts energy at the string frequencies, the output
stays within full scale, blocks join without clicks, and rendering is cheap.
"""

import time

import numpy as np
import pytest

from swara_aware import dsp
from swara_aware.config import Config
from swara_aware.drone import TanpuraDrone

CFG = Config()
SA = 146.83  # D3


def spectrum_peak_near(x: np.ndarray, f: float, sr: int) -> float:
    """Magnitude (dB) of the strongest bin within 2 Hz of ``f``, minus the median."""
    freqs, mags = dsp.magnitude_spectrum(x, sr)
    band = np.abs(freqs - f) <= 2.0
    return float(dsp.amplitude_to_db(mags[band].max()) - dsp.amplitude_to_db(np.median(mags)))


def test_render_length_and_level() -> None:
    drone = TanpuraDrone.from_config(CFG, SA)
    x = drone.render_seconds(4.0)
    assert len(x) == 4 * CFG.sample_rate
    assert np.max(np.abs(x)) < 1.0
    assert dsp.rms(x) > 0.01


@pytest.mark.parametrize("first,ratio", [("Pa", 3 / 4), ("Ma", 2 / 3)])
def test_strings_present_in_spectrum(first: str, ratio: float) -> None:
    drone = TanpuraDrone(SA, CFG.sample_rate, first_string=first)
    x = drone.render_seconds(4.0)  # one full cycle at 60 bpm
    for f in (SA, SA / 2, SA * ratio):
        assert spectrum_peak_near(x, f, CFG.sample_rate) > 30.0, f"no energy at {f:.1f} Hz"


def test_volume_zero_is_silent_and_blocks_are_continuous() -> None:
    drone = TanpuraDrone(SA, CFG.sample_rate)
    whole = drone.render(8192)
    drone.reset()
    parts = np.concatenate([drone.render(512) for _ in range(16)])
    assert np.allclose(whole, parts, atol=1e-9)  # block size does not change the sound

    drone.set_volume(0.0)
    drone.render(512)  # ramp block
    assert np.max(np.abs(drone.render(512))) == 0.0


def test_retuning_does_not_click() -> None:
    drone = TanpuraDrone(SA, CFG.sample_rate)
    a = drone.render(CFG.sample_rate // 2)
    drone.set_sa(SA * 2 ** (100 / 1200))
    b = drone.render(512)
    typical_step = np.max(np.abs(np.diff(a)))
    assert abs(b[0] - a[-1]) <= 1.5 * typical_step


def test_render_is_cheap() -> None:
    drone = TanpuraDrone(SA, CFG.sample_rate)
    start = time.perf_counter()
    for _ in range(100):  # 100 blocks of 1024 samples = 2.3 s of audio
        drone.render(1024)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5 * (100 * 1024 / CFG.sample_rate)  # well under real time
