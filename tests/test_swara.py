"""Tests for swara mapping: cents from Sa, tuning modes, octaves, note names, Sa estimate.

Concepts checked: cents = 1200*log2(f/Sa), equal vs just intonation, octave naming.
"""

import math

import numpy as np
import pytest

from swara_aware.config import Config
from swara_aware.swara import (
    SWARA_NAMES,
    estimate_sa,
    freq_to_cents,
    nearest_swara,
    note_to_hz,
    swara_frequency,
    tuning_status,
)

CFG = Config()
SA = 220.0


def test_freq_to_cents_basics() -> None:
    assert freq_to_cents(220.0, SA) == pytest.approx(0.0)
    assert freq_to_cents(440.0, SA) == pytest.approx(1200.0)
    assert freq_to_cents(110.0, SA) == pytest.approx(-1200.0)


def test_440_is_taar_sa_with_zero_cents() -> None:
    r = nearest_swara(440.0, SA)
    assert (r.name, r.octave, r.octave_name, r.label) == ("Sa", 1, "taar", "Sa'")
    assert r.cents_error == pytest.approx(0.0, abs=1e-9)
    assert r.target_hz == pytest.approx(440.0)
    assert r.status == "in_tune"


@pytest.mark.parametrize("f,expected_name,expected_octave", [
    (225.0, "Sa", 0), (215.0, "Sa", 0), (445.0, "Sa'", 1), (335.0, "Pa", 0),
    # 5 Hz above mandra Sa (110 Hz) is +77 cents: nearer to komal re than to Sa.
    (115.0, ".re", -1),
])
def test_5hz_offsets_give_signed_cents(f: float, expected_name: str, expected_octave: int) -> None:
    r = nearest_swara(f, SA)
    expected = 1200 * math.log2(f / r.target_hz)
    assert r.cents_error == pytest.approx(expected, abs=1e-9)
    assert (r.cents_error > 0) == (f > r.target_hz)  # positive = sharp
    assert r.label == expected_name and r.octave == expected_octave


def test_5hz_offset_values_and_status() -> None:
    sharp = nearest_swara(225.0, SA)   # +38.9 cents
    flat = nearest_swara(215.0, SA)    # -39.8 cents
    assert sharp.cents_error == pytest.approx(38.91, abs=0.01) and sharp.status == "sharp"
    assert flat.cents_error == pytest.approx(-39.80, abs=0.01) and flat.status == "flat"
    # Same 5 Hz at the octave above is only half the cents: pitch is logarithmic.
    assert nearest_swara(445.0, SA).cents_error == pytest.approx(19.56, abs=0.01)


def test_pa_330_just_vs_equal() -> None:
    just = nearest_swara(330.0, SA, mode="just")
    equal = nearest_swara(330.0, SA, mode="equal")
    assert just.name == equal.name == "Pa"
    assert just.cents_error == pytest.approx(0.0, abs=1e-6)
    assert equal.cents_error == pytest.approx(1.955, abs=0.01)  # 3/2 is 701.955 cents


@pytest.mark.parametrize("mode", ["equal", "just"])
def test_every_swara_round_trips_in_three_octaves(mode: str) -> None:
    for octave, oname in [(-1, "mandra"), (0, "madhya"), (1, "taar")]:
        for name in SWARA_NAMES:
            r = nearest_swara(swara_frequency(name, SA, octave, mode), SA, mode)
            assert (r.name, r.octave, r.octave_name) == (name, octave, oname)
            assert abs(r.cents_error) < 1e-6


def test_wraps_up_to_next_sa() -> None:
    r = nearest_swara(SA * 2 ** (1190 / 1200), SA)  # 10 cents below taar Sa
    assert r.label == "Sa'" and r.cents_error == pytest.approx(-10.0)


def test_status_thresholds() -> None:
    assert tuning_status(15.0, 15, 30) == "in_tune"
    assert tuning_status(-29.0, 15, 30) == "close"
    assert tuning_status(31.0, 15, 30) == "sharp"
    assert tuning_status(-40.0, 15, 30) == "flat"


def test_note_to_hz() -> None:
    assert note_to_hz("A4") == pytest.approx(440.0)
    assert note_to_hz("C#3") == pytest.approx(138.591, abs=1e-3)
    assert note_to_hz("Db3") == pytest.approx(note_to_hz("C#3"))
    assert note_to_hz("C4") == pytest.approx(261.626, abs=1e-3)
    assert note_to_hz("A4", a4_hz=432.0) == pytest.approx(432.0)
    with pytest.raises(ValueError):
        note_to_hz("H2")


def test_estimate_sa_picks_longest_stable_segment() -> None:
    rng = np.random.default_rng(0)
    hum = 196.0 * 2 ** (rng.normal(0, 4, 200) / 1200)          # long steady hum, 4 c jitter
    other = np.full(60, 261.6)                                 # shorter different note
    glide = 150.0 * 2 ** (np.linspace(0, 700, 40) / 1200)      # a slide: never stable
    track = np.concatenate([glide, [np.nan] * 5, other, [np.nan] * 5, hum, [np.nan] * 5])
    sa = estimate_sa(track)
    assert sa is not None and abs(1200 * math.log2(sa / 196.0)) < 3.0


def test_estimate_sa_none_when_no_stable_segment() -> None:
    assert estimate_sa(np.array([np.nan, 200.0, np.nan, 300.0])) is None
