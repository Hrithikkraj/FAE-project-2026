"""Map a frequency to a swara relative to the singer's own Sa.

Audio concepts: cents (a logarithmic pitch distance, 1200 per octave, 100 per equal
semitone), octaves as frequency doubling, and two tuning systems: equal temperament and
just intonation (whole-number frequency ratios, closer to how a tanpura's harmonics
line up).

Notation used in labels: komal (flat) swaras are lower case (re, ga, dha, ni), tivra Ma
is Ma'. Mandra (lower) octave gets a leading dot (.Pa), taar (upper) octave a trailing
apostrophe (Sa'), madhya (middle) has no mark.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .config import Config

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]
JUST_RATIOS = [1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8]
OCTAVE_NAMES = {-1: "mandra", 0: "madhya", 1: "taar"}


def freq_to_cents(f_hz: float | np.ndarray, sa_hz: float) -> float | np.ndarray:
    """Pitch distance from Sa in cents: 1200 * log2(f / Sa). 0 = Sa, 1200 = taar Sa."""
    return 1200.0 * np.log2(np.asarray(f_hz, dtype=float) / sa_hz)


def cents_to_freq(cents: float | np.ndarray, sa_hz: float) -> float | np.ndarray:
    """Inverse of :func:`freq_to_cents`: Sa * 2^(cents / 1200)."""
    return sa_hz * 2.0 ** (np.asarray(cents, dtype=float) / 1200.0)


def swara_positions(mode: str = "equal") -> list[float]:
    """Cents of the 12 swaras within one octave for a tuning mode."""
    if mode == "equal":
        return [100.0 * i for i in range(12)]
    if mode == "just":
        return [1200.0 * math.log2(r) for r in JUST_RATIOS]
    raise ValueError('mode must be "equal" or "just"')


def octave_name(octave: int) -> str:
    """mandra / madhya / taar for -1 / 0 / +1; 'ati-' prefix one further out."""
    if octave in OCTAVE_NAMES:
        return OCTAVE_NAMES[octave]
    if octave == -2:
        return "ati-mandra"
    if octave == 2:
        return "ati-taar"
    return f"octave {octave:+d}"


def swara_label(name: str, octave: int) -> str:
    """Notation label: '.Pa' (mandra), 'Pa' (madhya), "Sa'" (taar); dots/marks repeat."""
    if octave < 0:
        return "." * (-octave) + name
    return name + "'" * octave


@dataclass(frozen=True)
class SwaraResult:
    """The swara nearest to a sung pitch, and how far off the singer is.

    Attributes:
        name: Swara name, e.g. "Pa" or "ga" (komal Ga).
        index: 0..11 position within the octave (0 = Sa).
        octave: -1 mandra, 0 madhya, +1 taar (can go further).
        octave_name: "mandra", "madhya", "taar", ...
        label: Name with octave marks, e.g. ".Pa" or "Sa'".
        target_hz: Exact frequency of that swara for this Sa and tuning mode.
        cents: Sung pitch relative to Sa (madhya Sa = 0).
        cents_error: Signed error, sung minus target; positive = sharp.
        status: "in_tune", "close", "sharp" or "flat".
    """

    name: str
    index: int
    octave: int
    octave_name: str
    label: str
    target_hz: float
    cents: float
    cents_error: float
    status: str


def tuning_status(cents_error: float, in_tune_cents: float, close_cents: float) -> str:
    """Classify a signed error: in_tune / close (within thresholds), else sharp / flat."""
    if abs(cents_error) <= in_tune_cents:
        return "in_tune"
    if abs(cents_error) <= close_cents:
        return "close"
    return "sharp" if cents_error > 0 else "flat"


def nearest_swara(
    f0_hz: float,
    sa_hz: float,
    mode: str = Config.tuning_mode,
    in_tune_cents: float = Config.in_tune_cents,
    close_cents: float = Config.close_cents,
) -> SwaraResult:
    """Find the swara closest (in cents) to ``f0_hz`` for the given Sa and tuning.

    Candidates come from the octave the pitch is in and the octaves either side, so a
    pitch at 1190 cents correctly maps to taar Sa (+1200), not to Ni.
    """
    if f0_hz <= 0 or sa_hz <= 0:
        raise ValueError("frequencies must be positive")
    cents = float(freq_to_cents(f0_hz, sa_hz))
    positions = swara_positions(mode)
    base = math.floor(cents / 1200.0)
    best = None
    for octave in (base - 1, base, base + 1):
        for index, pos in enumerate(positions):
            target = 1200.0 * octave + pos
            if best is None or abs(cents - target) < abs(cents - best[2]):
                best = (octave, index, target)
    octave, index, target = best
    error = cents - target
    name = SWARA_NAMES[index]
    return SwaraResult(
        name=name,
        index=index,
        octave=octave,
        octave_name=octave_name(octave),
        label=swara_label(name, octave),
        target_hz=float(cents_to_freq(target, sa_hz)),
        cents=cents,
        cents_error=error,
        status=tuning_status(error, in_tune_cents, close_cents),
    )


def swara_frequency(name: str, sa_hz: float, octave: int = 0, mode: str = "equal") -> float:
    """Frequency of a named swara, e.g. swara_frequency("Pa", 220) = 330 Hz (just)."""
    index = SWARA_NAMES.index(name)
    return float(cents_to_freq(1200.0 * octave + swara_positions(mode)[index], sa_hz))


_NOTE_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_hz(note: str, a4_hz: float = Config.a4_hz) -> float:
    """Western note name with octave to Hz (equal temperament), e.g. "C#3" -> 138.59.

    Accepts a letter A-G, optional '#' or 'b', and an octave number (scientific pitch:
    middle C = C4, A4 = ``a4_hz``). Useful for choosing Sa from a harmonium key.
    """
    match = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d+)\s*", note)
    if not match:
        raise ValueError(f"cannot parse note name {note!r} (expected e.g. 'C#3')")
    letter, accidental, octave = match.groups()
    pitch_class = _NOTE_PITCH_CLASS[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
    midi = 12 * (int(octave) + 1) + pitch_class  # MIDI note number, A4 = 69
    return a4_hz * 2.0 ** ((midi - 69) / 12.0)


def estimate_sa(
    f0_track: np.ndarray, tolerance_cents: float = Config.close_cents, min_frames: int = 5
) -> float | None:
    """Estimate Sa as the median pitch of the longest stable segment of a track.

    A segment is a run of voiced frames (non-NaN) that each stay within
    ``tolerance_cents`` of the running mean of the segment. The singer hums a
    comfortable note for a few seconds; short wobbles and other notes are ignored.

    Returns:
        Sa in Hz, or None if no segment has at least ``min_frames`` frames.
    """
    f0 = np.asarray(f0_track, dtype=float)
    best: list[float] = []
    current: list[float] = []
    sum_cents = 0.0
    for f in f0:
        if np.isnan(f) or f <= 0:
            current, sum_cents = [], 0.0
            continue
        c = 1200.0 * math.log2(f)  # absolute cents (any reference works)
        if current and abs(c - sum_cents / len(current)) > tolerance_cents:
            current, sum_cents = [], 0.0  # jumped to another note: new segment
        current.append(f)
        sum_cents += c
        if len(current) > len(best):
            best = current  # same list object; it keeps growing while it is the longest
    if len(best) < min_frames:
        return None
    return float(np.median(best))
