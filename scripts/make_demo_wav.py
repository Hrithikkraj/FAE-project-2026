"""Write data/demo_phrase.wav: a synthetic "singer" for demoing the live app with --wav.

Audio concepts: a voice-like additive tone following a scripted pitch contour, plus
room-like pink noise at 35 dB SNR.

Contents (Sa = D3 = 146.83 Hz):
  0.0-0.5 s silence | 0.5-3.5 s steady hum on Sa (try "Find my Sa" here) |
  silence | Sa, Re, Ga held | meend Ga->Pa | Pa held 20 cents flat |
  gamaka (+-50 cents at 5 Hz) on Dha | Dha held 40 cents sharp | silence

Usage:  python scripts/make_demo_wav.py   then   python scripts/run_app.py --wav data/demo_phrase.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from swara_aware import audio_io, dsp, synth  # noqa: E402
from swara_aware.swara import cents_to_freq, note_to_hz  # noqa: E402

SR = 44100
# (duration s, start cents, end cents, gamaka depth in cents); None = silence
SCRIPT = [
    (0.5, None, None, 0), (3.0, 0, 0, 0), (0.6, None, None, 0),
    (0.9, 0, 0, 0), (0.06, 0, 200, 0), (0.8, 200, 200, 0), (0.06, 200, 400, 0),
    (0.8, 400, 400, 0), (0.5, 400, 700, 0), (1.2, 680, 680, 0),
    (0.15, 680, 900, 0), (1.5, 900, 900, 50), (1.2, 940, 940, 0), (0.8, None, None, 0),
]


def contour() -> np.ndarray:
    """Per-sample f0 in Hz (NaN = silence) for SCRIPT."""
    sa = note_to_hz("D3")
    parts = []
    for dur, c0, c1, depth in SCRIPT:
        n = int(round(dur * SR))
        if c0 is None:
            parts.append(np.full(n, np.nan))
            continue
        t = np.arange(n) / SR
        cents = np.linspace(c0, c1, n) + depth * np.sin(2 * np.pi * 5.0 * t)
        parts.append(cents_to_freq(cents, sa))
    return np.concatenate(parts)


def main() -> None:
    rng = np.random.default_rng(3)
    voice = synth.voice_from_contour(contour(), SR)
    x = dsp.add_noise(voice, 35.0, noise=synth.pink_noise(len(voice) / SR, SR, rng=rng))
    out = ROOT / "data" / "demo_phrase.wav"
    audio_io.save_wav(out, x, SR)
    print(f"Saved {out} ({len(x) / SR:.1f} s)")


if __name__ == "__main__":
    main()
