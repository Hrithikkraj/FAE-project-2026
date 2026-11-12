"""Reference pitch tracker: a thin wrapper around librosa's pYIN.

Audio concept: probabilistic YIN (pYIN, Mauch and Dixon 2014) adds an HMM over many
YIN thresholds for smooth, robust tracks. We use it ONLY for comparison and as an
approximate ground truth for real voice recordings, never inside our own methods.
librosa is imported lazily, so the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np


def reference_track(
    signal: np.ndarray,
    sample_rate: int,
    fmin: float,
    fmax: float,
    frame_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run librosa.pyin and return ``(times, f0, confidence)`` like ``estimate_track``.

    f0 is NaN for unvoiced frames; confidence is pYIN's voiced probability. Times are
    aligned to the same frame-centre convention as ``estimate_track``.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reference_track needs librosa: pip install librosa") from exc

    # Zero-pad exactly like dsp.frame_signal, and use center=False, so frame i covers
    # samples [i*hop, i*hop + frame_size) and the frame count matches estimate_track.
    x = np.asarray(signal, dtype=float)
    n_frames = 1 if len(x) <= frame_size else 1 + int(np.ceil((len(x) - frame_size) / hop_size))
    x = np.pad(x, (0, (n_frames - 1) * hop_size + frame_size - len(x)))
    f0, voiced, prob = librosa.pyin(
        x,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_size,
        hop_length=hop_size,
        center=False,
    )
    f0 = np.where(voiced, f0, np.nan)
    times = (np.arange(len(f0)) * hop_size + frame_size / 2) / sample_rate
    return times, f0, np.nan_to_num(prob)
