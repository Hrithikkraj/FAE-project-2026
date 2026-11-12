"""Swara-Aware: real-time, swara-based pitch feedback for learner singers.

Modules:
    config    -- tunable defaults (sampling rate, framing, thresholds)
    dsp       -- framing, windowing, RMS/dBFS, FFT, spectrogram, SNR
    synth     -- synthetic test signals with known pitch
    audio_io  -- microphone recording, live stream, WAV I/O
    pitch     -- pitch estimators (Stage 2)
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
