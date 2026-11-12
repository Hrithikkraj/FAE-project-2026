"""Pitch estimators written with numpy: FFT peak / HPS, autocorrelation, YIN.

Audio concepts: fundamental frequency, periodicity, autocorrelation, harmonics.
``reference.py`` wraps librosa's pYIN for comparison only.
"""

from __future__ import annotations

from ..config import Config
from .autocorr import AutocorrEstimator
from .base import PitchEstimator, PitchResult, cents_error, estimate_track
from .fft_peak import FFTPeakEstimator
from .yin import YinEstimator

# Estimator names in a fixed order (menus, tables, plots) with display labels.
METHOD_LABELS = {
    "fft_peak": "FFT peak",
    "fft_hps": "FFT + HPS",
    "autocorr": "Autocorrelation",
    "yin": "YIN",
}


def make_estimators(cfg: Config) -> list[PitchEstimator]:
    """Build every estimator from one Config, in a fixed order for comparisons."""
    common = dict(
        sample_rate=cfg.sample_rate,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        silence_threshold_dbfs=cfg.silence_threshold_dbfs,
    )
    fft_kw = dict(
        zero_pad_factor=cfg.fft_zero_pad_factor,
        voicing_threshold=cfg.fft_voicing_threshold,
        hps_harmonics=cfg.hps_harmonics,
    )
    return [
        FFTPeakEstimator(**common, **fft_kw, use_hps=False),
        FFTPeakEstimator(**common, **fft_kw, use_hps=True),
        AutocorrEstimator(
            **common,
            voicing_threshold=cfg.acf_voicing_threshold,
            peak_ratio=cfg.acf_peak_ratio,
        ),
        YinEstimator(**common, threshold=cfg.yin_threshold),
    ]


__all__ = [
    "AutocorrEstimator",
    "FFTPeakEstimator",
    "METHOD_LABELS",
    "PitchEstimator",
    "PitchResult",
    "YinEstimator",
    "cents_error",
    "estimate_track",
    "make_estimators",
]
