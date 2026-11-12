"""Tests for Stage 1 DSP basics, using synthetic signals with known answers.

Concepts checked: framing, RMS, FFT bin resolution, SNR, spectrogram shape.
"""

import numpy as np
import pytest

from swara_aware import dsp, synth
from swara_aware.config import Config

CFG = Config()
SR = CFG.sample_rate


# ---------- framing ----------

def test_frame_shape_exact_fit() -> None:
    n, hop = 2048, 512
    x = np.arange(n + 9 * hop, dtype=float)  # fits exactly into 10 frames
    frames = dsp.frame_signal(x, n, hop)
    assert frames.shape == (10, n)


def test_frame_overlap_is_correct() -> None:
    n, hop = 2048, 512
    x = np.arange(10_000, dtype=float)
    frames = dsp.frame_signal(x, n, hop)
    # Each frame starts hop samples after the previous one ...
    assert np.array_equal(frames[:, 0], hop * np.arange(len(frames)))
    # ... so the last (n - hop) samples of frame i equal the first of frame i+1.
    assert np.array_equal(frames[0, hop:], frames[1, : n - hop])
    assert np.array_equal(frames[3], x[3 * hop : 3 * hop + n])


def test_frame_pads_last_frame() -> None:
    x = np.ones(3000)
    frames = dsp.frame_signal(x, 2048, 512)
    assert frames.shape == (3, 2048)  # 1 + ceil((3000-2048)/512) = 3
    # Frames start at 0, 512, 1024; the last one holds only 3000 - 1024 = 1976 real samples.
    assert frames.sum(axis=1).tolist() == [2048, 2048, 1976]
    assert frames[-1, -1] == 0.0  # zero padding at the end


def test_frame_short_signal_gives_one_frame() -> None:
    frames = dsp.frame_signal(np.ones(100), 2048, 512)
    assert frames.shape == (1, 2048)


# ---------- windowing, RMS, dBFS ----------

def test_hann_window_shape() -> None:
    w = dsp.hann_window(8)
    assert w[0] == pytest.approx(0.0)
    assert w[4] == pytest.approx(1.0)  # peak at the centre (periodic Hann)
    assert np.allclose(w[1:], w[1:][::-1])  # symmetric apart from the first sample


def test_rms_of_unit_sine() -> None:
    x = synth.sine_tone(440.0, 1.0, SR, amplitude=1.0)
    assert dsp.rms(x) == pytest.approx(1 / np.sqrt(2), abs=1e-3)


def test_dbfs_values() -> None:
    assert dsp.to_dbfs(1.0) == pytest.approx(0.0)
    assert dsp.to_dbfs(1 / np.sqrt(2)) == pytest.approx(-3.01, abs=0.01)
    assert np.isfinite(dsp.to_dbfs(0.0))


def test_silence_gate() -> None:
    loud = synth.sine_tone(220.0, 0.5, SR, amplitude=0.3)
    quiet = np.zeros(len(loud))
    frames = dsp.frame_signal(np.concatenate([loud, quiet]), CFG.frame_size, CFG.hop_size)
    gate = dsp.silence_gate(frames, CFG.silence_threshold_dbfs)
    assert gate[0] and not gate[-1]


# ---------- FFT ----------

def test_fft_peak_of_440_sine_within_one_bin() -> None:
    n = CFG.frame_size
    x = synth.sine_tone(440.0, n / SR, SR, amplitude=0.5)
    freqs, mags = dsp.magnitude_spectrum(x, SR)
    bin_width = SR / n
    peak_freq = freqs[np.argmax(mags)]
    assert abs(peak_freq - 440.0) <= bin_width
    assert len(freqs) == n // 2 + 1


def test_fft_amplitude_scaling() -> None:
    # A sine exactly on a bin centre should read back its amplitude.
    n = CFG.frame_size
    f = 20 * SR / n
    x = synth.sine_tone(f, n / SR, SR, amplitude=0.5)
    _, mags = dsp.magnitude_spectrum(x, SR)
    assert mags.max() == pytest.approx(0.5, rel=1e-6)


# ---------- SNR ----------

@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0, 30.0])
def test_add_noise_hits_requested_snr(snr_db: float) -> None:
    rng = np.random.default_rng(0)
    clean = synth.voice_like_tone(196.0, 1.0, SR)
    noisy = dsp.add_noise(clean, snr_db, rng=rng)
    measured = dsp.measure_snr_db(clean, noisy - clean)
    assert abs(measured - snr_db) < 0.5


def test_add_noise_with_pink_noise() -> None:
    clean = synth.sine_tone(300.0, 1.0, SR)
    pink = synth.pink_noise(1.0, SR, rng=np.random.default_rng(1))
    noisy = dsp.add_noise(clean, 15.0, noise=pink)
    assert abs(dsp.measure_snr_db(clean, noisy - clean) - 15.0) < 0.5


# ---------- spectrogram ----------

def test_spectrogram_shape() -> None:
    x = synth.sine_tone(440.0, 1.0, SR)
    n, hop = CFG.frame_size, CFG.hop_size
    times, freqs, S = dsp.spectrogram(x, SR, n, hop)
    n_frames = dsp.frame_signal(x, n, hop).shape[0]
    assert S.shape == (n // 2 + 1, n_frames)
    assert times.shape == (n_frames,)
    assert freqs.shape == (n // 2 + 1,)
    # Every column's peak should be near 440 Hz.
    peak_freqs = freqs[np.argmax(S[:, 1:-1], axis=0)]
    assert np.all(np.abs(peak_freqs - 440.0) <= SR / n)


# ---------- synth sanity ----------

def test_missing_fundamental_has_no_f0_energy() -> None:
    n = CFG.frame_size
    f0 = 10 * SR / n  # on a bin centre
    x = synth.voice_like_tone(f0, n / SR, SR, missing_fundamental=True)
    freqs, mags = dsp.magnitude_spectrum(x, SR)
    k0 = int(round(f0 / (SR / n)))
    assert mags[k0] < 1e-3 * mags.max()


def test_glide_ends_at_target_frequency() -> None:
    f = synth.glide_frequencies(220.0, 330.0, 1.0, SR)
    assert f[0] == pytest.approx(220.0) and f[-1] == pytest.approx(330.0)
    assert len(synth.glide(220.0, 330.0, 1.0, SR)) == len(f)


def test_config_rejects_bad_values() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(CFG, hop_size=4096)
    with pytest.raises(ValueError):
        replace(CFG, fmin=2000.0)


# ---------- remaining Stage 1 helpers ----------

def test_apply_window_and_frame_dbfs() -> None:
    frames = np.ones((3, 8))
    assert np.allclose(dsp.apply_window(frames), dsp.hann_window(8))
    x = synth.sine_tone(440.0, 0.5, SR, amplitude=1.0)
    levels = dsp.frame_dbfs(x, CFG.frame_size, CFG.hop_size)
    assert np.allclose(levels[:-1], -3.01, atol=0.05)  # full-scale sine = -3 dBFS


def test_white_noise_level_and_seed() -> None:
    a = synth.white_noise(1.0, SR, amplitude=0.1, rng=np.random.default_rng(5))
    b = synth.white_noise(1.0, SR, amplitude=0.1, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert dsp.rms(a) == pytest.approx(0.1, rel=0.02)
