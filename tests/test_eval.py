"""Fast checks of the Stage 5 experiments (tiny sizes, temporary output folder).

Concepts checked: each experiment runs, writes its files, is reproducible with the fixed
seed, and its headline numbers are in a sane range.
"""

import csv

import numpy as np
import pytest

from swara_aware import audio_io, synth
from swara_aware.config import Config
from swara_aware.eval import accuracy, latency, noise, ornament_eval, sensitivity, voice
from swara_aware.eval.common import md_table, rng_for
from swara_aware.eval.report import write_report

CFG = Config()


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_seeded_generators_are_reproducible_and_independent() -> None:
    assert rng_for("E1").random() == rng_for("E1").random()
    assert rng_for("E1").random() != rng_for("E2").random()


def test_md_table() -> None:
    table = md_table(["a", "b"], [(1, 2.345), ("x", float("nan"))], digits=2)
    assert table.splitlines()[2] == "| 1 | 2.35 |" and "–" in table


def test_e1_accuracy(tmp_path) -> None:
    res = accuracy.run(CFG, tmp_path, n_freqs=4, trials=1)
    rows = {(r["signal"], r["method"]): r for r in read_csv(tmp_path / "e1_accuracy.csv")}
    assert float(rows[("harmonic", "yin")]["mae_cents"]) < 5
    assert float(rows[("missing_f0", "fft_peak")]["gross_pct"]) > 50  # known failure
    assert (tmp_path / "e1_accuracy.png").stat().st_size > 0 and res.finding
    again = accuracy.run(CFG, tmp_path, n_freqs=4, trials=1)  # same seed -> same numbers
    assert again.markdown == res.markdown


def test_e2_noise(tmp_path) -> None:
    noise.run(CFG, tmp_path, n_freqs=3, trials=1)
    rows = read_csv(tmp_path / "e2_noise.csv")
    clean = [r for r in rows if r["snr_db"] == "clean" and r["method"] == "yin"]
    assert clean and all(float(r["within_25c_pct"]) == 100.0 for r in clean)


def test_e3_latency(tmp_path) -> None:
    latency.run(CFG, tmp_path, n_frames=20)
    rows = read_csv(tmp_path / "e3_latency.csv")
    for r in rows:
        assert 0 < float(r["detect_display_ms"]) < 200
        assert float(r["detect_judged_ms"]) >= CFG.stable_min_ms


def test_e4_voice_skips_without_recordings(tmp_path) -> None:
    res = voice.run(CFG, tmp_path, voice_dir=tmp_path / "missing")
    assert not res.ran and "not as perfect ground truth" in res.markdown


def test_e4_voice_on_synthetic_recording(tmp_path) -> None:
    pytest.importorskip("librosa")
    voice_dir = tmp_path / "voice"
    sr = CFG.sample_rate
    t = np.arange(int(1.0 * sr)) / sr
    audio_io.save_wav(voice_dir / "glide.wav",
                      synth.voice_from_contour(196.0 * 2 ** (t * 0.5), sr), sr)
    res = voice.run(CFG, tmp_path, voice_dir=voice_dir)
    rows = read_csv(tmp_path / "e4_voice.csv")
    yin = next(r for r in rows if r["method"] == "yin")
    assert res.ran and float(yin["agree_25c_pct"]) > 90


def test_e5_ornament(tmp_path) -> None:
    res = ornament_eval.run(CFG, tmp_path, n_phrases=2)
    metrics = {r["metric"]: float(r["value"]) for r in read_csv(tmp_path / "e5_ornament.csv")}
    assert metrics["glide_frames_flagged_ours"] < metrics["glide_frames_flagged_naive"]
    assert metrics["frame_accuracy_settled"] > 80
    assert res.files


def test_e6_sensitivity(tmp_path) -> None:
    sensitivity.run(CFG, tmp_path, n_freqs=2, trials=1)
    rows = read_csv(tmp_path / "e6_frame_size.csv")
    lat = {int(r["frame_size"]): float(r["latency_ms"]) for r in rows if r["method"] == "yin"}
    assert lat[1024] < lat[2048] < lat[4096]
    assert len(read_csv(tmp_path / "e6_yin_threshold.csv")) == len(sensitivity.YIN_THRESHOLDS)


def test_report(tmp_path) -> None:
    results = [voice.run(CFG, tmp_path, voice_dir=tmp_path / "none")]
    write_report(results, CFG, tmp_path / "RESULTS.md", 1.0)
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "seed" in text and "drone was **off**" in text and "E4" in text
