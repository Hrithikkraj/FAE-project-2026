"""E4: real voice recordings compared with the librosa pYIN reference.

Audio concepts: agreement in cents between two pitch tracks, octave disagreements, and
voicing (which frames each method considers sung). pYIN is a well-regarded
probabilistic tracker but it is a REFERENCE, not perfect ground truth: when we disagree
with it, either side may be wrong.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..audio_io import load_wav
from ..config import Config
from ..pitch import estimate_track, make_estimators
from .common import (DATA_DIR, GROSS_CENTS, METHODS, WITHIN_CENTS, ExperimentResult,
                     cents_diff, label, md_table, write_csv)

DISCLAIMER = ("**pYIN is used as a reference, not as perfect ground truth.** "
              "Disagreement means one of the two trackers is wrong, not necessarily ours.")


def compare_tracks(ours: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Agreement statistics between one method's f0 track and the reference track."""
    n = min(len(ours), len(ref))
    ours, ref = ours[:n], ref[:n]
    ours_v, ref_v = ~np.isnan(ours), ~np.isnan(ref)
    both = ours_v & ref_v
    diff = np.abs(cents_diff(ours[both], ref[both]))
    return {
        "frames": n,
        "both_voiced": int(both.sum()),
        "agree_25c_pct": 100.0 * float(np.mean(diff <= WITHIN_CENTS)) if both.any() else np.nan,
        "octave_disagree_pct": 100.0 * float(np.mean(diff > GROSS_CENTS)) if both.any() else np.nan,
        "median_diff_cents": float(np.median(diff)) if both.any() else np.nan,
        "voiced_only_ours_pct": 100.0 * float(np.mean(ours_v & ~ref_v)),
        "voiced_only_ref_pct": 100.0 * float(np.mean(ref_v & ~ours_v)),
    }


def run(cfg: Config, out_dir: Path, voice_dir: Path | None = None) -> ExperimentResult:
    """Run E4 on every WAV in data/voice/ (or ``voice_dir``); skip cleanly if none."""
    voice_dir = voice_dir or DATA_DIR / "voice"
    files = sorted(voice_dir.glob("*.wav")) if voice_dir.is_dir() else []
    if not files:
        md = (f"{DISCLAIMER}\n\n**Not run: no recordings found in `data/voice/`.** "
              "To run it, record WAV files of sustained notes, a scale (Sa to taar Sa and "
              "back) and a phrase with a meend, put them in `data/voice/`, and re-run "
              "`python scripts/run_all_experiments.py --only E4`.")
        return ExperimentResult("E4", "Real voice vs pYIN reference", md,
                                "Not run: no voice recordings were available.", [], ran=False)
    try:
        from ..pitch.reference import reference_track
    except ImportError:
        md = f"{DISCLAIMER}\n\nNot run: librosa is not installed."
        return ExperimentResult("E4", "Real voice vs pYIN reference", md,
                                "Not run: librosa missing.", [], ran=False)

    rows = []
    for path in files:
        x, sr = load_wav(path)
        c = cfg if sr == cfg.sample_rate else replace(cfg, sample_rate=sr)
        _, ref, _ = reference_track(x, sr, c.fmin, c.fmax, c.frame_size, c.hop_size)
        for est in make_estimators(c):
            _, f0, _ = estimate_track(x, est, c.frame_size, c.hop_size)
            rows.append({"file": path.name, "method": est.name, **compare_tracks(f0, ref)})
    write_csv(out_dir / "e4_voice.csv", rows)

    table = md_table(["File", "Method", "Agree ≤25 c (%)", "Octave disagree (%)",
                      "Median diff (c)", "Voiced only by us (%)", "Voiced only by pYIN (%)"],
                     [(r["file"], label(r["method"]), r["agree_25c_pct"],
                       r["octave_disagree_pct"], r["median_diff_cents"],
                       r["voiced_only_ours_pct"], r["voiced_only_ref_pct"]) for r in rows], 1)
    md = (f"{DISCLAIMER}\n\n{len(files)} recording(s) from `data/voice/`. Agreement and "
          "octave figures are over frames that both trackers call voiced; voicing "
          "differences are a % of all frames.\n\n" + table)
    return ExperimentResult("E4", "Real voice vs pYIN reference", md, finding(rows),
                            ["e4_voice.csv"])


def finding(rows: list[dict]) -> str:
    mean_agree = {m: np.nanmean([r["agree_25c_pct"] for r in rows if r["method"] == m])
                  for m in METHODS}
    best = max(METHODS, key=lambda m: mean_agree[m])
    return (f"Averaged over the recordings, agreement with pYIN within 25 cents was "
            f"{', '.join(f'{label(m)} {mean_agree[m]:.0f}%' for m in METHODS)}; "
            f"{label(best)} agreed most. pYIN is only a reference.")
