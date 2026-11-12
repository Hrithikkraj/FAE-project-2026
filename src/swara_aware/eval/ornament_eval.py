"""E5: does the ornament-aware logic judge the right things?

Audio concepts: pitch contours with known structure (held notes, meends, gamakas,
deliberately off-pitch notes), frame classification, and the difference between judging
only held notes and naively judging every voiced frame.

Seeded random phrases are synthesised as audio (voice-like tone with slight vibrato and
30 dB pink noise), tracked with YIN and labelled by the OrnamentTracker, exactly as in
the app. The truth for every frame comes from the phrase script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import dsp, synth
from ..config import Config
from ..pitch import YinEstimator, estimate_track
from ..session import analyse_track
from ..swara import cents_to_freq, nearest_swara
from .common import ExperimentResult, md_table, plot_style, rng_for, write_csv

SA_HZ = 196.0
SCALE = [0, 200, 400, 500, 700, 900, 1100, 1200]
CLASSES = ["silent", "stable", "glide", "ornament"]
SETTLE_S = 0.15  # "settled" frames are at least this far from any segment boundary
GLIDE_SPLIT = 200.0  # cents/s: the default glide_threshold_cents_per_s, for the breakdown
GLIDE_TYPES = {"step": "quick step (60 ms)", "meend": "meend faster than 200 c/s",
               "slow": "meend slower than 200 c/s"}


@dataclass
class Piece:
    """One scripted segment of a phrase."""

    kind: str          # silent / stable / glide / ornament
    cents: np.ndarray  # per-sample pitch (NaN when silent)
    note_id: int = -1  # index into the notes list for held notes
    offset: float = 0.0
    glide_type: str = ""  # for glides: "step", "slow" (< glide threshold) or "meend"


def random_phrase(rng: np.random.Generator, sr: int) -> tuple[list[Piece], list[dict]]:
    """Seven held notes joined by meends or quick steps, sometimes with a gamaka."""
    def n(sec: float) -> int:
        return int(round(sec * sr))

    pieces = [Piece("silent", np.full(n(0.3), np.nan))]
    notes = []
    centre = float(rng.choice(SCALE[:5]))
    offset = 0.0
    for i in range(7):
        offset = float(rng.choice([-1, 1]) * rng.uniform(30, 45)) if rng.random() < 0.3 \
            else float(rng.uniform(-5, 5))
        if i > 0:  # transition from the previous pitch to this note
            start = pieces[-1].cents[-1]
            dur = rng.uniform(0.3, 0.7) if rng.random() < 0.5 else 0.06
            speed = abs(centre + offset - start) / dur
            kind = "step" if dur < 0.1 else ("slow" if speed < GLIDE_SPLIT else "meend")
            pieces.append(Piece("glide", np.linspace(start, centre + offset, n(dur)),
                                glide_type=kind))
        dur = rng.uniform(0.6, 1.3)
        t = np.arange(n(dur)) / sr
        vib = rng.uniform(0, 8) * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 6.28))
        notes.append({"id": i, "centre": centre, "offset": offset})
        pieces.append(Piece("stable", centre + offset + vib, i, offset))
        if rng.random() < 0.25:  # gamaka around the note
            dur = rng.uniform(0.8, 1.4)
            t = np.arange(n(dur)) / sr
            depth, rate = rng.uniform(35, 60), rng.uniform(3.5, 7.0)
            pieces.append(Piece("ornament", centre + depth * np.sin(2 * np.pi * rate * t)))
        nxt = [s for s in SCALE if s != centre and abs(s - centre) <= 700]
        centre = float(rng.choice(nxt))
    pieces.append(Piece("silent", np.full(n(0.4), np.nan)))
    return pieces, notes


def run(cfg: Config, out_dir: Path, n_phrases: int = 20) -> ExperimentResult:
    """Run E5 and write e5_frames.csv, e5_notes.csv, e5_ornament.csv, e5_ornament.png."""
    rng = rng_for("E5")
    sr, hop = cfg.sample_rate, cfg.hop_size
    yin = YinEstimator(sr, cfg.fmin, cfg.fmax, cfg.yin_threshold, cfg.silence_threshold_dbfs)
    confusion = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    confusion_all = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    flag = {k: {"ours": [0, 0], "naive": [0, 0]}
            for k in ("glide", "ornament", "in_tune_hold", *GLIDE_TYPES)}
    note_rows = []

    for phrase in range(n_phrases):
        pieces, notes = random_phrase(rng, sr)
        cents = np.concatenate([p.cents for p in pieces])
        kinds = np.concatenate([[p.kind] * len(p.cents) for p in pieces])
        note_of = np.concatenate([[p.note_id] * len(p.cents) for p in pieces])
        gtype = np.concatenate([[p.glide_type] * len(p.cents) for p in pieces])
        bounds = np.cumsum([len(p.cents) for p in pieces])[:-1]
        voice = synth.voice_from_contour(cents_to_freq(cents, SA_HZ), sr)
        x = dsp.add_noise(voice, 30.0, noise=synth.pink_noise(len(voice) / sr, sr, rng=rng))

        times, f0, conf = estimate_track(x, yin, cfg.frame_size, hop)
        session = analyse_track(times, f0, conf, cfg, SA_HZ)
        pred = np.array([fr.state for fr in session.frames])
        centre_idx = np.clip(np.round(times * sr).astype(int), 0, len(cents) - 1)
        truth = kinds[centre_idx]
        dist = np.min(np.abs(centre_idx[:, None] - bounds[None, :]), axis=1) / sr
        settled = dist >= SETTLE_S

        # Ours: a frame is "flagged" if it belongs to a judged held note and is off by
        # more than in_tune_cents. Naive: every voiced frame is judged on its own.
        ours_err = np.full(len(times), np.nan)
        index_of = {round(t, 6): i for i, t in enumerate(times)}
        held = session.held_notes()
        for note in held:
            i0 = index_of[round(note.start, 6)]
            ours_err[i0:i0 + len(note.errors)] = note.errors
        naive_err = np.full(len(times), np.nan)
        for i, fr in enumerate(session.frames):
            if not np.isnan(fr.cents):
                naive_err[i] = nearest_swara(float(cents_to_freq(fr.cents, SA_HZ)), SA_HZ).cents_error
        ours_flag = np.abs(np.nan_to_num(ours_err, nan=0.0)) > cfg.in_tune_cents
        naive_flag = np.abs(np.nan_to_num(naive_err, nan=0.0)) > cfg.in_tune_cents

        for t_cls, p_cls, s in zip(truth, pred, settled):
            confusion_all[t_cls][p_cls] += 1
            if s:
                confusion[t_cls][p_cls] += 1
        frame_note = note_of[centre_idx]
        offsets = {nt["id"]: nt["offset"] for nt in notes}
        in_tune_hold = (truth == "stable") & settled & np.array(
            [abs(offsets.get(k, 99)) <= 5 for k in frame_note])
        frame_gtype = gtype[centre_idx]
        masks = [("glide", truth == "glide"), ("ornament", truth == "ornament"),
                 ("in_tune_hold", in_tune_hold)]
        masks += [(g, (truth == "glide") & (frame_gtype == g)) for g in GLIDE_TYPES]
        for key, mask in masks:
            flag[key]["ours"][0] += int(np.sum(ours_flag & mask))
            flag[key]["naive"][0] += int(np.sum(naive_flag & mask))
            flag[key]["ours"][1] += int(mask.sum())
            flag[key]["naive"][1] += int(mask.sum())

        # Note level: which judged held note covers each scripted held note?
        for nt in notes:
            frames_of_note = np.where((frame_note == nt["id"]) & (truth == "stable"))[0]
            t0, t1 = times[frames_of_note].min(), times[frames_of_note].max()
            best, best_overlap = None, 0.0
            for hn in held:
                overlap = min(t1, hn.end) - max(t0, hn.start)
                if overlap > best_overlap:
                    best, best_overlap = hn, overlap
            naive_settled = naive_err[frames_of_note[settled[frames_of_note]]]
            naive_med = float(np.nanmedian(naive_settled)) if np.any(~np.isnan(naive_settled)) else np.nan
            note_rows.append({
                "phrase": phrase, "note": nt["id"], "offset_cents": nt["offset"],
                "judged": best is not None,
                "ours_mean_err": best.mean_error if best else np.nan,
                "ours_status": best.status if best else "not judged",
                "naive_median_err": naive_med,
            })

    write_csv(out_dir / "e5_notes.csv", note_rows)
    conf_rows = [{"truth": t, **{f"pred_{p}": confusion[t][p] for p in CLASSES}} for t in CLASSES]
    write_csv(out_dir / "e5_frames.csv", conf_rows)
    summary = summarise(confusion, confusion_all, flag, note_rows, cfg)
    write_csv(out_dir / "e5_ornament.csv", [{"metric": k, "value": v} for k, v in summary.items()])
    plot(confusion, summary, out_dir / "e5_ornament.png")
    return ExperimentResult("E5", "Ornament-aware logic", markdown(confusion, summary, n_phrases),
                            finding(summary),
                            ["e5_ornament.csv", "e5_frames.csv", "e5_notes.csv", "e5_ornament.png"])


def summarise(confusion: dict, confusion_all: dict, flag: dict, notes: list[dict],
              cfg: Config) -> dict[str, float]:
    """Headline rates (all in %)."""
    def recall(c: dict, cls: str) -> float:
        total = sum(c[cls].values())
        return 100.0 * c[cls][cls] / total if total else np.nan

    def rate(key: str, who: str) -> float:
        hit, total = flag[key][who]
        return 100.0 * hit / total if total else np.nan

    def acc(c: dict) -> float:
        return 100.0 * sum(c[k][k] for k in CLASSES) / sum(sum(v.values()) for v in c.values())

    off = [r for r in notes if abs(r["offset_cents"]) >= 30]
    good = [r for r in notes if abs(r["offset_cents"]) <= 5]

    def pct(rows: list[dict], cond) -> float:  # noqa: ANN001
        return 100.0 * sum(1 for r in rows if cond(r)) / len(rows) if rows else np.nan

    return {
        "frame_accuracy_settled": acc(confusion),
        "frame_accuracy_all": acc(confusion_all),
        **{f"recall_{c}_settled": recall(confusion, c) for c in CLASSES},
        "glide_frames_flagged_ours": rate("glide", "ours"),
        "glide_frames_flagged_naive": rate("glide", "naive"),
        **{f"glide_{g}_flagged_ours": rate(g, "ours") for g in GLIDE_TYPES},
        **{f"glide_{g}_flagged_naive": rate(g, "naive") for g in GLIDE_TYPES},
        **{f"glide_{g}_frames": float(flag[g]["ours"][1]) for g in GLIDE_TYPES},
        "ornament_frames_flagged_ours": rate("ornament", "ours"),
        "ornament_frames_flagged_naive": rate("ornament", "naive"),
        "in_tune_hold_frames_flagged_ours": rate("in_tune_hold", "ours"),
        "in_tune_hold_frames_flagged_naive": rate("in_tune_hold", "naive"),
        "off_notes": float(len(off)),
        "off_notes_flagged_ours": pct(off, lambda r: r["ours_status"] in ("close", "sharp", "flat")),
        "off_notes_sharp_flat_ours": pct(off, lambda r: r["ours_status"] in ("sharp", "flat")),
        "off_notes_right_direction_ours": pct(
            off, lambda r: r["judged"] and np.sign(r["ours_mean_err"]) == np.sign(r["offset_cents"])),
        "off_notes_not_judged_ours": pct(off, lambda r: not r["judged"]),
        "off_notes_flagged_naive": pct(off, lambda r: abs(r["naive_median_err"]) > cfg.in_tune_cents),
        "in_tune_notes": float(len(good)),
        "in_tune_notes_false_alarm_ours": pct(good, lambda r: r["ours_status"] in ("close", "sharp", "flat")),
        "in_tune_notes_not_judged_ours": pct(good, lambda r: not r["judged"]),
        "in_tune_notes_false_alarm_naive": pct(good, lambda r: abs(r["naive_median_err"]) > cfg.in_tune_cents),
    }


def markdown(confusion: dict, s: dict, n_phrases: int) -> str:
    conf_table = md_table(["Truth \\ predicted"] + CLASSES + ["recall (%)"],
                          [[t] + [confusion[t][p] for p in CLASSES] + [s[f"recall_{t}_settled"]]
                           for t in CLASSES], 1)
    cmp_table = md_table(
        ["Measure", "Ours (held notes only)", "Naive (judge every frame)"],
        [("Glide frames wrongly flagged as errors (%)", s["glide_frames_flagged_ours"], s["glide_frames_flagged_naive"]),
         *[(f"  of which {GLIDE_TYPES[g]} (n={s[f'glide_{g}_frames']:.0f} frames)",
            s[f"glide_{g}_flagged_ours"], s[f"glide_{g}_flagged_naive"]) for g in GLIDE_TYPES],
         ("Ornament frames wrongly flagged as errors (%)", s["ornament_frames_flagged_ours"], s["ornament_frames_flagged_naive"]),
         ("In-tune held frames flagged (false alarm, %)", s["in_tune_hold_frames_flagged_ours"], s["in_tune_hold_frames_flagged_naive"]),
         (f"Off-pitch notes (≥30 c, n={s['off_notes']:.0f}) flagged (%)", s["off_notes_flagged_ours"], s["off_notes_flagged_naive"]),
         (f"In-tune notes (≤5 c, n={s['in_tune_notes']:.0f}) wrongly flagged (%)", s["in_tune_notes_false_alarm_ours"], s["in_tune_notes_false_alarm_naive"])], 1)
    return (f"{n_phrases} seeded random phrases: 7 held notes each (30% of them sung "
            f"30–45 cents off on purpose, the rest within ±5 cents, all with up to ±8 cents of "
            f"vibrato), joined by meends (0.3–0.7 s) or quick 60 ms steps; 25% of notes are "
            f"followed by a gamaka (3.5–7 Hz, ±35–60 cents). Audio → YIN → ornament logic. "
            f"'Settled' frames are at least {SETTLE_S * 1000:.0f} ms from any segment "
            f"boundary.\n\n**Frame classification (settled frames)**: accuracy "
            f"{s['frame_accuracy_settled']:.1f}% (all frames: {s['frame_accuracy_all']:.1f}%).\n\n"
            f"{conf_table}\n\n**Judging: ours vs a naive baseline** (a frame is 'flagged' if it is "
            f"judged and more than 15 cents off; a note is 'flagged' if it is shown as close, "
            f"sharp or flat):\n\n{cmp_table}\n\nOf the off-pitch notes, "
            f"{s['off_notes_sharp_flat_ours']:.0f}% were shown as sharp/flat (beyond 30 c) and the "
            f"rest as 'close'; the direction (sharp vs flat) was right for "
            f"{s['off_notes_right_direction_ours']:.0f}%; {s['off_notes_not_judged_ours']:.0f}% were "
            f"never judged. {s['in_tune_notes_not_judged_ours']:.0f}% of in-tune notes were never "
            f"judged (too short after trimming, or misclassified).\n\n![E5](e5_ornament.png)")


def finding(s: dict) -> str:
    weakest = min(["stable", "glide", "ornament"], key=lambda c: s[f"recall_{c}_settled"])
    why = {"ornament": "a gamaka is only recognised once a full 350 ms window has seen it "
                       "oscillate",
           "glide": "short steps and the slow ends of meends look steady to a 120 ms window",
           "stable": "vibrato and noise sometimes push a held note's 120 ms range over 25 cents"}
    return (f"Only {s['glide_frames_flagged_ours']:.1f}% of glide frames and "
            f"{s['ornament_frames_flagged_ours']:.1f}% of ornament frames were wrongly marked as "
            f"errors (meends slower than 200 c/s: {s['glide_slow_flagged_ours']:.0f}%, "
            f"because the logic treats anything slower than the glide threshold as a held "
            f"note; faster meends: {s['glide_meend_flagged_ours']:.0f}%), against "
            f"{s['glide_frames_flagged_naive']:.0f}% and "
            f"{s['ornament_frames_flagged_naive']:.0f}% for the naive judge-every-frame baseline. "
            f"{s['off_notes_flagged_ours']:.0f}% of notes sung 30–45 cents off were flagged "
            f"(naive: {s['off_notes_flagged_naive']:.0f}%), with "
            f"{s['in_tune_notes_false_alarm_ours']:.0f}% false alarms on in-tune notes. "
            f"Settled frames were classified correctly {s['frame_accuracy_settled']:.0f}% of "
            f"the time; the weakest class is {weakest} ({s[f'recall_{weakest}_settled']:.0f}% "
            f"recall), likely because {why[weakest]}.")


def plot(confusion: dict, s: dict, path: Path) -> None:
    """Row-normalised confusion matrix and ours-vs-naive bars."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    m = np.array([[confusion[t][p] for p in CLASSES] for t in CLASSES], dtype=float)
    norm = 100 * m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 60 else "#0b0b0b")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set(xlabel="predicted", ylabel="truth", title="Frame classification (settled frames, row %)")

    ax = axes[1]
    labels = ["glide frames\nflagged", "ornament frames\nflagged", "in-tune notes\nflagged",
              "off-pitch notes\nflagged"]
    ours = [s["glide_frames_flagged_ours"], s["ornament_frames_flagged_ours"],
            s["in_tune_notes_false_alarm_ours"], s["off_notes_flagged_ours"]]
    naive = [s["glide_frames_flagged_naive"], s["ornament_frames_flagged_naive"],
             s["in_tune_notes_false_alarm_naive"], s["off_notes_flagged_naive"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, ours, 0.38, color="#2a78d6", label="ours (held notes only)")
    ax.bar(x + 0.2, naive, 0.38, color="#eb6834", label="naive (every frame)")
    for xi, v in zip(x - 0.2, ours):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, naive):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set(ylim=(0, 110), ylabel="%", title="Lower is better for the first three; higher for the last")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    plot_style(ax)
    fig.suptitle("E5: ornament-aware judging vs naive judging", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
