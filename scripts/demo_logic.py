"""Stage 3 demo: a synthetic sung phrase through YIN, swara mapping and ornament logic.

Audio concepts: pitch tracking, cents relative to Sa, held-note judgement vs glides
(meend) that are deliberately NOT judged, and session statistics.

The phrase: Sa, Re, Ga, then a meend from Ga up to Pa, and Pa held 20 cents flat.

Usage (from the swara-aware folder):
    python scripts/demo_logic.py                 # Sa = D3
    python scripts/demo_logic.py --sa C#3 --mode just
    python scripts/demo_logic.py --drone-bleed -6

The drone is OFF by default, as in all evaluation scripts: a drone played through
speakers leaks into the microphone and confuses the pitch tracker. ``--drone-bleed DB``
mixes the drone into the "microphone" signal at DB relative to the voice, to show that
effect (use headphones in real practice).

Saves results/demo_logic_trace.png, demo_logic_accuracy.png and demo_logic_track.csv.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from swara_aware import dsp, synth  # noqa: E402
from swara_aware.config import Config  # noqa: E402
from swara_aware.drone import TanpuraDrone  # noqa: E402
from swara_aware.pitch import YinEstimator, estimate_track  # noqa: E402
from swara_aware.session import Session, analyse_track  # noqa: E402
from swara_aware.swara import cents_to_freq, nearest_swara, note_to_hz, swara_frequency  # noqa: E402

# (description, duration s, start cents, end cents); None = silence
PHRASE = [
    ("silence", 0.30, None, None),
    ("Sa", 0.80, 0, 0),
    ("step", 0.06, 0, 200),
    ("Re", 0.70, 200, 200),
    ("step", 0.06, 200, 400),
    ("Ga", 0.60, 400, 400),
    ("meend Ga->Pa", 0.45, 400, 700),
    ("Pa (20 c flat)", 1.00, 680, 680),
    ("silence", 0.30, None, None),
]


def phrase_contour(sa_hz: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Per-sample f0 (NaN = silence) for PHRASE, with a slight natural pitch wobble."""
    parts = []
    for _, dur, c0, c1 in PHRASE:
        n = int(round(dur * sr))
        if c0 is None:
            parts.append(np.full(n, np.nan))
            continue
        # Glides are exponential in Hz = linear in cents.
        parts.append(np.linspace(c0, c1, n))
    cents = np.concatenate(parts)
    # Slow random wobble of about +-3 cents: real voices are never perfectly steady.
    wobble = np.cumsum(rng.standard_normal(len(cents)))
    wobble = np.convolve(wobble - wobble.mean(), np.ones(4410) / 4410, mode="same")
    wobble *= 3.0 / (np.max(np.abs(wobble)) + 1e-12)
    return cents_to_freq(cents + wobble, sa_hz)


def segment_table(session: Session) -> None:
    """Print one row per run of equal state, with the judged result for held notes."""
    notes = session.held_notes()
    rows, start = [], 0
    frames = session.frames
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i].state != frames[start].state:
            rows.append(frames[start:i])
            start = i
    print(f"\n{'start':>6s} {'end':>6s}  {'state':9s} {'swara':6s} {'mean err':>9s}  judged")
    for run in rows:
        t0, t1 = run[0].time, run[-1].time
        if t1 - t0 + session.frame_seconds < 0.04:
            continue  # hide 1-3 frame transients in the printout
        cents = np.array([fr.cents for fr in run])
        if np.all(np.isnan(cents)):
            print(f"{t0:6.2f} {t1:6.2f}  {run[0].state:9s}")
            continue
        sw = nearest_swara(float(cents_to_freq(np.nanmedian(cents), session.sa_hz)),
                           session.sa_hz, session.mode)
        judged = [n for n in notes if n.start >= t0 - 1e-9 and n.end <= t1 + 1e-9]
        verdict = (f"{judged[0].status} ({judged[0].mean_error:+.1f} c)" if judged
                   else "no (not a held note)")
        print(f"{t0:6.2f} {t1:6.2f}  {run[0].state:9s} {sw.label:6s} "
              f"{np.nanmedian(cents) - (sw.cents - sw.cents_error):+9.1f}  {verdict}")


def main() -> None:
    """Synthesise the phrase, analyse it, print tables, save plots and CSV."""
    parser = argparse.ArgumentParser(description="Stage 3 logic demo for Swara-Aware.")
    parser.add_argument("--sa", default="D3", help="Sa as a note name (default D3)")
    parser.add_argument("--mode", choices=["equal", "just"], default=None)
    parser.add_argument("--drone-bleed", type=float, default=None, metavar="DB",
                        help="mix the drone into the mic signal at DB re the voice")
    args = parser.parse_args()

    cfg = Config() if args.mode is None else replace(Config(), tuning_mode=args.mode)
    sr = cfg.sample_rate
    sa_hz = note_to_hz(args.sa, cfg.a4_hz)
    rng = np.random.default_rng(7)
    print(f"Sa = {args.sa} = {sa_hz:.2f} Hz, tuning: {cfg.tuning_mode}. "
          f"Pa should be {swara_frequency('Pa', sa_hz, mode=cfg.tuning_mode):.2f} Hz.")

    voice = synth.voice_from_contour(phrase_contour(sa_hz, sr, rng), sr)
    mic = dsp.add_noise(voice, 35.0, noise=synth.pink_noise(len(voice) / sr, sr, rng=rng))
    if args.drone_bleed is not None:
        # Simulates a drone played on speakers and picked up by the mic. Don't do this
        # in practice: use headphones.
        drone = TanpuraDrone.from_config(cfg, sa_hz).render(len(voice))
        gain = np.sqrt(dsp.signal_power(voice) / dsp.signal_power(drone)
                       * 10 ** (args.drone_bleed / 10))
        mic = mic + gain * drone
        print(f"WARNING: drone mixed into the mic signal at {args.drone_bleed:+.0f} dB "
              "re the voice (simulated speaker bleed).")

    yin = YinEstimator(sr, cfg.fmin, cfg.fmax, cfg.yin_threshold, cfg.silence_threshold_dbfs)
    times, f0, conf = estimate_track(mic, yin, cfg.frame_size, cfg.hop_size)
    session = analyse_track(times, f0, conf, cfg, sa_hz)

    segment_table(session)
    summary = session.summary()
    print("\nPer swara (held notes only):")
    print(f"  {'swara':6s} {'held s':>7s} {'mean err':>9s} {'in tune':>8s}")
    for label, s in summary["per_swara"].items():
        print(f"  {label:6s} {s['seconds']:7.2f} {s['mean_cents']:+9.1f} {s['in_tune_pct']:7.0f}%")
    secs = summary["seconds_by_state"]
    print(f"\nTime by state: " + ", ".join(f"{k} {v:.2f} s" for k, v in secs.items()))
    pct = summary["in_tune_pct"]
    print(f"Judged (held) time: {summary['judged_seconds']:.2f} s, in tune "
          f"(±{cfg.in_tune_cents:.0f} c): {pct:.0f}%" if pct is not None else "Nothing judged.")

    out = ROOT / "results"
    session.export_csv(out / "demo_logic_track.csv")
    suffix = "" if args.drone_bleed is None else f" (drone bleed {args.drone_bleed:+.0f} dB)"
    session.plot_trace(out / "demo_logic_trace.png",
                       f"Sa-Re-Ga, meend Ga→Pa, Pa held 20¢ flat{suffix}")
    session.plot_accuracy(out / "demo_logic_accuracy.png")
    print(f"\nSaved {out / 'demo_logic_trace.png'}, demo_logic_accuracy.png, "
          "demo_logic_track.csv")


if __name__ == "__main__":
    main()
