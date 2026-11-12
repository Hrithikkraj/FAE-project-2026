"""E3: computing cost and latency.

Audio concepts: the real-time budget (one hop = hop_size / sample_rate seconds), frame
length as buffering delay, and the end-to-end chain microphone -> sound-card buffer ->
analysis frame -> processing -> UI refresh -> screen.

Measured here: processing time per frame for each method; the *detection delay* after
a sudden note change, streamed through the app's real ProcessingWorker (audio time from
the change until the displayed swara changes, until it is "stable", until it is judged);
and the input latency PortAudio reports for the default microphone.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import synth
from ..app.sources import ArraySource
from ..app.worker import FrameResult, ProcessingWorker
from ..config import Config
from ..pitch import make_estimators
from ..swara import cents_to_freq
from .common import (METHOD_STYLE, METHODS, ExperimentResult, label, log_spaced, md_table,
                     plot_style, rng_for, write_csv)

UI_POLL_MS = 33.0            # the app's QTimer interval
DISPLAY_REFRESH_MS = 16.7    # a 60 Hz screen
SA_HZ = 220.0
STEP_AT_S = 1.0


def time_per_frame(cfg: Config, rng: np.random.Generator, n_frames: int = 600) -> dict:
    """Mean / median / 95th-percentile microseconds per frame, on voiced frames."""
    sr, n = cfg.sample_rate, cfg.frame_size
    freqs = rng.choice(log_spaced(80, 900, 40), n_frames)
    frames = [synth.voice_like_tone(f, n / sr, sr) for f in freqs]
    out = {}
    for est in make_estimators(cfg):
        est.estimate(frames[0])  # warm-up
        times = []
        for frame in frames:
            t0 = time.perf_counter()
            est.estimate(frame)
            times.append(1e6 * (time.perf_counter() - t0))
        t = np.array(times)
        out[est.name] = {"mean_us": float(t.mean()), "median_us": float(np.median(t)),
                         "p95_us": float(np.percentile(t, 95))}
    return out


def detection_delays(cfg: Config, method: str) -> dict[str, float]:
    """Stream Sa (1 s) -> Pa (1 s) through the worker; delays after the step, in ms.

    A frame's result is available when its last sample has arrived, i.e. at
    frame-centre time + frame_size / 2 / sample_rate.
    """
    sr = cfg.sample_rate
    t = np.arange(int(2 * STEP_AT_S * sr)) / sr
    f0 = cents_to_freq(np.where(t < STEP_AT_S, 0.0, 700.0), SA_HZ)
    source = ArraySource(synth.voice_from_contour(f0, sr), sr, cfg.hop_size, realtime=False)
    worker = ProcessingWorker(source, cfg, SA_HZ, estimator=method)
    source.start()
    worker.start()
    worker.join(timeout=60)
    source.stop()
    frames = []
    while not worker.results.empty():
        item = worker.results.get()
        if isinstance(item, FrameResult):
            frames.append(item)
    half = cfg.frame_size / 2 / sr

    def first(cond) -> float:  # noqa: ANN001
        for r in frames:
            if r.time + half > STEP_AT_S and cond(r):
                return 1000.0 * (r.time + half - STEP_AT_S)
        return np.nan

    return {"display_ms": first(lambda r: r.swara == "Pa"),
            "stable_ms": first(lambda r: r.swara == "Pa" and r.state == "stable"),
            "judged_ms": first(lambda r: r.swara == "Pa" and r.state == "stable"
                               and r.held_ms >= cfg.stable_min_ms)}


def device_input_latency_ms() -> tuple[float, float, str]:
    """PortAudio's reported (low, high) input latency for the default microphone."""
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        return (1000.0 * dev["default_low_input_latency"],
                1000.0 * dev["default_high_input_latency"], dev["name"])
    except Exception as exc:  # no device / no PortAudio
        return np.nan, np.nan, f"unavailable ({exc})"


def run(cfg: Config, out_dir: Path, n_frames: int = 600) -> ExperimentResult:
    """Run E3 and write e3_latency.csv and e3_latency.png."""
    rng = rng_for("E3")
    cost = time_per_frame(cfg, rng, n_frames)
    frame_ms = 1000.0 * cfg.frame_size / cfg.sample_rate
    hop_ms = cfg.hop_duration_ms
    dev_low, dev_high, dev_name = device_input_latency_ms()
    rows = []
    for m in METHODS:
        d = detection_delays(cfg, m)
        proc_ms = cost[m]["mean_us"] / 1000.0
        rows.append({"method": m, **cost[m], "budget_pct": 100.0 * proc_ms / hop_ms,
                     "frame_ms": frame_ms, "hop_ms": hop_ms,
                     "theoretical_latency_ms": frame_ms + hop_ms + proc_ms,
                     "detect_display_ms": d["display_ms"], "detect_stable_ms": d["stable_ms"],
                     "detect_judged_ms": d["judged_ms"],
                     "est_mic_to_display_ms": dev_low + d["display_ms"] + proc_ms
                     + UI_POLL_MS / 2 + DISPLAY_REFRESH_MS / 2})
    write_csv(out_dir / "e3_latency.csv", rows)
    plot(rows, hop_ms, out_dir / "e3_latency.png")

    t1 = md_table(["Method", "Mean (µs)", "Median (µs)", "95th pct (µs)", "% of hop budget",
                   "Theoretical latency (ms)"],
                  [(label(r["method"]), r["mean_us"], r["median_us"], r["p95_us"],
                    r["budget_pct"], r["theoretical_latency_ms"]) for r in rows], 1)
    t2 = md_table(["Method", "Swara shown (ms)", "Marked stable (ms)", "Judged (ms)",
                   "Estimated mic→display (ms)"],
                  [(label(r["method"]), r["detect_display_ms"], r["detect_stable_ms"],
                    r["detect_judged_ms"], r["est_mic_to_display_ms"]) for r in rows], 0)
    md = (f"Frame {cfg.frame_size} samples = {frame_ms:.1f} ms, hop {cfg.hop_size} = "
          f"{hop_ms:.1f} ms (the real-time budget per frame), {cfg.sample_rate} Hz. "
          f"Processing time is measured on {n_frames} voiced frames on this laptop (Python, "
          f"single thread). Theoretical latency = frame + hop + processing.\n\n{t1}\n\n"
          f"**Detection delay**: a held Sa switches instantly to Pa, and the signal is "
          f"streamed through the app's own `ProcessingWorker`. Times are audio time from "
          f"the switch until the display shows Pa, until Pa is marked stable, and until it "
          f"is judged (held for {cfg.stable_min_ms:.0f} ms).\n\n{t2}\n\n"
          f"**Estimated mic→display** = input-device latency reported by PortAudio for "
          f"'{dev_name}' ({dev_low:.1f} ms low / {dev_high:.1f} ms high), plus the "
          f"display detection delay, plus processing, plus half the UI poll interval "
          f"({UI_POLL_MS:.0f} ms), plus half a 60 Hz screen refresh. This is an estimate "
          f"added up from parts, **not** a physical measurement. To measure it physically: "
          f"film the laptop screen and your mouth with a phone in slow motion (240 fps), "
          f"sing a sudden loud note or clap a pitched instrument, and count the video "
          f"frames between the sound starting and the trace changing (each frame = 4.2 ms). "
          f"Repeat 10 times and take the median.\n\n![E3](e3_latency.png)")
    return ExperimentResult("E3", "Latency and computing cost", md, finding(rows),
                            ["e3_latency.csv", "e3_latency.png"])


def finding(rows: list[dict]) -> str:
    worst = max(rows, key=lambda r: r["mean_us"])
    disp = [r["detect_display_ms"] for r in rows]
    judged = [r["detect_judged_ms"] for r in rows]
    est = [r["est_mic_to_display_ms"] for r in rows]
    return (f"Every method costs well under the {rows[0]['hop_ms']:.1f} ms budget (slowest: "
            f"{label(worst['method'])}, {worst['mean_us']:.0f} µs = {worst['budget_pct']:.1f}%), "
            "so computing power is not the bottleneck: buffering is. After a sudden note "
            f"change the new swara appears after {np.nanmin(disp):.0f}–{np.nanmax(disp):.0f} ms "
            f"of audio and is judged after {np.nanmin(judged):.0f}–{np.nanmax(judged):.0f} ms. "
            f"The estimated mic-to-display delay is about {np.nanmin(est):.0f}–{np.nanmax(est):.0f} ms "
            "(an estimate from components, not measured physically).")


def plot(rows: list[dict], hop_ms: float, path: Path) -> None:
    """Per-frame cost (log scale) and detection delays per method."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = [label(r["method"]) for r in rows]
    colors = [METHOD_STYLE[r["method"]][0] for r in rows]
    ax = axes[0]
    med = [r["median_us"] for r in rows]
    p95 = [r["p95_us"] for r in rows]
    ax.bar(names, med, color=colors, width=0.6)
    ax.errorbar(names, med, yerr=[np.zeros(len(med)), np.array(p95) - np.array(med)],
                fmt="none", ecolor="#52514e", capsize=4, lw=1)
    ax.axhline(hop_ms * 1000, color="#52514e", ls="--", lw=1)
    ax.text(len(names) - 0.5, hop_ms * 1000 * 0.8, f"real-time budget {hop_ms:.1f} ms",
            ha="right", va="top", fontsize=9, color="#52514e")
    ax.set_yscale("log")
    ax.set(title="Processing time per frame (µs, median; whisker = 95th pct)")
    plot_style(ax)

    ax = axes[1]
    keys = [("detect_display_ms", "swara shown"), ("detect_stable_ms", "marked stable"),
            ("detect_judged_ms", "judged")]
    x = np.arange(len(rows))
    shades = ["#9fc3ee", "#5b9be0", "#2a78d6"]  # one hue, light -> dark (sequence of stages)
    for i, (k, lab) in enumerate(keys):
        ax.bar(x + (i - 1) * 0.26, [r[k] for r in rows], 0.24, color=shades[i], label=lab)
    ax.set_xticks(x, names)
    ax.set(title="Delay after a sudden note change (ms of audio)")
    ax.legend(fontsize=9, frameon=False)
    plot_style(ax)
    fig.suptitle("E3: computing cost and latency", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
