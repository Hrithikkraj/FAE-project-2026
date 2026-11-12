"""Practice-session statistics: which swaras were held, for how long, and how in tune.

Audio concepts: cents error against a target pitch, tolerance bands (in tune / close),
and time-weighted statistics over frames (each frame lasts one hop).

Only *held notes* are judged: runs of "stable" frames lasting at least
``stable_min_ms``. Each held note is compared with the swara nearest to its median
pitch, so a slight drift during the note does not flip the target. Glides and
ornaments are recorded and plotted but never counted as errors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .ornament import FrameState, OrnamentTracker
from .swara import (SWARA_NAMES, cents_to_freq, nearest_swara, swara_label,
                    swara_positions, tuning_status)

# Fixed colour per ornament state (categorical palette, fixed order).
STATE_COLORS = {"stable": "#2a78d6", "glide": "#eb6834", "ornament": "#1baf7a"}


@dataclass(frozen=True)
class HeldNote:
    """One judged held note.

    Attributes:
        start, end: Time of the first and last frame (s).
        label: Swara with octave marks, e.g. "Pa" or "Sa'".
        target_cents: Target pitch relative to Sa.
        errors: Per-frame signed cents error (positive = sharp).
        in_tune_frac: Fraction of frames within ``in_tune_cents``.
        status: Status of the mean error ("in_tune", "close", "sharp", "flat").
    """

    start: float
    end: float
    label: str
    target_cents: float
    errors: np.ndarray
    in_tune_frac: float
    status: str

    @property
    def mean_error(self) -> float:
        """Mean signed cents error over the note."""
        return float(np.mean(self.errors))


class Session:
    """Collects FrameStates from an OrnamentTracker and judges the held notes."""

    def __init__(self, cfg: Config, sa_hz: float, mode: str | None = None) -> None:
        """Start an empty session for Sa = ``sa_hz`` (tuning mode defaults to cfg)."""
        self.cfg = cfg
        self.sa_hz = sa_hz
        self.mode = mode or cfg.tuning_mode
        self.frames: list[FrameState] = []

    @property
    def frame_seconds(self) -> float:
        """Duration represented by one frame: one hop."""
        return self.cfg.hop_size / self.cfg.sample_rate

    def add(self, frame: FrameState) -> None:
        """Record one labelled frame (call once per hop in live use)."""
        self.frames.append(frame)

    def _stable_runs(self) -> list[list[FrameState]]:
        """Consecutive runs of "stable" frames (a gap or other state ends a run)."""
        runs: list[list[FrameState]] = []
        current: list[FrameState] = []
        for fr in self.frames:
            contiguous = current and fr.time - current[-1].time <= 1.5 * self.frame_seconds
            if fr.state == "stable" and (not current or contiguous):
                current.append(fr)
            else:
                if current:
                    runs.append(current)
                current = [fr] if fr.state == "stable" else []
        if current:
            runs.append(current)
        return runs

    def _trim_edges(self, run: list[FrameState]) -> list[FrameState]:
        """Drop leading/trailing frames far from the run's median (glide tails).

        "stable" is decided over a trailing window, so a run can start with the last
        few cents of the arriving glide and end with the first few cents of the next
        one. Those edge frames (more than half of ``stable_max_range_cents`` from the
        median) are not part of the held note and must not be judged. Frames inside
        the note are always kept, so a note held off-pitch is still judged as such.
        """
        cents = np.array([fr.cents for fr in run])
        keep = np.abs(cents - np.median(cents)) <= self.cfg.stable_max_range_cents / 2
        if not keep.any():
            return []
        first = int(np.argmax(keep))
        last = len(keep) - int(np.argmax(keep[::-1]))
        return run[first:last]

    def held_notes(self) -> list[HeldNote]:
        """All stable runs lasting at least ``stable_min_ms``, judged against a swara."""
        notes = []
        min_frames = int(np.ceil(self.cfg.stable_min_ms / 1000.0 / self.frame_seconds))
        for run in self._stable_runs():
            run = self._trim_edges(run)
            if len(run) < min_frames:
                continue
            cents = np.array([fr.cents for fr in run])
            target = nearest_swara(float(cents_to_freq(np.median(cents), self.sa_hz)),
                                   self.sa_hz, self.mode,
                                   self.cfg.in_tune_cents, self.cfg.close_cents)
            target_cents = target.cents - target.cents_error
            errors = cents - target_cents
            notes.append(HeldNote(
                start=run[0].time, end=run[-1].time, label=target.label,
                target_cents=target_cents, errors=errors,
                in_tune_frac=float(np.mean(np.abs(errors) <= self.cfg.in_tune_cents)),
                status=tuning_status(float(np.mean(errors)), self.cfg.in_tune_cents,
                                     self.cfg.close_cents),
            ))
        return notes

    def per_swara(self) -> dict[str, dict[str, float]]:
        """Per swara label (low to high): seconds held, mean signed error, % in tune."""
        grouped: dict[str, list[np.ndarray]] = {}
        targets: dict[str, float] = {}
        for note in self.held_notes():
            grouped.setdefault(note.label, []).append(note.errors)
            targets[note.label] = note.target_cents
        out = {}
        for label in sorted(grouped, key=targets.__getitem__):
            errs = np.concatenate(grouped[label])
            out[label] = {
                "target_cents": targets[label],
                "seconds": len(errs) * self.frame_seconds,
                "mean_cents": float(np.mean(errs)),
                "in_tune_pct": 100.0 * float(np.mean(np.abs(errs) <= self.cfg.in_tune_cents)),
            }
        return out

    def summary(self) -> dict[str, Any]:
        """Plain-dict summary of the session (safe to print or save as JSON)."""
        notes = self.held_notes()
        judged = np.concatenate([n.errors for n in notes]) if notes else np.zeros(0)
        states = [fr.state for fr in self.frames]
        seconds = {s: states.count(s) * self.frame_seconds
                   for s in ("silent", "stable", "glide", "ornament")}
        return {
            "sa_hz": self.sa_hz,
            "mode": self.mode,
            "duration_s": len(self.frames) * self.frame_seconds,
            "seconds_by_state": seconds,
            "judged_seconds": len(judged) * self.frame_seconds,
            "held_notes": len(notes),
            "in_tune_pct": 100.0 * float(np.mean(np.abs(judged) <= self.cfg.in_tune_cents))
            if len(judged) else None,
            "mean_abs_cents": float(np.mean(np.abs(judged))) if len(judged) else None,
            "per_swara": self.per_swara(),
        }

    def export_csv(self, path: str | Path) -> None:
        """Write every frame: time, f0, cents, swara, state (empty cells when silent)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "f0", "cents", "swara", "state"])
            for fr in self.frames:
                if np.isnan(fr.cents):
                    writer.writerow([f"{fr.time:.4f}", "", "", "", fr.state])
                    continue
                sw = nearest_swara(float(cents_to_freq(fr.cents, self.sa_hz)),
                                   self.sa_hz, self.mode)
                f0 = "" if np.isnan(fr.f0_hz) else f"{fr.f0_hz:.3f}"
                writer.writerow([f"{fr.time:.4f}", f0, f"{fr.cents:.2f}", sw.label, fr.state])

    # ---------- plots ----------

    def plot_trace(self, path: str | Path, title: str = "Pitch trace") -> None:
        """Cents vs time with swara lines, in-tune bands, states and judged notes."""
        import matplotlib.pyplot as plt

        times = np.array([fr.time for fr in self.frames])
        cents = np.array([fr.cents for fr in self.frames])
        states = np.array([fr.state for fr in self.frames])
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)

        voiced = cents[~np.isnan(cents)]
        lo = np.floor((voiced.min() - 60) / 100) * 100 if len(voiced) else -100
        hi = np.ceil((voiced.max() + 60) / 100) * 100 if len(voiced) else 1300
        positions = swara_positions(self.mode)
        for octave in range(int(np.floor(lo / 1200)), int(np.ceil(hi / 1200)) + 1):
            for idx, pos in enumerate(positions):
                c = 1200 * octave + pos
                if not lo <= c <= hi:
                    continue
                ax.axhspan(c - self.cfg.in_tune_cents, c + self.cfg.in_tune_cents,
                           color="#e6e6e3", lw=0, zorder=0)
                ax.axhline(c, color="#b5b4ad", lw=0.6, ls="--", zorder=1)
                ax.text(1.005, c, swara_label(SWARA_NAMES[idx], octave), va="center",
                        fontsize=8, color="#52514e", transform=ax.get_yaxis_transform())

        ax.plot(times, cents, color="#b5b4ad", lw=1.0, zorder=2)  # whole contour
        for state, color in STATE_COLORS.items():
            ax.plot(times, np.where(states == state, cents, np.nan), color=color, lw=2.2,
                    zorder=3, label=state + (" (judged if held)" if state == "stable"
                                             else " (not judged)"))
        for note in self.held_notes():
            ax.annotate(f"{note.label} {round(note.mean_error):+d}¢",
                        ((note.start + note.end) / 2, note.target_cents),
                        xytext=(0, 12), textcoords="offset points", ha="center",
                        fontsize=8, color="#0b0b0b")
        ax.set(title=title, xlabel="Time (s)", ylabel="Cents from Sa", ylim=(lo, hi))
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def plot_accuracy(self, path: str | Path, title: str = "In-tune time per swara") -> None:
        """Bar chart: % of held time in tune per swara, annotated with the mean error."""
        import matplotlib.pyplot as plt

        stats = self.per_swara()
        order = list(stats)  # already sorted low to high
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(order) + 2), 4),
                               constrained_layout=True)
        values = [stats[lab]["in_tune_pct"] for lab in order]
        ax.bar(order, values, color=STATE_COLORS["stable"], width=0.6)
        for i, lab in enumerate(order):
            s = stats[lab]
            ax.text(i, values[i] + 2, f"{s['in_tune_pct']:.0f}%\n{round(s['mean_cents']):+d}¢ avg",
                    ha="center", va="bottom", fontsize=8, color="#0b0b0b")
        ax.set(title=f"{title} (±{self.cfg.in_tune_cents:.0f}¢)", ylabel="% of held time in tune",
               ylim=(0, 125))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)


def analyse_track(times: np.ndarray, f0: np.ndarray, confidence: np.ndarray,
                  cfg: Config, sa_hz: float, mode: str | None = None) -> Session:
    """Run the ornament tracker over a pitch track and collect a Session."""
    tracker = OrnamentTracker(cfg, sa_hz)
    session = Session(cfg, sa_hz, mode)
    for frame in tracker.process(times, f0, confidence):
        session.add(frame)
    return session
