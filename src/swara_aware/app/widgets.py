"""Display widgets: the scrolling pitch trace on a swara ladder, and the readout panel.

Audio concept shown: pitch in cents relative to Sa over time, with the in-tune band
(±in_tune_cents) around the nearest swara. These widgets only draw; they never analyse.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..config import Config
from ..swara import SWARA_NAMES, swara_label, swara_positions
from . import theme
from .worker import FrameResult

_PEN_STYLES = {"solid": Qt.SolidLine, "dash": Qt.DashLine, "dot": Qt.DotLine}


class PitchTraceWidget(pg.PlotWidget):
    """Last ``window_s`` seconds of pitch (cents from Sa) drawn over swara lines."""

    def __init__(self, cfg: Config, window_s: float = 10.0) -> None:
        super().__init__(background=theme.BG)
        self.cfg = cfg
        self.window_s = window_s
        n = int(window_s * cfg.sample_rate / cfg.hop_size) + 64
        self._t: deque[float] = deque(maxlen=n)
        self._c: deque[float] = deque(maxlen=n)
        self._s: deque[str] = deque(maxlen=n)
        self._lines: dict[float, pg.InfiniteLine] = {}
        self._target: float | None = None
        self._y_centre = 500.0

        self.hideButtons()
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().hideAxis("left")
        bottom = self.getPlotItem().getAxis("bottom")
        bottom.setPen(pg.mkPen(theme.FAINT))
        bottom.setTextPen(pg.mkPen(theme.MUTED))
        bottom.setLabel("seconds", color=theme.MUTED)
        self.setXRange(-window_s, 0, padding=0)

        accent = QColor(theme.ACCENT)
        accent.setAlpha(40)
        self._band = pg.LinearRegionItem(orientation="horizontal", movable=False,
                                         brush=pg.mkBrush(accent), pen=pg.mkPen(None))
        self._band.setZValue(-10)
        self._band.hide()
        self.addItem(self._band)

        self._curves = {}
        for state, (color, width, style) in theme.TRACE_STYLES.items():
            pen = pg.mkPen(color, width=width, style=_PEN_STYLES[style])
            curve = pg.PlotDataItem(pen=pen, connect="finite", antialias=True)
            curve.setZValue(10)
            self.addItem(curve)
            self._curves[state] = curve

    def set_ladder(self, mode: str) -> None:
        """(Re)draw the dashed swara lines for three octaves (plus margins)."""
        for line in self._lines.values():
            self.removeItem(line)
        self._lines.clear()
        self._target = None
        font = QFont()
        font.setPointSize(10)
        for octave in range(-2, 3):
            for index, pos in enumerate(swara_positions(mode)):
                cents = 1200.0 * octave + pos
                is_sa = index == 0
                pen = pg.mkPen(theme.MUTED if is_sa else theme.FAINT,
                               width=1.2 if is_sa else 1, style=Qt.DashLine)
                line = pg.InfiniteLine(pos=cents, angle=0, pen=pen, movable=False,
                                       label=swara_label(SWARA_NAMES[index], octave),
                                       labelOpts={"position": 0.015, "color": theme.MUTED,
                                                  "anchors": [(0, 0.5), (0, 0.5)]})
                line.label.setFont(font)
                line.setZValue(-5)
                self.addItem(line)
                self._lines[round(cents, 3)] = line

    def clear_trace(self) -> None:
        """Forget the drawn history (e.g. after Sa changes)."""
        self._t.clear()
        self._c.clear()
        self._s.clear()

    def push(self, r: FrameResult) -> None:
        """Add one frame to the history (cheap; drawing happens in :meth:`redraw`)."""
        self._t.append(r.time)
        self._c.append(r.cents)
        self._s.append(r.state)

    def redraw(self, latest: FrameResult | None) -> None:
        """Redraw curves, band, highlighted target and follow the singer vertically."""
        if not self._t:
            return
        t = np.array(self._t) - self._t[-1]
        c = np.array(self._c)
        s = np.array(self._s)
        for state, curve in self._curves.items():
            # Draw a frame in this style if it, or its neighbour, has this state, so
            # the segments of different styles join up without gaps.
            mask = s == state
            mask[1:] |= mask[:-1].copy()
            curve.setData(t, np.where(mask, c, np.nan))

        target = None if latest is None or np.isnan(latest.target_cents) else latest.target_cents
        self._highlight(target)
        if target is None:
            self._band.hide()
        else:
            tol = self.cfg.in_tune_cents
            self._band.setRegion((target - tol, target + tol))
            self._band.show()

        recent = c[t > -2.0]
        recent = recent[~np.isnan(recent)]
        if len(recent):
            self._y_centre += 0.15 * (float(np.median(recent)) - self._y_centre)
        self.setYRange(self._y_centre - 750, self._y_centre + 750, padding=0)

    def _highlight(self, target: float | None) -> None:
        """Make the current target swara's line saffron; restore the previous one."""
        key = None if target is None else round(target, 3)
        if key == self._target:
            return
        if self._target in self._lines:
            old = self._lines[self._target]
            is_sa = abs(self._target % 1200) < 1e-6
            old.setPen(pg.mkPen(theme.MUTED if is_sa else theme.FAINT,
                                width=1.2 if is_sa else 1, style=Qt.DashLine))
            old.label.setColor(theme.MUTED)
        if key in self._lines:
            self._lines[key].setPen(pg.mkPen(theme.ACCENT, width=2, style=Qt.DashLine))
            self._lines[key].label.setColor(theme.ACCENT)
        self._target = key


class CentsMeter(QWidget):
    """Horizontal meter from -50 to +50 cents with a needle and tolerance zones."""

    RANGE = 50.0

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.value: float | None = None
        self.active = False
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, cents: float | None, active: bool) -> None:
        """Show ``cents`` (None hides the needle); ``active`` = judged state (stable)."""
        self.value, self.active = cents, active
        self.update()

    def _x(self, cents: float, rect: QRectF) -> float:
        frac = (np.clip(cents, -self.RANGE, self.RANGE) + self.RANGE) / (2 * self.RANGE)
        return rect.left() + frac * rect.width()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(8, 22, self.width() - 16, 12)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.FAINT))
        p.drawRoundedRect(track, 6, 6)
        for tol, color, alpha in [(self.cfg.close_cents, theme.STATUS["close"][1], 70),
                                  (self.cfg.in_tune_cents, theme.STATUS["in_tune"][1], 110)]:
            col = QColor(color)
            col.setAlpha(alpha)
            p.setBrush(col)
            x0, x1 = self._x(-tol, track), self._x(tol, track)
            p.drawRect(QRectF(x0, track.top(), x1 - x0, track.height()))
        p.setPen(QPen(QColor(theme.MUTED), 1))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        for c in (-50, -30, -15, 0, 15, 30, 50):
            x = self._x(c, track)
            p.drawLine(int(x), int(track.bottom() + 2), int(x), int(track.bottom() + 6))
            p.drawText(QRectF(x - 20, track.bottom() + 6, 40, 14), Qt.AlignCenter, f"{c:+d}" if c else "0")
        if self.value is not None and not np.isnan(self.value):
            x = self._x(self.value, track)
            color = QColor(theme.ACCENT if self.active else theme.MUTED)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x - 4, track.top() - 10, 8, track.height() + 20), 3, 3)
        p.end()


class StatusChip(QLabel):
    """Rounded label showing the tuning status in words on a reserved status colour."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(40)
        self.set_status("silent")

    def set_status(self, status: str) -> None:
        text, color = theme.STATUS.get(status, theme.STATUS["silent"])
        self.setText(text)
        self.setStyleSheet(f"background: {color}; color: #10120f; border-radius: 20px;"
                           "font-size: 18px; font-weight: bold; letter-spacing: 2px;"
                           "padding: 4px 18px;")


class ReadoutPanel(QFrame):
    """Right-hand panel: big swara name, signed cents, meter and status chip."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.setObjectName("panel")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)
        caption = QLabel("NOW SINGING")
        caption.setObjectName("caption")
        self.swara = QLabel("–")
        self.swara.setStyleSheet("font-size: 112px; font-weight: bold;")
        self.cents = QLabel(" ")
        self.cents.setStyleSheet(f"font-size: 34px; font-weight: bold; color: {theme.ACCENT};")
        self.meter = CentsMeter(cfg)
        self.chip = StatusChip()
        self.hint = QLabel("Press Start and sing.")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        for w in (caption, self.swara, self.cents, self.meter, self.chip, self.hint):
            layout.addWidget(w)
        layout.addStretch(1)

    def show_result(self, r: FrameResult | None) -> None:
        """Update every element from the latest frame."""
        if r is None or r.status == "silent" or not r.swara:
            self.swara.setText("–")
            self.swara.setStyleSheet(f"font-size: 112px; font-weight: bold; color: {theme.MUTED};")
            self.cents.setText(" ")
            self.meter.set_value(None, False)
            self.chip.set_status("silent")
            self.hint.setText("Listening… sing a note." if r is not None else "Press Start and sing.")
            return
        judged = r.state == "stable"
        color = theme.TEXT if judged else theme.MUTED
        self.swara.setText(r.swara)
        self.swara.setStyleSheet(f"font-size: 112px; font-weight: bold; color: {color};")
        self.cents.setText(f"{round(r.cents_error):+d} cents")
        self.cents.setStyleSheet(f"font-size: 34px; font-weight: bold; "
                                 f"color: {theme.ACCENT if judged else theme.MUTED};")
        self.meter.set_value(r.cents_error, judged)
        self.chip.set_status(r.status)
        if r.state == "stable":
            if r.held_ms >= self.cfg.stable_min_ms:
                self.hint.setText(f"Held {r.held_ms / 1000:.1f} s · counted in your score")
            else:
                self.hint.setText("Hold the note to have it judged…")
        elif r.state == "glide":
            self.hint.setText("Meend / glide: not judged.")
        else:
            self.hint.setText("Ornament (gamaka): not judged.")
