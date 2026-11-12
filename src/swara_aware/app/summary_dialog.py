"""End-of-session summary dialog: total in-tune %, per-swara chart, save buttons.

Audio concept shown: time-weighted intonation accuracy over held notes only.
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout)

from ..session import Session
from . import theme

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


class SummaryDialog(QDialog):
    """Shows Session.summary() and lets the user save the CSV and the plots."""

    def __init__(self, session: Session, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Session summary")
        self.resize(720, 520)
        s = session.summary()
        layout = QVBoxLayout(self)

        if s["in_tune_pct"] is None:
            headline = "No held notes yet: hold a note for at least " \
                       f"{session.cfg.stable_min_ms / 1000:.2f} s to have it judged."
        else:
            headline = (f"In tune {s['in_tune_pct']:.0f}% of {s['judged_seconds']:.1f} s held "
                        f"(±{session.cfg.in_tune_cents:.0f} cents) · "
                        f"{s['held_notes']} held notes · average |error| "
                        f"{s['mean_abs_cents']:.1f} cents")
        title = QLabel(headline)
        title.setWordWrap(True)
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {theme.TEXT};")
        layout.addWidget(title)
        secs = s["seconds_by_state"]
        detail = QLabel(" · ".join(f"{k} {v:.1f} s" for k, v in secs.items())
                        + "   (glides and ornaments are never counted as errors)")
        detail.setObjectName("muted")
        layout.addWidget(detail)

        layout.addWidget(self._chart(s["per_swara"]), 1)

        buttons = QHBoxLayout()
        save_csv = QPushButton("Save CSV…")
        save_plots = QPushButton("Save plots…")
        close = QPushButton("Close")
        close.setObjectName("primary")
        save_csv.clicked.connect(self._save_csv)
        save_plots.clicked.connect(self._save_plots)
        close.clicked.connect(self.accept)
        buttons.addWidget(save_csv)
        buttons.addWidget(save_plots)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _chart(self, per_swara: dict) -> pg.PlotWidget:
        """Bar chart of % of held time in tune per swara, labelled with mean error."""
        plot = pg.PlotWidget(background=theme.BG)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setTitle("% of held time in tune, per swara", color=theme.MUTED)
        labels = list(per_swara)
        values = [per_swara[k]["in_tune_pct"] for k in labels]
        if labels:
            plot.addItem(pg.BarGraphItem(x=list(range(len(labels))), height=values, width=0.6,
                                         brush=theme.ACCENT, pen=pg.mkPen(None)))
            for i, k in enumerate(labels):
                text = pg.TextItem(f"{values[i]:.0f}%\n{round(per_swara[k]['mean_cents']):+d}¢ avg",
                                   color=theme.TEXT, anchor=(0.5, 1.0))
                text.setPos(i, values[i] + 3)
                plot.addItem(text)
        axis = plot.getAxis("bottom")
        axis.setTicks([[(i, k) for i, k in enumerate(labels)]])
        for name in ("bottom", "left"):
            plot.getAxis(name).setTextPen(theme.MUTED)
            plot.getAxis(name).setPen(theme.FAINT)
        plot.setYRange(0, 120, padding=0)
        plot.setXRange(-0.7, max(len(labels), 1) - 0.3, padding=0)
        return plot

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save pitch track", str(RESULTS_DIR / "session.csv"),
                                              "CSV files (*.csv)")
        if path:
            self.session.export_csv(path)
            QMessageBox.information(self, "Saved", f"Pitch track saved to\n{path}")

    def _save_plots(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder for plots", str(RESULTS_DIR))
        if folder:
            self.session.plot_trace(Path(folder) / "session_trace.png")
            self.session.plot_accuracy(Path(folder) / "session_accuracy.png")
            QMessageBox.information(self, "Saved",
                                    f"Saved session_trace.png and session_accuracy.png in\n{folder}")
