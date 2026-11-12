"""Main window: controls, pitch trace and readout. Draws only; all analysis is in the worker.

Audio concepts surfaced to the user: choosing Sa (reference pitch), tuning mode, pitch
method, and latency (one hop of buffering plus processing time per frame).
"""

from __future__ import annotations

import math
import queue
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from .. import audio_io
from ..config import Config
from ..drone import TanpuraDrone
from ..pitch import METHOD_LABELS as ESTIMATOR_LABELS
from ..swara import note_to_hz
from . import theme
from .sources import WavFileSource
from .summary_dialog import RESULTS_DIR, SummaryDialog
from .widgets import PitchTraceWidget, ReadoutPanel
from .worker import EndOfStream, FrameResult, ProcessingWorker, SaEstimate

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
POLL_MS = 33  # UI refresh: about 30 frames per second
FIND_SA_SECONDS = 3.0

HEADPHONE_WARNING = (
    "Please use headphones while the drone is on.\n\n"
    "If the drone plays through the laptop speakers, the microphone hears it. Its "
    "steady Sa and Pa can make the pitch tracker follow the drone instead of your "
    "voice (in our tests a held Pa was read as low Sa)."
)


def nearest_note(hz: float, a4_hz: float) -> tuple[str, int]:
    """Closest equal-tempered note name and octave for a frequency."""
    midi = int(round(69 + 12 * math.log2(hz / a4_hz)))
    return NOTE_NAMES[midi % 12], midi // 12 - 1


class MainWindow(QMainWindow):
    """The Swara-Aware live window."""

    def __init__(self, cfg: Config, wav_path: str | None = None, loop: bool = False,
                 sa_note: str = "D3", method: str = "yin") -> None:
        super().__init__()
        self.cfg = cfg
        self.wav_path = wav_path
        self.loop = loop
        self.sa_hz = note_to_hz(sa_note, cfg.a4_hz)
        self.mode = cfg.tuning_mode
        self.method = method
        self.source = None
        self.worker: ProcessingWorker | None = None
        self.last_session = None
        self.latest: FrameResult | None = None
        self.proc_ms_avg = 0.0
        self.headphone_warned = False
        self.recording = False
        self.find_sa_remaining = 0.0
        self.drone = TanpuraDrone.from_config(cfg, self.sa_hz)

        self.setWindowTitle("Swara-Aware")
        self.resize(1280, 760)
        self.setStyleSheet(theme.STYLE_SHEET)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_top_bar())
        centre = QHBoxLayout()
        self.trace = PitchTraceWidget(cfg)
        self.trace.set_ladder(self.mode)
        self.readout = ReadoutPanel(cfg)
        centre.addWidget(self.trace, 3)
        centre.addWidget(self.readout, 1)
        layout.addLayout(centre, 1)
        layout.addWidget(self._build_bottom_bar())
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        # PreciseTimer: Windows' default coarse timer rounds 33 ms up to ~47 ms (21 fps).
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._poll)
        self.timer.start(POLL_MS)
        self.find_sa_timer = QTimer(self)
        self.find_sa_timer.timeout.connect(self._find_sa_tick)
        self._update_latency()
        if not self.wav_path and not self.devices:
            QTimer.singleShot(0, lambda: self._error(
                "No microphone found",
                "No audio input device was found.\n\nConnect or enable a microphone and "
                "restart, or click “Open audio file…” to analyse a recording."))

    # ---------- building the bars ----------

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("bar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        row.addWidget(QLabel("Sa"))
        self.note_box = QComboBox()
        self.note_box.addItems(NOTE_NAMES)
        self.octave_box = QComboBox()
        self.octave_box.addItems([str(o) for o in range(1, 6)])
        self.hz_box = QDoubleSpinBox()
        self.hz_box.setRange(50.0, 800.0)
        self.hz_box.setDecimals(2)
        self.hz_box.setSuffix(" Hz")
        self.hz_box.setKeyboardTracking(False)  # apply on Enter / focus-out, not each key
        self._show_sa_in_controls()
        self.note_box.currentIndexChanged.connect(self._on_note_changed)
        self.octave_box.currentIndexChanged.connect(self._on_note_changed)
        self.hz_box.valueChanged.connect(self.set_sa)
        for w in (self.note_box, self.octave_box, self.hz_box):
            row.addWidget(w)

        self.find_sa_button = QPushButton("Find my Sa")
        self.find_sa_button.setToolTip(f"Hum a comfortable steady note for {FIND_SA_SECONDS:.0f} s")
        self.find_sa_button.clicked.connect(self.find_my_sa)
        row.addWidget(self.find_sa_button)
        row.addSpacing(16)

        row.addWidget(QLabel("Tuning"))
        self.mode_box = QComboBox()
        self.mode_box.addItems(["equal", "just"])
        self.mode_box.setCurrentText(self.mode)
        self.mode_box.currentTextChanged.connect(self.set_mode)
        row.addWidget(self.mode_box)

        row.addWidget(QLabel("Method"))
        self.method_box = QComboBox()
        for name, label in ESTIMATOR_LABELS.items():
            self.method_box.addItem(label, name)
        self.method_box.setCurrentIndex(list(ESTIMATOR_LABELS).index(self.method))
        self.method_box.currentIndexChanged.connect(
            lambda _: self.set_method(self.method_box.currentData()))
        row.addWidget(self.method_box)

        row.addWidget(QLabel("Input"))
        self.device_box = QComboBox()
        self.device_box.setMinimumWidth(220)
        try:
            self.devices = audio_io.list_input_devices()
        except Exception:  # PortAudio missing or broken: treat as no device
            self.devices = []
        self._fill_input_box()
        self.device_box.currentIndexChanged.connect(self._on_device_changed)
        row.addWidget(self.device_box, 1)

        open_button = QPushButton("Open audio file…")
        open_button.setToolTip("Analyse a recording (WAV, FLAC, OGG…) instead of the microphone")
        open_button.clicked.connect(self.open_audio_file)
        row.addWidget(open_button)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(self.loop)
        self.loop_check.setToolTip("Repeat the audio file")
        self.play_check = QCheckBox("Play aloud")
        self.play_check.setChecked(True)
        self.play_check.setToolTip("Hear the file while it is analysed")
        row.addWidget(self.loop_check)
        row.addWidget(self.play_check)
        return bar

    def _fill_input_box(self) -> None:
        """List the loaded audio file (if any) and the microphones; select the file."""
        self.device_box.blockSignals(True)
        self.device_box.clear()
        if self.wav_path:
            self.device_box.addItem(f"File: {Path(self.wav_path).name}", ("file", self.wav_path))
        if self.devices:
            self.device_box.addItem("Default microphone", ("mic", None))
            for d in self.devices:
                self.device_box.addItem(d["name"], ("mic", d["index"]))
        if self.device_box.count() == 0:
            self.device_box.addItem("No input: open an audio file", ("none", None))
        self.device_box.setCurrentIndex(0)
        self.device_box.blockSignals(False)

    def open_audio_file(self) -> None:
        """Choose a recording and start analysing it straight away."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio file", str(Path(self.wav_path).parent if self.wav_path
                                         else Path.cwd()),
            "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.aif);;All files (*)")
        if not path:
            return
        self.stop_listening()
        self.wav_path = path
        self._fill_input_box()
        self.start_button.setEnabled(True)
        self.start_listening()

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("bar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        self.drone_check = QCheckBox("Drone")
        self.drone_check.toggled.connect(self.toggle_drone)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(self.cfg.drone_volume * 100))
        self.volume.setFixedWidth(140)
        self.volume.valueChanged.connect(lambda v: self.drone.set_volume(v / 100))
        row.addWidget(self.drone_check)
        row.addWidget(self.volume)
        row.addSpacing(16)

        self.start_button = QPushButton("Start listening")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_listening)
        self.start_button.setEnabled(bool(self.wav_path or self.devices))  # Open file enables it
        self.record_button = QPushButton("● Record")
        self.record_button.setCheckable(True)
        self.record_button.setEnabled(False)
        self.record_button.toggled.connect(self.toggle_recording)
        row.addWidget(self.start_button)
        row.addWidget(self.record_button)
        row.addStretch(1)

        self.latency_label = QLabel()
        self.latency_label.setObjectName("muted")
        self.latency_label.setToolTip(
            "Update interval (one hop) plus processing time per frame. The pitch you see "
            "also lags by about half a frame (~23 ms, the analysis window is centred in "
            "the past), the 5-frame median filter (~23 ms) and the sound card buffer.")
        row.addWidget(self.latency_label)
        row.addSpacing(16)
        end = QPushButton("End session")
        end.clicked.connect(self.end_session)
        row.addWidget(end)
        return bar

    # ---------- Sa, mode, method, device ----------

    def _show_sa_in_controls(self) -> None:
        """Reflect self.sa_hz in the note/octave/Hz controls without triggering them."""
        name, octave = nearest_note(self.sa_hz, self.cfg.a4_hz)
        for w in (self.note_box, self.octave_box, self.hz_box):
            w.blockSignals(True)
        self.note_box.setCurrentText(name)
        if 1 <= octave <= 5:
            self.octave_box.setCurrentText(str(octave))
        self.hz_box.setValue(self.sa_hz)
        for w in (self.note_box, self.octave_box, self.hz_box):
            w.blockSignals(False)

    def _on_note_changed(self) -> None:
        note = f"{self.note_box.currentText()}{self.octave_box.currentText()}"
        self.set_sa(note_to_hz(note, self.cfg.a4_hz))

    def set_sa(self, hz: float) -> None:
        """Change Sa everywhere: worker (new session), drone, ladder, controls."""
        if abs(hz - self.sa_hz) < 1e-6:
            return
        self.sa_hz = hz
        if self.worker is not None:
            self.worker.set_sa(hz)
        self.drone.set_sa(hz)
        self.trace.clear_trace()
        self._show_sa_in_controls()
        name, octave = nearest_note(hz, self.cfg.a4_hz)
        self.statusBar().showMessage(f"Sa = {hz:.2f} Hz (nearest note {name}{octave}). "
                                     "New session started.", 5000)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if self.worker is not None:
            self.worker.set_mode(mode)
        self.trace.set_ladder(mode)

    def set_method(self, name: str) -> None:
        self.method = name
        if self.worker is not None:
            self.worker.set_estimator(name)
        self._update_latency()

    def _on_device_changed(self) -> None:
        if self.worker is not None:  # restart on the newly chosen device
            self.stop_listening()
            self.start_listening()

    # ---------- listening ----------

    def toggle_listening(self) -> None:
        if self.worker is None:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self) -> None:
        """Open the microphone (or file) and start the worker thread."""
        if self.worker is not None:
            return
        kind, value = self.device_box.currentData() or ("none", None)
        if kind == "none":
            self._error("No input", "No microphone was found. Use “Open audio file…” to "
                        "analyse a recording.")
            return
        try:
            if kind == "file":
                # Resampled to the app's rate; played aloud if wanted (the mic is unused).
                self.source = WavFileSource(value, self.cfg.hop_size, realtime=True,
                                            loop=self.loop_check.isChecked(),
                                            target_rate=self.cfg.sample_rate,
                                            play=self.play_check.isChecked())
            else:
                self.source = audio_io.LiveInputStream(self.cfg.sample_rate, self.cfg.hop_size,
                                                       device=value)
            self.source.start()
        except Exception as exc:  # unreadable file, no device, permission denied, busy...
            self.source = None
            if kind == "file":
                self._error("Cannot open the audio file",
                            f"{exc}\n\nTry a WAV file (in Audacity: File > Export > WAV).")
            else:
                self._error("Cannot open the microphone",
                            f"{exc}\n\nCheck that a microphone is connected and that this app "
                            "is allowed to use it (Windows: Settings > Privacy > Microphone; "
                            "macOS: System Settings > Privacy & Security > Microphone).")
            return
        self.worker = ProcessingWorker(self.source, self.cfg, self.sa_hz, self.method, self.mode)
        self.worker.start()
        self.trace.clear_trace()
        self.start_button.setText("Stop listening")
        self.record_button.setEnabled(True)
        what = f"file {Path(value).name}" if kind == "file" else "microphone"
        self.statusBar().showMessage(f"Listening to {what}…", 4000)

    def stop_listening(self) -> None:
        """Stop the worker and the input; keep the session for the summary."""
        if self.worker is None:
            return
        if self.recording:
            self.record_button.setChecked(False)  # saves the recording
        self.find_sa_timer.stop()
        self.find_sa_button.setEnabled(True)
        self.find_sa_button.setText("Find my Sa")
        self.worker.stop()
        self.worker.join(timeout=2.0)
        self.last_session = self.worker.snapshot_session()
        self.worker = None
        if self.source is not None:
            self.source.stop()
            self.source = None
        self.start_button.setText("Start listening")
        self.record_button.setEnabled(False)
        self.readout.show_result(None)

    # ---------- find my Sa ----------

    def find_my_sa(self) -> None:
        if self.worker is None:
            self.start_listening()
            if self.worker is None:
                return
        self.worker.find_sa(FIND_SA_SECONDS)
        self.find_sa_remaining = FIND_SA_SECONDS
        self.find_sa_button.setEnabled(False)
        self._find_sa_tick()
        self.find_sa_timer.start(250)

    def _find_sa_tick(self) -> None:
        self.find_sa_button.setText(f"Hum Sa… {max(self.find_sa_remaining, 0):.0f}")
        self.find_sa_remaining -= 0.25

    def _on_sa_estimate(self, est: SaEstimate) -> None:
        self.find_sa_timer.stop()
        self.find_sa_button.setEnabled(True)
        self.find_sa_button.setText("Find my Sa")
        if est.sa_hz is None:
            QMessageBox.information(self, "Find my Sa",
                                    "Could not find a steady note. Hum one comfortable note, "
                                    "a little louder and steadier, for 3 seconds.")
            return
        self.set_sa(est.sa_hz)
        name, octave = nearest_note(est.sa_hz, self.cfg.a4_hz)
        self.statusBar().showMessage(f"Found your Sa: {est.sa_hz:.1f} Hz "
                                     f"(close to {name}{octave}).", 8000)

    # ---------- drone ----------

    def toggle_drone(self, on: bool) -> None:
        if not on:
            self.drone.stop()
            return
        if not self.headphone_warned:
            self.headphone_warned = True
            QMessageBox.information(self, "Use headphones", HEADPHONE_WARNING)
        try:
            self.drone.start()
        except Exception as exc:
            self.drone_check.blockSignals(True)
            self.drone_check.setChecked(False)
            self.drone_check.blockSignals(False)
            self._error("Cannot play the drone", f"No usable audio output device.\n\n{exc}")

    # ---------- recording and summary ----------

    def toggle_recording(self, on: bool) -> None:
        if self.worker is None:
            return
        if on:
            self.worker.start_recording()
            self.recording = True
            self.record_button.setText("■ Stop recording")
            return
        self.recording = False
        self.record_button.setText("● Record")
        audio = self.worker.stop_recording()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = RESULTS_DIR / "sessions"
        audio_io.save_wav(folder / f"session_{stamp}.wav", audio, self.cfg.sample_rate)
        self.worker.snapshot_session().export_csv(folder / f"session_{stamp}.csv")
        self.statusBar().showMessage(f"Saved {folder / f'session_{stamp}'}.wav and .csv", 8000)

    def end_session(self) -> None:
        session = self.worker.snapshot_session() if self.worker else self.last_session
        if session is None or not session.frames:
            QMessageBox.information(self, "Session summary", "Nothing recorded yet. "
                                    "Press Start listening and sing first.")
            return
        SummaryDialog(session, self).exec()
        if self.worker is not None:
            self.worker.new_session()
        self.last_session = None

    # ---------- the UI timer: drain results and draw ----------

    def _poll(self) -> None:
        if self.worker is None:
            return
        got_frame = False
        while True:
            try:
                item = self.worker.results.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, FrameResult):
                self.trace.push(item)
                self.latest = item
                self.proc_ms_avg += 0.05 * (item.proc_ms - self.proc_ms_avg)
                got_frame = True
            elif isinstance(item, SaEstimate):
                self._on_sa_estimate(item)
            elif isinstance(item, EndOfStream):
                self.stop_listening()
                self.statusBar().showMessage("End of file. Press Start listening to replay.", 0)
                return
        if got_frame:
            self.trace.redraw(self.latest)
            self.readout.show_result(self.latest)
            self._update_latency()

    def _update_latency(self) -> None:
        hop_ms = self.cfg.hop_duration_ms
        self.latency_label.setText(f"Latency ≈ {hop_ms:.1f} ms hop + {self.proc_ms_avg:.2f} ms "
                                   f"processing · {ESTIMATOR_LABELS[self.method]}")

    def _error(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    # ---------- clean shutdown ----------

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 (Qt API)
        """Stop timers, threads and audio streams before the window closes."""
        self.timer.stop()
        self.find_sa_timer.stop()
        self.stop_listening()
        self.drone.stop()
        event.accept()
