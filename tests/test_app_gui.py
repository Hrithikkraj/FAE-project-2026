"""Optional GUI smoke test (offscreen Qt): the window builds, listens to a WAV, closes cleanly.

Skipped automatically if PySide6 / pyqtgraph are not installed.
"""

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from swara_aware import audio_io, synth  # noqa: E402
from swara_aware.config import Config  # noqa: E402


def test_window_listens_to_wav_and_closes(tmp_path, monkeypatch) -> None:
    from swara_aware.app.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    cfg = Config()
    wav = tmp_path / "tone.wav"
    audio_io.save_wav(wav, synth.voice_like_tone(220.0, 2.0, cfg.sample_rate), cfg.sample_rate)

    app = QApplication.instance() or QApplication([])
    window = MainWindow(cfg, wav_path=str(wav))
    window.play_check.setChecked(False)  # never make sound during tests
    window.show()
    window.start_listening()
    pump(app, 1.5)
    assert window.latest is not None and not np.isnan(window.latest.cents)

    window.set_method("autocorr")      # live changes must not crash
    window.set_mode("just")
    window.set_sa(196.0)
    app.processEvents()
    window.close()
    assert window.worker is None and window.source is None
    assert not any(t.name == "ProcessingWorker" and t.is_alive() for t in threading.enumerate())


def pump(app, seconds: float) -> None:  # noqa: ANN001
    """Run the Qt event loop for a while."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def test_open_audio_file_button(tmp_path, monkeypatch) -> None:
    """'Open audio file…' loads a 48 kHz recording (resampled), analyses it at the right
    pitch, and lists it in the Input box next to the microphones."""
    from PySide6.QtWidgets import QFileDialog

    from swara_aware.app.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    wav = tmp_path / "sa_48k.wav"
    audio_io.save_wav(wav, synth.voice_like_tone(220.0, 2.0, 48000), 48000)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(wav), "")))

    app = QApplication.instance() or QApplication([])
    window = MainWindow(Config(), sa_note="A3")  # Sa = 220 Hz
    window.play_check.setChecked(False)
    window.show()
    window.open_audio_file()
    pump(app, 1.5)
    assert window.device_box.currentData() == ("file", str(wav))
    assert window.source.sample_rate == Config().sample_rate  # resampled to 44.1 kHz
    assert window.latest is not None and window.latest.swara == "Sa"
    assert abs(window.latest.cents_error) < 5  # resampling kept the pitch exact
    window.close()
    assert window.worker is None
