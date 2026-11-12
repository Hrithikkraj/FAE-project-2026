"""Audio sources that behave like the microphone stream, for demos and tests.

Audio concepts: block-based streaming. The live microphone (audio_io.LiveInputStream)
delivers hop-sized blocks into a queue; these sources do the same from a WAV file or a
numpy array, optionally paced in real time, so the rest of the pipeline cannot tell
the difference.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..audio_io import load_wav


class ArraySource:
    """Feeds a signal as hop-sized blocks into a queue, like LiveInputStream.

    Args:
        signal: Mono float signal.
        sample_rate: Its sampling rate.
        hop_size: Block length in samples.
        realtime: Pace blocks at the real playback rate (True for the app); False
            pushes them as fast as possible (tests).
        loop: Start again from the beginning when the end is reached.
    """

    def __init__(self, signal: np.ndarray, sample_rate: int, hop_size: int,
                 realtime: bool = True, loop: bool = False) -> None:
        self.signal = np.asarray(signal, dtype=float)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.realtime = realtime
        self.loop = loop
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self.finished = threading.Event()  # set when the signal has been fully sent
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        """Producer thread: cut the signal into blocks and queue them."""
        n_blocks = int(np.ceil(len(self.signal) / self.hop_size))
        start = time.perf_counter()
        sent = 0
        while not self._stop.is_set():
            i = sent % n_blocks
            if i == 0 and sent > 0 and not self.loop:
                break
            block = self.signal[i * self.hop_size:(i + 1) * self.hop_size]
            if len(block) < self.hop_size:
                block = np.pad(block, (0, self.hop_size - len(block)))
            if self.realtime:  # wait until this block would have been recorded
                due = start + sent * self.hop_size / self.sample_rate
                delay = due - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
            try:
                self.blocks.put(block, timeout=0.5)
            except queue.Full:
                continue  # consumer is gone or stuck; check the stop flag again
            sent += 1
        self.finished.set()

    def start(self) -> None:
        """Begin sending blocks (in a background thread)."""
        if self._thread is None:
            self._stop.clear()
            self.finished.clear()
            self._thread = threading.Thread(target=self._run, name="ArraySource", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop sending blocks."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Next block, or None if none arrived within ``timeout`` seconds."""
        try:
            return self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def active(self) -> bool:
        """True while blocks are still being sent."""
        return self._thread is not None and not self.finished.is_set()


class WavFileSource(ArraySource):
    """An ArraySource that reads its signal from an audio file (mono-mixed).

    Args:
        path: WAV (or FLAC/OGG, anything soundfile can read).
        hop_size: Block length in samples.
        realtime: Pace blocks at playback speed.
        loop: Repeat the file.
        target_rate: Resample the file to this rate (Hz), so it matches the app's
            analysis settings; None keeps the file's own rate.
        play: Also play the file through the speakers while it is analysed. Safe here
            because the microphone is not being used in file mode.
    """

    def __init__(self, path: str | Path, hop_size: int, realtime: bool = True,
                 loop: bool = False, target_rate: int | None = None,
                 play: bool = False) -> None:
        signal, sample_rate = load_wav(path)
        if target_rate is not None and target_rate != sample_rate:
            signal = resample(signal, sample_rate, target_rate)
            sample_rate = target_rate
        super().__init__(signal, sample_rate, hop_size, realtime, loop)
        self.path = Path(path)
        self.play = play

    def start(self) -> None:
        """Start sending blocks, and start audible playback if requested."""
        if self.play and self._thread is None:
            try:
                import sounddevice as sd
                sd.play(self.signal, self.sample_rate, loop=self.loop)
            except Exception:  # no output device: analyse silently
                self.play = False
        super().start()

    def stop(self) -> None:
        """Stop sending blocks and stop playback."""
        if self.play:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        super().stop()


def resample(signal: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Change the sample rate with a polyphase filter (anti-aliasing included)."""
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(rate_in, rate_out)
    return resample_poly(signal, rate_out // g, rate_in // g)
