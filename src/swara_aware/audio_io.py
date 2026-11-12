"""Microphone input and WAV file input/output.

Audio concepts: sampling (the sound card turns air pressure into numbers at
``sample_rate`` samples per second), block-based streaming (audio arrives in small
blocks via a callback), and buffering between the audio thread and the analysis code.

Uses sounddevice (PortAudio) for the microphone and soundfile for WAV files.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf


def list_input_devices() -> list[dict[str, Any]]:
    """Return the audio devices that can record (at least one input channel).

    Each entry has ``index``, ``name``, ``channels`` and ``default_samplerate``.
    """
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(
                {
                    "index": index,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
    return devices


def record(seconds: float, sample_rate: int, device: int | None = None) -> np.ndarray:
    """Record ``seconds`` of mono audio from the microphone (blocking).

    Args:
        seconds: Recording length.
        sample_rate: Samples per second.
        device: Input device index (see :func:`list_input_devices`); None = default.

    Returns:
        1-D float array in [-1, 1].
    """
    n_samples = int(round(seconds * sample_rate))
    data = sd.rec(
        n_samples, samplerate=sample_rate, channels=1, dtype="float32", device=device
    )
    sd.wait()  # block until the recording is finished
    return data[:, 0].astype(float)


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average the channels of a (n_samples, n_channels) array into one channel."""
    x = np.asarray(x, dtype=float)
    return x if x.ndim == 1 else x.mean(axis=1)


def save_wav(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    """Save a float signal as a 16-bit PCM WAV file (values are clipped to [-1, 1])."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sample_rate, subtype="PCM_16")


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float signal.

    Returns:
        ``(x, sample_rate)``. No resampling is done; check the rate against your Config.
    """
    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    return to_mono(data), int(sample_rate)


class LiveInputStream:
    """Non-blocking microphone stream that delivers hop-sized blocks into a queue.

    PortAudio calls ``_callback`` on its own audio thread each time ``hop_size`` new
    samples are ready. The callback only copies the block into a thread-safe
    ``queue.Queue``; all analysis happens on the reader's side. If the reader falls
    behind and the queue is full, the oldest block is dropped so latency stays low.

    Example::

        with LiveInputStream(44100, 512) as stream:
            block = stream.read(timeout=1.0)   # 512 samples, or None
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        device: int | None = None,
        max_blocks: int = 64,
    ) -> None:
        """Set up the stream (it does not start recording until :meth:`start`)."""
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.device = device
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_blocks)
        self.dropped_blocks = 0
        self.status_flags = 0  # count of over/underflow reports from PortAudio
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Audio-thread callback: copy one block into the queue (keep it short)."""
        if status:
            self.status_flags += 1
        block = indata[:, 0].astype(float)  # copy: indata is reused by PortAudio
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            try:
                self.blocks.get_nowait()  # drop the oldest block
            except queue.Empty:
                pass
            self.dropped_blocks += 1
            self.blocks.put_nowait(block)

    def start(self) -> None:
        """Open the microphone and begin delivering blocks."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.hop_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Return the next block, waiting up to ``timeout`` seconds; None if none came."""
        try:
            return self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self) -> list[np.ndarray]:
        """Return every block waiting in the queue right now (may be empty)."""
        out = []
        while True:
            try:
                out.append(self.blocks.get_nowait())
            except queue.Empty:
                return out

    @property
    def active(self) -> bool:
        """True while the stream is running."""
        return self._stream is not None and self._stream.active

    def __enter__(self) -> "LiveInputStream":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
