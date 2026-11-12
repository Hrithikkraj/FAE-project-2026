"""Tanpura-style drone by additive synthesis.

Audio concepts: additive synthesis (a tone built as a sum of harmonics), amplitude
envelopes (pluck attack and exponential decay), slowly varying harmonic amplitudes
(a simple imitation of the tanpura's jawari "buzz"), phase accumulators (continuous
phase across audio blocks, so changing pitch does not click), and block-based real-time
output through a sound-card callback.

The four strings are plucked in a repeating cycle: Pa (or Ma / Ni) in the lower octave,
Sa, Sa, then low Sa (one octave below), at ``tempo_bpm`` plucks per minute.

IMPORTANT, use headphones: if the drone plays through the laptop speakers, the
microphone hears it. Its Sa and Pa are strong, steady pitches, so the pitch tracker
may lock onto the drone instead of the voice (or report a mixture). Use headphones
for live practice. Evaluation scripts always run with the drone OFF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config

# First string, as a ratio of madhya Sa (all in the mandra, i.e. lower, octave).
FIRST_STRING_RATIO = {"Pa": 3 / 4, "Ma": 2 / 3, "Ni": 15 / 16}
ATTACK_S = 0.005    # pluck attack time
DAMP_S = 0.03       # fade just before a string is plucked again (avoids a click)
MAX_HARMONIC_HZ = 8000.0  # skip harmonics above this (inaudible in a drone, saves CPU)
JAWARI_SWEEP_S = 1.2      # time for the bright "buzz" peak to sweep down the harmonics
JAWARI_WIDTH = 2.5        # width of that peak, in harmonic numbers
JAWARI_BOOST = 2.0        # how much the peak lifts the harmonics it passes over
PEAK_TARGET = 0.7         # approximate output peak at volume 1.0


class TanpuraDrone:
    """Stateful drone generator: call :meth:`render` repeatedly for consecutive blocks."""

    def __init__(
        self,
        sa_hz: float,
        sample_rate: int = Config.sample_rate,
        tempo_bpm: float = Config.drone_tempo_bpm,
        volume: float = Config.drone_volume,
        decay_s: float = Config.drone_decay_s,
        n_harmonics: int = Config.drone_harmonics,
        first_string: str = Config.drone_first_string,
        device: int | None = None,
    ) -> None:
        """Create a drone tuned to ``sa_hz``. Nothing plays until :meth:`start`."""
        self.sa_hz = sa_hz
        self.sample_rate = sample_rate
        self.tempo_bpm = tempo_bpm
        self.volume = float(np.clip(volume, 0.0, 1.0))
        self.decay_s = decay_s
        self.n_harmonics = n_harmonics
        self.first_string = first_string
        self.device = device
        self._stream = None
        self.reset()

    @classmethod
    def from_config(cls, cfg: Config, sa_hz: float) -> "TanpuraDrone":
        """Build a drone from the Config defaults."""
        return cls(sa_hz, cfg.sample_rate, cfg.drone_tempo_bpm, cfg.drone_volume,
                   cfg.drone_decay_s, cfg.drone_harmonics, cfg.drone_first_string)

    def reset(self) -> None:
        """Restart the pluck cycle from the first string."""
        self._sample = 0  # samples rendered so far (global clock)
        self._phases = np.zeros((4, self.n_harmonics))  # one phase per string, harmonic
        self._current_volume = self.volume
        k = np.arange(1, self.n_harmonics + 1)
        # Fixed per-harmonic shimmer rates and offsets (deterministic, so tests repeat).
        self._swell_rate = 0.6 + 0.12 * k          # Hz: higher harmonics shimmer faster
        self._swell_phase = 1.7 * k
        self._base_amp = 1.0 / k**0.7               # rich spectrum, slow roll-off
        # Harmonic phases are unrelated, so the sum behaves like noise: its peak is a few
        # times its RMS, not the sum of all amplitudes. Scale by RMS (sqrt of summed
        # powers), allowing for the jawari boost and two strings ringing together.
        rms_one_string = np.sqrt(0.5 * np.sum((self._base_amp * (1 + JAWARI_BOOST)) ** 2))
        self._norm = PEAK_TARGET / (1.5 * np.sqrt(2.0) * rms_one_string)

    def string_ratios(self) -> list[float]:
        """Frequency ratios (to madhya Sa) of the four strings in pluck order."""
        return [FIRST_STRING_RATIO[self.first_string], 1.0, 1.0, 0.5]

    def set_volume(self, volume: float) -> None:
        """Change the volume (0..1); it ramps over the next block, so no click."""
        self.volume = float(np.clip(volume, 0.0, 1.0))

    def set_sa(self, sa_hz: float) -> None:
        """Retune to a new Sa; phases are continuous, so the change does not click."""
        self.sa_hz = sa_hz

    def render(self, n: int) -> np.ndarray:
        """Synthesise the next ``n`` samples of the drone (float, mono)."""
        sr = self.sample_rate
        interval = max(1, int(round(60.0 / self.tempo_bpm * sr)))  # samples between plucks
        cycle = 4 * interval
        clock = self._sample + np.arange(n)
        out = np.zeros(n)
        k = np.arange(1, self.n_harmonics + 1)
        steps = np.arange(1, n + 1)

        for s, ratio in enumerate(self.string_ratios()):
            f = self.sa_hz * ratio
            use = k * f < min(sr / 2, MAX_HARMONIC_HZ)

            # Envelope: time since this string's latest pluck (string s plucks at
            # s*interval, then every cycle). Attack, exponential decay, and a short
            # damping fade just before the next pluck of the same string.
            since = (clock - s * interval) % cycle
            t = since / sr
            env = (1.0 - np.exp(-t / ATTACK_S)) * np.exp(-t / self.decay_s)
            env *= np.clip((cycle - since) / (DAMP_S * sr), 0.0, 1.0)
            env *= clock >= s * interval  # not plucked yet at the very start

            # Phase accumulators: phase grows by 2*pi*k*f/sr per sample.
            dphi = 2.0 * np.pi * k[use] * f / sr
            phase = self._phases[s, use][:, None] + dphi[:, None] * steps[None, :]
            self._phases[s, use] = phase[:, -1] % (2.0 * np.pi)

            # Jawari imitation: a bright spectral peak starts at the top harmonics after
            # each pluck and sweeps down through them, lifting each harmonic as it
            # passes; a slow shimmer on every harmonic keeps the sound alive.
            kk = k[use][:, None]
            centre = 2.0 + (kk.max() - 2.0) * np.exp(-t / JAWARI_SWEEP_S)[None, :]
            jawari = 1.0 + JAWARI_BOOST * np.exp(-((kk - centre) / JAWARI_WIDTH) ** 2)
            shimmer = 0.85 + 0.15 * np.sin(2.0 * np.pi * self._swell_rate[use, None]
                                           * t[None, :] + self._swell_phase[use, None])
            amps = self._base_amp[use, None] * jawari * shimmer
            out += env * np.sum(amps * np.sin(phase), axis=0)

        self._sample += n
        # Volume ramps linearly from the previous block's value: no zipper noise.
        gain = np.linspace(self._current_volume, self.volume, n)
        self._current_volume = self.volume
        return out * self._norm * gain

    def render_seconds(self, seconds: float) -> np.ndarray:
        """Render ``seconds`` of drone (continues from the current state)."""
        return self.render(int(round(seconds * self.sample_rate)))

    def save_wav(self, path: str | Path, seconds: float) -> None:
        """Render ``seconds`` of drone from the start of the cycle and save as WAV."""
        from .audio_io import save_wav

        self.reset()
        save_wav(path, self.render_seconds(seconds), self.sample_rate)

    # ---------- real-time output ----------

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        """Sound-card callback: fill the output block with the next drone samples."""
        outdata[:, 0] = self.render(frames)

    def start(self, blocksize: int = 1024) -> None:
        """Start playing through the default (or chosen) output device.

        Use headphones: the microphone must not hear the drone (see module docstring).
        """
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1,
                                       dtype="float32", blocksize=blocksize,
                                       device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        """Stop playback."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def playing(self) -> bool:
        """True while the output stream is running."""
        return self._stream is not None and self._stream.active

    def __enter__(self) -> "TanpuraDrone":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
