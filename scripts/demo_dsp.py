"""Stage 1 demo: look at a voice recording four ways.

Audio concepts: sampling (waveform), framing + RMS in dBFS with a silence gate, the FFT
magnitude spectrum of one frame (harmonics), and the spectrogram.

Usage (from the swara-aware folder):
    python scripts/demo_dsp.py                  # record 3 s from the default mic
    python scripts/demo_dsp.py --wav data/x.wav # analyse an existing file
    python scripts/demo_dsp.py --synth          # use a synthetic voice-like tone
    python scripts/demo_dsp.py --show           # also open the plot window

The figure is saved to results/demo_dsp.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

import numpy as np  # noqa: E402

from swara_aware import audio_io, dsp, synth  # noqa: E402
from swara_aware.config import Config  # noqa: E402


def get_signal(args: argparse.Namespace, cfg: Config) -> tuple[np.ndarray, int, str]:
    """Return (signal, sample_rate, title) from a WAV file, a synthetic tone or the mic."""
    if args.wav:
        x, sr = audio_io.load_wav(args.wav)
        return x, sr, f"File: {Path(args.wav).name}"
    if args.synth:
        sr = cfg.sample_rate
        tone = synth.voice_like_tone(220.0, 2.0, sr, amplitude=0.4)
        silence = np.zeros(int(0.5 * sr))
        x = np.concatenate([silence, tone, silence])
        x = dsp.add_noise(x, snr_db=40.0, noise=synth.pink_noise(len(x) / sr, sr))
        return x, sr, "Synthetic voice-like tone, f0 = 220 Hz"
    print(f"Recording {args.seconds:.1f} s from the microphone... sing a steady note.")
    x = audio_io.record(args.seconds, cfg.sample_rate, device=args.device)
    print("Done.")
    out_wav = ROOT / "data" / "demo_recording.wav"
    audio_io.save_wav(out_wav, x, cfg.sample_rate)
    print(f"Saved recording to {out_wav}")
    return x, cfg.sample_rate, "Microphone recording"


def plot_demo(x: np.ndarray, sr: int, cfg: Config, title: str, out_path: Path) -> None:
    """Draw waveform, RMS dBFS, one frame's spectrum and the spectrogram; save as PNG."""
    import matplotlib.pyplot as plt

    n, hop = cfg.frame_size, cfg.hop_size
    t = np.arange(len(x)) / sr

    frames = dsp.frame_signal(x, n, hop)
    levels = dsp.to_dbfs(dsp.rms(frames))
    voiced = dsp.silence_gate(frames, cfg.silence_threshold_dbfs)
    frame_times = (np.arange(len(frames)) * hop + n / 2) / sr

    loudest = int(np.argmax(levels))  # pick the loudest frame for the spectrum
    freqs, mags = dsp.magnitude_spectrum(frames[loudest], sr)
    peak_bin = int(np.argmax(mags[1:])) + 1

    times, sfreqs, S = dsp.spectrogram(x, sr, n, hop)
    S_db = dsp.amplitude_to_db(S, floor_db=-100.0)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    fig.suptitle(f"Swara-Aware DSP demo: {title}", fontsize=13)

    ax = axes[0]
    ax.plot(t, x, lw=0.6, color="tab:blue")
    ax.set(title="1. Waveform (sampled signal)", xlabel="Time (s)", ylabel="Amplitude",
           xlim=(0, t[-1] if len(t) else 1), ylim=(-1, 1))

    ax = axes[1]
    ax.plot(frame_times, levels, color="tab:orange", label="RMS level")
    ax.axhline(cfg.silence_threshold_dbfs, color="tab:red", ls="--",
               label=f"Silence threshold ({cfg.silence_threshold_dbfs:.0f} dBFS)")
    ax.fill_between(frame_times, -120, 0, where=voiced, color="tab:green", alpha=0.12,
                    label="Gate open (sound)")
    ax.axvline(frame_times[loudest], color="gray", lw=0.8, ls=":")
    ax.set(title=f"2. RMS per frame (frame {n}, hop {hop}) in dBFS",
           xlabel="Time (s)", ylabel="Level (dBFS)", ylim=(-100, 0),
           xlim=(0, t[-1] if len(t) else 1))
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    ax.plot(freqs, dsp.amplitude_to_db(mags), color="tab:purple", lw=0.8)
    ax.axvline(freqs[peak_bin], color="gray", ls=":",
               label=f"Strongest peak: {freqs[peak_bin]:.1f} Hz")
    ax.set(title=f"3. Magnitude spectrum of the loudest frame (t = {frame_times[loudest]:.2f} s, "
                 f"Hann window, bin width {sr / n:.1f} Hz)",
           xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, 3000), ylim=(-120, 0))
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    mesh = ax.pcolormesh(times, sfreqs, S_db, shading="auto", cmap="magma")
    ax.set(title="4. Spectrogram (our framing + FFT)", xlabel="Time (s)",
           ylabel="Frequency (Hz)", ylim=(0, 3000))
    fig.colorbar(mesh, ax=ax, label="dB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")


def main() -> None:
    """Parse arguments, get a signal, and make the figure."""
    parser = argparse.ArgumentParser(description="Stage 1 DSP demo for Swara-Aware.")
    parser.add_argument("--wav", help="analyse this audio file instead of recording")
    parser.add_argument("--synth", action="store_true", help="use a synthetic test tone")
    parser.add_argument("--seconds", type=float, default=3.0, help="recording length")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--show", action="store_true", help="open the plot window too")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # save to file only, no window needed
    cfg = Config()
    x, sr, title = get_signal(args, cfg)
    if sr != cfg.sample_rate:
        print(f"Note: file sample rate is {sr} Hz (Config default {cfg.sample_rate} Hz).")
    plot_demo(x, sr, cfg, title, ROOT / "results" / "demo_dsp.png")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
