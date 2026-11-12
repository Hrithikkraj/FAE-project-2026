# Swara-Aware — project brief

This file is the permanent brief for anyone (human or AI assistant) working on this
repository. Read it before changing code.

## Summary

Swara-Aware is a free, open-source, real-time pitch feedback tool for learner singers,
built as a software-only project for the Fundamentals of Audio Engineering (FAE) course
(Group 5: Swarit Kumar 2023540, Hritik Raj 2023245).

It listens to the singer through the microphone, finds the pitch, and shows which swara
(Sa Re Ga Ma Pa Dha Ni) they are singing and how many cents sharp or flat they are,
relative to the singer's **own chosen Sa**. It plays a tanpura-style drone (Sa and Pa) as
a reference, and does **not** mark deliberate glides (meend, gamaka) as mistakes.

We also compare several pitch-detection methods that we write ourselves, and measure
their accuracy, latency and robustness to noise.

## Constraints

- Software only, no extra hardware. Laptop microphone and speakers/headphones.
- Python 3.10+, cross-platform (Windows, macOS, Linux).
- Allowed libraries: numpy, scipy, sounddevice, soundfile, matplotlib, pytest,
  PySide6 and pyqtgraph (live UI, Stage 4), librosa (**only** as a reference pitch
  tracker for comparison — never inside the core algorithms).
- Core DSP and pitch algorithms are written from scratch with numpy so that every line
  can be explained in a viva. Prefer simple, readable code over clever code.
- Every module has a short docstring naming the audio-engineering concept it demonstrates
  (sampling, framing, windowing, RMS, FFT, autocorrelation, cents, SNR, ...).
- Keep it simple: no frameworks, databases, web servers or packaging beyond what is listed.

## Folder layout

```
swara-aware/
  CLAUDE.md            this brief
  README.md            setup and usage
  requirements.txt     pinned minimum versions
  LICENSE              MIT
  src/swara_aware/
    __init__.py
    config.py          Config dataclass with every tunable default
    audio_io.py        device listing, recording, WAV I/O, live input stream
    dsp.py             framing, windowing, RMS/dBFS, FFT, silence gate, spectrogram, add_noise
    synth.py           test signals: sine, voice-like tone, glide, white/pink noise
    swara.py           cents from Sa, 12 swaras x 3 octaves, equal/just tuning, note_to_hz, estimate_sa
    ornament.py        OrnamentTracker: silent / stable / glide / ornament per frame (live or batch)
    session.py         Session: judged held notes, per-swara stats, CSV, plots; analyse_track()
    drone.py           TanpuraDrone: additive-synthesis tanpura, real-time output or render to array
    pitch/
      base.py          PitchEstimator, PitchResult, estimate_track, parabolic interpolation
      fft_peak.py      FFT peak picking + Harmonic Product Spectrum variant
      autocorr.py      normalized autocorrelation via FFT (Wiener-Khinchin)
      yin.py           YIN (difference function, CMNDF, absolute threshold)
      reference.py     librosa pYIN wrapper (comparison only, lazy import)
    app/               live desktop app (Stage 4)
      sources.py       ArraySource / WavFileSource: same interface as LiveInputStream (no Qt)
      worker.py        ProcessingWorker thread: blocks -> estimator -> tracker -> FrameResult (no Qt)
      theme.py         dark theme, saffron accent, reserved status colours
      widgets.py       PitchTraceWidget, CentsMeter, StatusChip, ReadoutPanel
      summary_dialog.py SummaryDialog (per-swara chart, save CSV/plots)
      main_window.py   MainWindow (top bar, trace + readout, bottom bar)
      main.py          argparse (--wav, --loop, --sa, --method), QApplication
    eval/              Stage 5 experiments (no Qt, no drone)
      common.py        master seed 2026 + per-experiment rng_for(name), styles, CSV/markdown helpers
      accuracy.py      E1  noise.py E2  latency.py E3  voice.py E4  ornament_eval.py E5
      sensitivity.py   E6  report.py -> results/RESULTS.md (findings generated from the data)
    __main__.py        `PYTHONPATH=src python -m swara_aware`
  tests/               pytest tests (87; GUI test offscreen, skipped without PySide6)
  scripts/             demos, run_app.py, make_demo_wav.py, run_all_experiments.py
  data/                demo_phrase.wav; put real recordings in data/voice/, noise in data/noise/
  results/             experiment CSV/PNG + RESULTS.md (committed); sessions/ (git-ignored)
  docs/                algorithms.md, user_guide.md, demo_script.md, app_screenshot.png
```

Scripts add `src/` to `sys.path` themselves, and `tests/conftest.py` does the same for
pytest, so no install step is needed.

## Coding rules

- Type hints on every function signature.
- A docstring on every module and public function (what it does, parameters, returns,
  and the audio concept involved).
- Small functions that do one thing. Pure functions for DSP: arrays in, arrays out.
- No hidden global state. Parameters come from a `Config` instance passed in explicitly
  (or as explicit keyword arguments); nothing is hard-coded outside `config.py`.
- Audio signals are 1-D `float` numpy arrays in the range [-1, 1], mono.
- Every new DSP or pitch function gets a pytest test using synthetic signals from
  `synth.py`, where the right answer is known.
- Run `python -m pytest` before committing.

## Default parameters (`src/swara_aware/config.py`)

| Name                          | Default | Meaning                                              |
|-------------------------------|---------|------------------------------------------------------|
| `sample_rate`                 | 44100   | samples per second (Hz)                              |
| `frame_size`                  | 2048    | samples per analysis frame (~46 ms)                  |
| `hop_size`                    | 512     | samples between frame starts (~11.6 ms, 75% overlap) |
| `fmin`                        | 70.0    | lowest pitch searched (Hz)                           |
| `fmax`                        | 1000.0  | highest pitch searched (Hz)                          |
| `silence_threshold_dbfs`      | -50.0   | frames quieter than this RMS level are silence       |
| `in_tune_cents`               | 15.0    | within ±15 cents of the swara = in tune              |
| `close_cents`                 | 30.0    | within ±30 cents = close                             |
| `stable_min_ms`               | 250.0   | a note must hold this long to count as a held swara  |
| `glide_threshold_cents_per_s` | 200.0   | faster pitch change than this = glide (meend)        |
| `yin_threshold`               | 0.15    | YIN: first CMNDF dip below this is the period        |
| `acf_voicing_threshold`       | 0.5     | autocorr: peak height needed to call a frame voiced  |
| `acf_peak_ratio`              | 0.9     | autocorr: first peak >= 0.9 x best peak is chosen    |
| `fft_zero_pad_factor`         | 4       | FFT length = next pow2 >= frame_size x 4             |
| `fft_voicing_threshold`       | 0.8     | FFT: peak prominence (1 - median/peak) to be voiced  |
| `hps_harmonics`               | 4       | harmonics multiplied in the Harmonic Product Spectrum|

| `tuning_mode`                 | "equal" | "equal" (100-cent steps) or "just" (ratios)          |
| `a4_hz`                       | 440.0   | reference for note names when choosing Sa            |
| `min_confidence`              | 0.5     | lower pitch confidence = silent                      |
| `median_filter_frames`        | 5       | causal median on the cents track (~58 ms)            |
| `slope_window_ms`             | 120     | slope and stable-range window                        |
| `ornament_window_ms`          | 350     | oscillation window (>= one cycle at 3 Hz)            |
| `stable_max_range_cents`      | 25      | max range in slope window for "stable"               |
| `ornament_min_range_cents`    | 40      | oscillation this wide = ornament; smaller = vibrato  |
| `ornament_max_drift_cents_per_s` | 250  | max net drift of an oscillation                      |
| `ornament_turn_fraction`      | 0.3     | back-travel >= 30% of range = oscillating            |
| `drone_tempo_bpm`             | 60      | tanpura plucks per minute                            |
| `drone_volume`                | 0.5     | drone gain 0..1                                      |
| `drone_decay_s`               | 1.6     | pluck decay time constant                            |
| `drone_harmonics`             | 16      | harmonics per string                                 |
| `drone_first_string`          | "Pa"    | "Pa", "Ma" or "Ni"                                   |

**Headphones rule:** the drone must never reach the microphone (speaker bleed makes the
tracker lock onto Sa/Pa of the drone). Live use = headphones; evaluation scripts never
play the drone. `demo_logic.py --drone-bleed DB` exists only to demonstrate the problem.

Estimator constructors take these as keyword defaults (`Config.yin_threshold` etc.), and
`pitch.make_estimators(cfg)` builds all of them from one Config.

## What was built (summary, all five stages done)

- **DSP basics** (`dsp.py`, `synth.py`, `audio_io.py`): framing, Hann window, RMS/dBFS,
  FFT spectrum, spectrogram, silence gate, add-noise-at-SNR, test signals, mic and WAV I/O.
- **Pitch** (`pitch/`): FFT peak (+ HPS), normalised autocorrelation via Wiener-Khinchin,
  YIN; pYIN wrapper for reference only. About 0.2 ms per frame each.
- **Musical logic**: `swara.py` (cents from Sa, 12 swaras x 3 octaves, equal/just,
  `estimate_sa`), `ornament.py` (silent/stable/glide/ornament, causal, live or batch),
  `session.py` (judges held notes only; stats, CSV, plots), `drone.py` (additive tanpura).
- **Live app** (`app/`): PySide6 + pyqtgraph, worker thread, 30 fps UI, `--wav` file mode.
- **Evaluation** (`eval/`, `scripts/run_all_experiments.py`): E1-E6, reproducible, ~30 s,
  writes `results/RESULTS.md`. E4 (real voice vs pYIN) needs WAVs in `data/voice/`.
- **Docs**: README, `docs/algorithms.md`, `docs/user_guide.md`, `docs/demo_script.md`.

Main known weaknesses (see RESULTS.md): FFT peak/HPS fail on missing fundamentals; HPS
fails on pure sines; YIN with threshold 0.15 goes unvoiced below ~7 dB SNR (0.25-0.30
looked free on synthetic data; verify on real voice before changing); meends slower
than 200 c/s are judged as held notes; mic-to-display latency (~150-180 ms) is estimated,
not measured, and dominated by the MME input buffer on Windows.

Bugs fixed along the way (keep in mind when editing): parabolic interpolation is clamped
to +-0.5 (unclamped it produced negative Hz at range edges); estimators reject zero-energy
frames even with the gate off; held runs are edge-trimmed before judging; the UI timer is
a PreciseTimer (Windows coarse timer gave 21 fps).

## Stage plan

Status: all five stages done. Live app threading: the audio callback only queues blocks;
`ProcessingWorker` (thread) does all DSP and emits `FrameResult`s; the Qt UI drains that
queue with a 33 ms QTimer and only draws. Settings reach the worker through its command
queue (never by touching its state directly). Keep Qt out of `worker.py`/`sources.py`.
Stage 2 notes: Known estimator behaviour (from `tests/test_pitch.py` and
`scripts/compare_pitch.py`): FFT peak makes octave errors on weak/missing fundamentals;
HPS fixes weak-f0 cases but fails on pure sines and missing fundamentals; FFT methods
false-voice ~15-20% of loud pink-noise frames; YIN goes unvoiced below ~7 dB SNR and
reads ~2 cents early on fast glides. Autocorrelation and YIN handle missing fundamentals.

1. **Setup and DSP basics** — structure, config, framing, windowing, RMS/dBFS, FFT,
   silence gate, spectrogram, add-noise-at-SNR, test-signal synthesis, audio I/O,
   DSP demo plot, tests.
2. **Pitch estimators** — three from-scratch estimators (e.g. FFT peak / harmonic
   method, autocorrelation, YIN) plus librosa's pYIN as the reference tracker.
3. **Musical layer** — map frequency to swara and cents from the chosen Sa
   (`1200 * log2(f / Sa)`), tanpura-style drone, ornament-aware logic (glides are not
   errors), session statistics.
4. **Live desktop app** — PySide6 + pyqtgraph UI: live pitch curve on swara lines,
   current swara, cents meter, drone controls.
5. **Evaluation and documentation** — accuracy, latency and noise-robustness
   experiments for each estimator, plots in `results/`, report in `docs/`.
