"""Default parameters for Swara-Aware.

Audio concepts: sampling rate, framing (frame size and hop size), silence threshold in
dBFS, and pitch tolerances in cents.

Every tunable number in the project lives here. Other modules receive a ``Config``
instance (or explicit arguments) instead of hard-coding values. The dataclass is frozen,
so to change a value make a modified copy::

    from dataclasses import replace
    cfg = replace(Config(), frame_size=4096)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All tunable defaults for analysis and feedback.

    Attributes:
        sample_rate: Samples per second (Hz). 44100 is the CD standard.
        frame_size: Samples per analysis frame. 2048 at 44.1 kHz is about 46 ms,
            long enough to hold several periods of a 70 Hz voice.
        hop_size: Samples between the starts of consecutive frames (75% overlap).
        fmin: Lowest pitch to search for (Hz).
        fmax: Highest pitch to search for (Hz).
        silence_threshold_dbfs: Frames whose RMS level is below this are silence.
        in_tune_cents: |deviation| at or below this counts as in tune.
        close_cents: |deviation| at or below this counts as close.
        stable_min_ms: A pitch must hold this long to count as a held swara.
        glide_threshold_cents_per_s: Pitch moving faster than this is a glide (meend),
            which is not judged as a mistake.
        yin_threshold: YIN absolute threshold: first CMNDF dip below this is the period.
        acf_voicing_threshold: Autocorrelation peak height needed to call a frame voiced.
        acf_peak_ratio: Autocorrelation picks the first peak at least this fraction of
            the highest peak (prefers the shortest true period, avoids octave-low errors).
        fft_zero_pad_factor: FFT length = next power of two >= frame_size * this.
        fft_voicing_threshold: FFT confidence (peak prominence) needed to call voiced.
        hps_harmonics: Number of harmonics multiplied in the Harmonic Product Spectrum.
        tuning_mode: "equal" (100-cent steps) or "just" (whole-number frequency ratios).
        a4_hz: Reference pitch for note names (A4), used when choosing Sa as e.g. "C#3".
        min_confidence: Frames with lower pitch confidence are treated as silent.
        median_filter_frames: Length of the causal median filter on the cents track
            (removes single-frame outliers such as octave jumps).
        slope_window_ms: Window for the local slope (cents per second) and the
            stable-range check. 25 cents in 120 ms is ~200 cents/s, matching the
            glide threshold: anything moving faster cannot look stable.
        ornament_window_ms: Longer window for the oscillation test. Must cover at
            least one gamaka cycle: 350 ms handles oscillations of 3 Hz and faster.
        stable_max_range_cents: Max pitch range within the slope window for "stable".
        ornament_min_range_cents: Min range within the ornament window for "ornament";
            smaller oscillations are vibrato and count as stable (judged on the mean).
        ornament_max_drift_cents_per_s: Max net drift (fitted slope over the ornament
            window) for an oscillation. Higher than the glide threshold on purpose:
            a window holding ~1 oscillation cycle has an inflated fitted slope.
        ornament_turn_fraction: An oscillation must travel back at least this fraction
            of its range ("back-travel"); a monotonic glide has back-travel ~0.
        drone_tempo_bpm: Tanpura plucks per minute.
        drone_volume: Drone output gain, 0 to 1.
        drone_decay_s: Time constant of each pluck's exponential decay.
        drone_harmonics: Harmonics synthesised per tanpura string.
        drone_first_string: "Pa" or "Ma" (or "Ni"), the first string of the cycle.
    """

    sample_rate: int = 44100
    frame_size: int = 2048
    hop_size: int = 512
    fmin: float = 70.0
    fmax: float = 1000.0
    silence_threshold_dbfs: float = -50.0
    in_tune_cents: float = 15.0
    close_cents: float = 30.0
    stable_min_ms: float = 250.0
    glide_threshold_cents_per_s: float = 200.0
    # --- Pitch estimators (Stage 2) ---
    yin_threshold: float = 0.15
    acf_voicing_threshold: float = 0.5
    acf_peak_ratio: float = 0.9
    fft_zero_pad_factor: int = 4
    fft_voicing_threshold: float = 0.8
    hps_harmonics: int = 4
    # --- Swara mapping (Stage 3) ---
    tuning_mode: str = "equal"
    a4_hz: float = 440.0
    # --- Ornament-aware logic (Stage 3) ---
    min_confidence: float = 0.5
    median_filter_frames: int = 5
    slope_window_ms: float = 120.0
    ornament_window_ms: float = 350.0
    stable_max_range_cents: float = 25.0
    ornament_min_range_cents: float = 40.0
    ornament_max_drift_cents_per_s: float = 250.0
    ornament_turn_fraction: float = 0.3
    # --- Tanpura drone (Stage 3) ---
    drone_tempo_bpm: float = 60.0
    drone_volume: float = 0.5
    drone_decay_s: float = 1.6
    drone_harmonics: int = 16
    drone_first_string: str = "Pa"

    def __post_init__(self) -> None:
        """Check that the values make sense together."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.hop_size <= self.frame_size:
            raise ValueError("hop_size must be in (0, frame_size]")
        if not 0 < self.fmin < self.fmax < self.sample_rate / 2:
            raise ValueError("need 0 < fmin < fmax < Nyquist (sample_rate / 2)")
        if not 0 < self.in_tune_cents <= self.close_cents:
            raise ValueError("need 0 < in_tune_cents <= close_cents")
        if self.tuning_mode not in ("equal", "just"):
            raise ValueError('tuning_mode must be "equal" or "just"')
        if self.drone_first_string not in ("Pa", "Ma", "Ni"):
            raise ValueError('drone_first_string must be "Pa", "Ma" or "Ni"')
        if not 0.0 <= self.drone_volume <= 1.0:
            raise ValueError("drone_volume must be in [0, 1]")

    @property
    def frame_duration_ms(self) -> float:
        """Length of one frame in milliseconds."""
        return 1000.0 * self.frame_size / self.sample_rate

    @property
    def hop_duration_ms(self) -> float:
        """Time between frame starts in milliseconds (the update interval)."""
        return 1000.0 * self.hop_size / self.sample_rate

    @property
    def min_lag(self) -> int:
        """Shortest period (in samples) to search, from fmax."""
        return int(self.sample_rate // self.fmax)

    @property
    def max_lag(self) -> int:
        """Longest period (in samples) to search, from fmin."""
        return int(self.sample_rate // self.fmin)
