"""Structured measurements produced by Layer 3 (TZ §6).

These are the quantitative features that Layer 4 (findings) and Layer 5 (causal
analysis) consume. All durations in milliseconds, amplitudes in microvolts,
angles in degrees, rate in bpm.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Intervals(BaseModel):
    """Global interval measurements (TZ §6)."""

    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    # Rate-corrected QT with three formulas (TZ §6).
    qtc_bazett_ms: float | None = None
    qtc_fridericia_ms: float | None = None
    qtc_framingham_ms: float | None = None
    qt_lead: str | None = None           # lead the QT was measured in (longest QT)
    qt_source: str | None = None         # "tangent" (multi-lead) | "dwt" (fallback)


class Axes(BaseModel):
    """Electrical axes (TZ §6)."""

    p_deg: float | None = None
    qrs_deg: float | None = None
    t_deg: float | None = None


class STMeasurement(BaseModel):
    """ST-segment deviation per lead at J and J+60/80 ms (TZ §6)."""

    lead: str
    j_point_uv: float | None = None
    j60_uv: float | None = None
    j80_uv: float | None = None
    qrs_net_uv: float | None = None   # signed net QRS deflection (concordance checks)
    s_depth_uv: float | None = None   # depth of the S wave below the PR baseline (proportional ST rules)
    r_height_uv: float | None = None  # height of the R wave above the PR baseline


class Voltages(BaseModel):
    """Per-lead QRS voltages for hypertrophy criteria (µV / ms; TZ §7.4)."""

    r_avl_uv: float | None = None
    s_v3_uv: float | None = None
    r_v5_uv: float | None = None
    r_v6_uv: float | None = None
    s_v1_uv: float | None = None
    s_v2_uv: float | None = None
    r_v1_uv: float | None = None
    s_v6_uv: float | None = None
    r_max_limb_uv: float | None = None          # tallest R or deepest S in any limb lead
    intrinsicoid_v5_v6_ms: float | None = None  # QRS onset -> R peak (V5/V6, the later one)
    p_terminal_v1_uv_ms: float | None = None    # negative terminal P force in V1 (|µV| x ms)


class TWave(BaseModel):
    """Per-lead T-wave descriptor measured on the median beat (TZ §6; D61).

    ``half_width_ms`` is the width at half the peak amplitude and ``asymmetry``
    the up-slope / down-slope time ratio: a hyperkalaemic T is tall, narrow and
    symmetric ("tented"), a normal tall T is broad-based and asymmetric, and a
    deflection narrower than ~40 ms in the T window is a QRS remnant, not a T."""

    lead: str
    amplitude_uv: float
    qrs_ratio: float | None = None
    half_width_ms: float | None = None
    asymmetry: float | None = None


class Morphology(BaseModel):
    """Waveform morphology descriptors (TZ §6)."""
    # Initial Q per lead on the 12-lead median beat: (Q ms, Q depth µV, R µV, QS) (D36).
    q_waves: dict[str, tuple[float, float, float, bool]] = Field(default_factory=dict)
    # Right precordial ST profile (V1, V2): (J, J+40, J+80, T min) µV — Brugada type 1 (D89).
    right_precordial_st: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)
    # Terminal S in I/V6 on the median beat, µV — true RBBB excludes Brugada (D90).
    terminal_s_lateral: dict[str, float] = Field(default_factory=dict)

    st: list[STMeasurement] = Field(default_factory=list)
    t_waves: list[TWave] = Field(default_factory=list)
    # Leads exhibiting each morphology, for evidence trails.
    peaked_t_leads: list[str] = Field(default_factory=list)
    flat_t_leads: list[str] = Field(default_factory=list)
    inverted_t_leads: list[str] = Field(default_factory=list)
    u_wave_leads: list[str] = Field(default_factory=list)
    pathological_q_leads: list[str] = Field(default_factory=list)
    # Delta-wave descriptor (D51): per lead, the time from QRS onset until the
    # slope first reaches half of that lead's maximal QRS slope (ms); a slurred
    # pre-excited upstroke is slow for 40-100 ms, a normal QRS for 15-35 ms.
    delta_ms_by_lead: dict[str, float] = Field(default_factory=dict)
    delta_ms: float | None = None         # median over the measurable leads


class Measurements(BaseModel):
    """Full Layer-3 output (TZ §6)."""

    heart_rate_bpm: float | None = None
    rhythm_regular: bool | None = None
    rr_std_ms: float | None = None            # RR interval variability
    rr_cv: float | None = None                # RR coefficient of variation
    hrv_sdnn_ms: float | None = None
    rr_mad_cv: float | None = None            # robust RR dispersion (1.4826·MAD / median)
    max_rr_ms: float | None = None            # longest RR gap (pause detection, §7.1/§10)
    pause_ratio: float | None = None          # longest RR / median RR (dropped-beat cue)
    # Median correlation of each QRS to the median beat (§5): noise has no repeatable beat.
    qrs_template_consistency: float | None = None
    # Median over beats of the best correlation to any other beat (D89): high when the
    # strip has several stable QRS shapes that blend a single median template.
    qrs_partner_consistency: float | None = None
    p_wave_ratio: float | None = None         # fraction of beats with an organised P (AF gate)
    fwave_index: float | None = None          # fibrillatory-wave band-power index (AF evidence, D58)
    p_onset_ratio: float | None = None        # fraction of beats with a refined P onset (PR)
    pr_std_ms: float | None = None            # beat-to-beat PR variability (AV dissociation gate)
    qrs_v1_terminal_positive: bool | None = None  # terminal forces in V1 (R' -> RBBB)
    qrs_v6_terminal_positive: bool | None = None  # terminal forces in V6 (broad R -> LBBB)
    qrs_amplitude_limb_uv: float | None = None    # median limb-lead QRS peak-to-peak
    atrial_rate_bpm: float | None = None          # P/flutter-wave rate (TZ §7.1)
    av_dissociation: bool | None = None           # complete AV block signal (TZ §7.1)
    p_wave_duration_ms: float | None = None       # atrial enlargement (P-mitrale)
    p_wave_amplitude_uv: float | None = None      # atrial enlargement (P-pulmonale)
    ml_probabilities: dict[str, float] | None = None   # learnable layer, per finding code
    n_beats: int = 0
    r_peaks_samples: list[int] = Field(default_factory=list)
    # The subset of r_peaks_samples added from the multi-lead envelope (D75): marked at the
    # envelope peak, not the rhythm lead's R, so shape comparisons must allow a shift (D87).
    r_peaks_recovered: list[int] = Field(default_factory=list)
    intervals: Intervals = Field(default_factory=Intervals)
    voltages: Voltages = Field(default_factory=Voltages)
    axes: Axes = Field(default_factory=Axes)
    morphology: Morphology = Field(default_factory=Morphology)
    # Sokolow-Lyon voltage sum for LVH assessment (µV).
    sokolow_lyon_uv: float | None = None
    # Digitised page only: whether the final beats match the column detector's (D92).
    # False = the rhythm is uncertain; None = nothing to compare.
    page_beat_agreement: bool | None = None
    notes: list[str] = Field(default_factory=list)  # TODO / low-confidence markers
