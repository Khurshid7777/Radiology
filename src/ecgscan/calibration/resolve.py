"""Combine calibration sources into the ScanResult calibration (TZ §5.4, SD-011).

Sources and their weight:

* **user** (doctor's confirmation) — final;
* **header OCR** ("25 mm/s 10 mm/mV") — strong for speed and gain;
* **calibration pulse** — strong for gain (height), weak for speed (width, SD-011);
* **format registry** — reserved for recognised devices (TODO: no device signatures yet).

Speed is *established* only by OCR or the user: a pulse-width guess never yields
``ok`` (a wrong speed halves or doubles the heart rate, TZ §5.4). Gain is
established by the pulse, alone or agreeing with OCR; OCR alone or a disagreement
-> ``needs_confirmation``. Nothing is defaulted silently: an unestablished value is
``None`` and the best candidate (if any) is what the traces are scaled with, with
the warning saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ecgscan.contract import Calibration, Candidate


@dataclass
class CalibrationDecision:
    calibration: Calibration
    warnings: list[str] = field(default_factory=list)
    speed_for_traces: float = 25.0      # what the traces are scaled with (established or best candidate)
    gain_for_traces: float = 10.0
    established: bool = False


def snap(value: float | None, standards: list[float], tol: float) -> float | None:
    if value is None or value <= 0:
        return None
    best = min(standards, key=lambda s: abs(s - value))
    return float(best) if abs(best - value) <= tol * best else None


def resolve(*, px_per_mm: float, grid_residual_mm: float | None, pulse: dict | None, ocr_speeds: list[float],
            ocr_gains: list[float], cfg: dict, pulse_width_s: float | None = 0.2,
            user_speed: float | None = None, user_gain: float | None = None) -> CalibrationDecision:
    speeds, gains = cfg["speeds_mm_s"], cfg["gains_mm_mv"]
    tol = cfg["value_tolerance"]
    pc, oc = cfg["pulse"], cfg["ocr"]
    warnings: list[str] = []

    # --- candidates per source
    speed_c: list[Candidate] = []
    gain_c: list[Candidate] = []
    ocr_s = {snap(v, speeds, tol) for v in ocr_speeds} - {None}
    ocr_g = {snap(v, gains, tol) for v in ocr_gains} - {None}
    for v in sorted(ocr_s):
        speed_c.append(Candidate(value=v, source="header_ocr", confidence=oc["conf"] / len(ocr_s)))
    for v in sorted(ocr_g):
        gain_c.append(Candidate(value=v, source="header_ocr", confidence=oc["conf"] / len(ocr_g)))
    pulse_gain = pulse_speed = None
    if pulse:
        pulse_gain = snap(pulse["height_mm"], gains, tol)
        if pulse_gain is not None:
            conf = pc["gain_conf_multi"] if pulse["n_agree"] >= pc["min_agreeing"] else pc["gain_conf_single"]
            gain_c.append(Candidate(value=pulse_gain, source="calibration_pulse", confidence=conf))
        if pulse_width_s:
            pulse_speed = snap(pulse["width_mm"] / pulse_width_s, speeds, tol)
            if pulse_speed is not None:
                speed_c.append(Candidate(value=pulse_speed, source="calibration_pulse", confidence=pc["speed_conf"]))

    # --- speed
    speed = speed_src = None
    if user_speed is not None:
        speed, speed_src = float(user_speed), "user"
    elif len(ocr_s) == 1:
        (v,) = ocr_s
        if pulse_speed is not None and pulse_speed != v:
            warnings.append("SPEED_SOURCES_DISAGREE")
        else:
            speed, speed_src = v, "header_ocr"
    else:
        if len(ocr_s) > 1 or (pulse_speed is not None and ocr_s and pulse_speed not in ocr_s):
            warnings.append("SPEED_SOURCES_DISAGREE")
    if speed is None:
        warnings.append("SPEED_UNCERTAIN")

    # --- gain
    gain = gain_src = None
    if user_gain is not None:
        gain, gain_src = float(user_gain), "user"
    elif pulse_gain is not None and (not ocr_g or ocr_g == {pulse_gain}):
        gain, gain_src = pulse_gain, "calibration_pulse"
    else:
        if (pulse_gain is not None and ocr_g and pulse_gain not in ocr_g) or len(ocr_g) > 1:
            warnings.append("GAIN_SOURCES_DISAGREE")
    if gain is None:
        warnings.append("GAIN_UNCERTAIN")

    def best(cands: list[Candidate], fallback: float) -> float:
        return max(cands, key=lambda c: c.confidence).value if cands else fallback

    speed_t = speed if speed is not None else best(speed_c, speeds[0])
    gain_t = gain if gain is not None else best(gain_c, gains[1] if len(gains) > 1 else gains[0])
    conf_parts = [1.0 if speed_src == "user" else (oc["conf"] if speed is not None else 0.0),
                  1.0 if gain_src == "user" else (max((c.confidence for c in gain_c if c.value == gain), default=0.0)
                                                  if gain is not None else 0.0)]
    cal = Calibration(
        px_per_mm_x=round(px_per_mm, 4), px_per_mm_y=round(px_per_mm, 4),
        grid_residual_mm=None if grid_residual_mm is None else round(grid_residual_mm, 4),
        paper_speed_mm_s=None if speed is None else int(speed), gain_mm_mv=None if gain is None else int(gain),
        speed_source=speed_src, gain_source=gain_src, speed_candidates=speed_c, gain_candidates=gain_c,
        confidence=round(min(conf_parts), 3))
    return CalibrationDecision(cal, warnings, float(speed_t), float(gain_t), speed is not None and gain is not None)
