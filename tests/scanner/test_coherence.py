"""SD-117: beat coherence of a lead and the LEAD_INCOHERENT flag (off by default)."""

import numpy as np

from ecgscan.contract import ScanResult, Trace
from ecgscan.validate.coherence import beat_times, coherence_by_lead, lead_coherence

FS = 500.0


def _beats(n_s: float, hr: float = 60.0, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    t = np.arange(int(n_s * FS)) / FS
    x = np.zeros_like(t)
    for b in np.arange(0.4, n_s - 0.3, 60.0 / hr):
        x += np.exp(-((t - b) / 0.012) ** 2) - 0.3 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
    return x + noise * np.random.default_rng(seed).standard_normal(t.size)


def _tr(x, lead="II", start=0.0):
    return Trace.from_array(x, start_s=start, fs=FS, lead=lead, quality=1.0, method="dp_path")


def test_beats_are_found_and_a_clean_lead_repeats_itself():
    tr = _tr(_beats(10.0))
    beats = beat_times([tr], FS)
    assert len(beats) == 10 and np.allclose(np.diff(beats), 1.0, atol=0.02)
    assert lead_coherence(tr, beats, FS) > 0.97


def test_noise_instead_of_a_trace_does_not_repeat_itself():
    # the beats come from all leads at once: one wrong lead among several right ones does not move them
    leads = {n: _tr(_beats(10.0) * k, n) for n, k in zip(("I", "II", "III", "V1", "V2"), (1.0, 1.5, 0.7, 1.2, 0.9), strict=True)}
    leads["V3"] = _tr(np.random.default_rng(1).standard_normal(5000) * 0.3, "V3")
    leads["V4"] = _tr(np.zeros(5000), "V4")
    coh = coherence_by_lead(leads, [], FS)
    assert min(coh[n] for n in ("I", "II", "III", "V1", "V2")) > 0.97
    assert coh["V3"] < 0.5 and coh["V4"] == 0.0


def test_a_lead_with_too_few_beats_gets_no_verdict():
    assert coherence_by_lead({"II": _tr(_beats(1.8))}, [], FS)["II"] is None
    assert beat_times([], FS).size == 0


def test_the_flag_is_in_the_contract_and_the_languages():
    import json
    from pathlib import Path

    from ecgscan.contract.models import WARNING_CODES

    assert "LEAD_INCOHERENT" in WARNING_CODES
    for d in ("configs/scanner/i18n", "mobile/ecg_scanner/assets/i18n"):
        for lang in ("ru", "uz"):
            assert json.loads(Path(f"{d}/{lang}.json").read_text())["warning.LEAD_INCOHERENT"]
    assert json.loads(Path("src/ecgscan/contract/scan_result.schema.json").read_text())


def test_scan_flags_an_incoherent_lead_only_when_asked(synth_mv, monkeypatch):
    import copy

    import cv2

    from ecgscan import pipeline as P
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.config import load_yaml
    from ecgscan.layout import get_layout

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    data, hints = cv2.imencode(".png", img)[1].tobytes(), P.ScanHints(paper_speed_mm_s=25, gain_mm_mv=10)
    cfg = copy.deepcopy(load_yaml("pipeline.yaml"))
    assert cfg["validate"]["coherence_min"] is None                      # off by default
    base = P.scan(data, hints=hints, config=cfg)
    assert base.status == "ok" and "LEAD_INCOHERENT" not in base.warnings
    monkeypatch.setattr(P, "coherence_by_lead", lambda leads, rhythm, fs: {"V3": 0.5, "II": 0.99, "V1": None})
    assert P.scan(data, hints=hints, config=cfg).status == "ok"          # still off: nothing changes
    cfg["validate"]["coherence_min"] = 0.9
    res = P.scan(data, hints=hints, config=cfg)
    assert res.status == "needs_confirmation" and "LEAD_INCOHERENT" in res.warnings
    ScanResult.model_validate_json(res.model_dump_json())
    monkeypatch.setattr(P, "coherence_by_lead", lambda leads, rhythm, fs: {"V3": 0.95, "V1": None})
    assert P.scan(data, hints=hints, config=cfg).status == "ok"          # above the line, or no verdict: untouched
