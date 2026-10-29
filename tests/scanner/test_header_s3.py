"""S3 (TZ 18.6): what a printed header says besides speed and gain, read without personal data;
the recorder's personal-data zones masked by the profile (the app's twin)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgscan.bench.headers import render_header
from ecgscan.calibration.header_fields import is_phi, parse_fields
from ecgscan.config import load_yaml
from ecgscan.masking import mask_zones, zone_polygon

MODELS = load_yaml("pipeline")["calibration"]["ocr"]["device_models"]
# matplotlib's own DejaVu Sans: the system's fonts differ between hosts, and the generator picks
# one of those it finds, so a seed alone does not fix the picture (SD-098).
BUNDLED_FONT = str(Path(__import__("matplotlib").__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf")
FIX = Path(__file__).parent / "fixtures" / "mask_zones"


@pytest.mark.parametrize("style", ["ru", "en", "uz"])
def test_every_field_of_a_clean_header(style):
    """The parser on the generator's own lines (no OCR): every field, and the 2 personal lines dropped."""
    for seed in range(12):
        c = render_header(seed, style=style)
        t = c.truth
        f = parse_fields([ln.text for ln in c.lines], models=MODELS)
        m = f.measurements
        assert (f.speeds[:1], f.gains[:1]) == ([t["speed"]], [t["gain"]])
        assert (m.hr_bpm, m.pr_ms, m.qrs_ms, m.qt_ms, m.qtc_ms, m.axis_deg) == \
            (t["hr"], t["pr"], t["qrs"], t["qt"], t["qtc"], t["axis"])
        assert (f.recorded_at, f.device_model, f.conclusion) == (t["recorded_at"], t["model"], t["conclusion"])
        assert f.phi_lines_dropped == 2
        kept = " ".join(f.safe_lines)
        assert t["phi"]["id"] not in kept and t["phi"]["dob"] not in kept
        assert not any(w in kept for w in t["phi"]["name"].split())


@pytest.mark.parametrize("line", [
    "Пациент: Каримов Акмал Рустамович   ID: 1234567",
    "F.I.Sh.: Karimov Akmal   ID: 1234567",
    "Name: Karimov Akmal   ID: 1234567",
    "Дата рождения: 05.02.2008   Пол: М",
    "D0B: 16.07.1979  Male",                 # the OCR's 0 for O (S3 bench: this line leaked)
    "D0B09.10.1974  Male",
    "Tug 'ilgan sana: 21.03.1949",           # a gap before the apostrophe (leaked too)
    "12.03.1961",                            # a date without a time: a birth date
    "Каримов Акмал Рустамович",              # an unlabelled name
])
def test_personal_data_lines_are_dropped(line):
    assert is_phi(line)


@pytest.mark.parametrize("line", [
    "ЧСС: 72 уд/мин   PQ: 160 мс   QRS: 92 мс", "25 мм/с  10 мм/мВ  0,05-150 Гц",
    "04.05.202314:42  Mindray BeneHeart R12",   # the recording's date (its time glued on)
    "Синусовый ритм. Нормальная ЭКГ.", "Sinus rhythm. Normal ECG.",
])
def test_recorder_lines_are_kept(line):
    assert not is_phi(line)


@pytest.mark.parametrize("lines, want", [
    (["4СС. 129 уд/мин  PQ. 201 MC QRS. 93 MC QT/QTC: 449/658 M"], (129, 201, 93, 449, 658)),
    (["ЧСС: 76 уд/мин  РQ: 168MC  QRS: 74 мC  с QT/QTс: 339/382 мс"], (76, 168, 74, 339, 382)),
    (["ЧСС: 120  уд/мин  PQ: ---  MC  QRS: 90 mc  QT/QTc: 330  330/467  MC"], (120, None, 90, 330, 467)),
    (["Vent rate 104 bpm  PR int 180  ms  QRS  s dur 129 ms"], (104, 180, 129, None, None)),
])
def test_measurements_through_the_ocr_s_distortions(lines, want):
    """What the Cyrillic recogniser returned on the S3 bench (mixed scripts, '.' for ':', stray boxes)."""
    m = parse_fields(lines).measurements
    assert (m.hr_bpm, m.pr_ms, m.qrs_ms, m.qt_ms, m.qtc_ms) == want


@pytest.mark.parametrize("line, want", [
    ("11.07.2024 08:02   ЭК12Т-01-P-Д", "ЭК12Т-01-Р-Д"),       # Latin P for Р (the S3 recogniser)
    ("ЭK12T-01-P-Д", "ЭК12Т-01-Р-Д"),                       # Latin K, T, P
    ("Мindray BeneHeart R12", "Mindray BeneHeart R12"),     # Cyrillic М in a Latin name
])
def test_a_model_read_with_look_alike_letters(line, want):
    f = parse_fields([line, "Синусовый ритм."], models=MODELS)
    assert f.device_model == want and f.conclusion == "Синусовый ритм."


def test_axis_through_the_ocr_s_distortions():
    assert parse_fields(["Ось Э0С: -15°  25mm/c"]).measurements.axis_deg == -15
    assert parse_fields(["Ось.-36°25 мм/C 10 мM/MB"]).measurements.axis_deg == -36
    m = parse_fields(["QT/QTc 383X499 ms  P-R-T axe$ 14  -30 19"]).measurements
    assert (m.qt_ms, m.qtc_ms, m.p_axis_deg, m.axis_deg, m.t_axis_deg) == (383, 499, 14, -30, 19)


def test_zone_mapping_matches_the_app_s_reference_vectors():
    ref = json.loads((FIX / "vectors.json").read_text(encoding="utf-8"))
    assert len(ref["cases"]) >= 48
    for c in ref["cases"]:
        got = np.array(zone_polygon(c["zone"], c["corners"], c["rotate_deg"], c["margin"]))
        assert np.abs(got - np.array(c["polygon"])).max() <= ref["tolerance_px"], c["name"]


def test_masking_paints_the_zone_and_only_it():
    photo = np.zeros((300, 400, 3), np.uint8)
    out = mask_zones(photo, [[0.0, 0.0, 0.5, 0.1]], [[0, 0], [400, 0], [400, 300], [0, 300]])
    assert out[10, 20].min() == 255 and out[10, 300].max() == 0 and out[200, 20].max() == 0
    turned = mask_zones(np.zeros((400, 300, 3), np.uint8), [[0.0, 0.0, 1.0, 0.1]],
                        [[0, 0], [300, 0], [300, 400], [0, 400]], rotate_deg=90)
    assert turned[200, 10].min() == 255 and turned[10, 150].max() == 0     # the photo's left strip


def test_a_profile_zone_outside_the_sheet_is_refused(tmp_path, monkeypatch):
    from ecgscan.layout import devices

    (tmp_path / "devices").mkdir()
    (tmp_path / "devices" / "bad.yaml").write_text("model: X\nphi_zones: [[0.5, 0.0, 1.2, 0.1]]\n")
    monkeypatch.setattr(devices, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(devices, "load_yaml", lambda name: __import__("yaml").safe_load((tmp_path / name).read_text()))
    with pytest.raises(ValueError, match="not inside the sheet"):
        devices.device_profiles()


def test_a_missing_or_altered_model_makes_ocr_unavailable(tmp_path):
    from ecgscan.calibration.header_ocr import ocr_available

    ok, why = ocr_available(str(tmp_path / "none.onnx"), "0" * 64)
    assert not ok and "not found" in why
    fake = tmp_path / "rec.onnx"
    fake.write_bytes(b"not the model")
    ok, why = ocr_available(str(fake), "0" * 64)
    assert not ok and "sha256" in why


def test_a_scanned_page_carries_the_header_fields_and_no_personal_data(synth_mv):
    """End to end: a Russian header over a 3x4 page. The result keeps the recorder's fields and
    names the engine; the result JSON has no name, ID or birth date in it."""
    import cv2

    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout
    from ecgscan.pipeline import scan

    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    h = render_header(5, style="ru", px_per_mm=10.0, font=BUNDLED_FONT)     # the same glyphs on every host
    top = np.full((h.image.shape[0], page.shape[1], 3), 255, np.uint8)
    top[:, :min(page.shape[1], h.image.shape[1])] = h.image[:, :page.shape[1]]
    res = scan(cv2.imencode(".png", np.vstack([top, page]))[1].tobytes())
    ho = res.header_ocr
    assert ho is not None and ho.engine == "rapidocr:cyrillic_PP-OCRv5_rec_mobile"
    t = h.truth
    m = ho.device_measurements
    assert (m.hr_bpm, m.qrs_ms, m.qt_ms, m.qtc_ms) == (t["hr"], t["qrs"], t["qt"], t["qtc"])
    assert ho.recorded_at == t["recorded_at"] and ho.device_model == t["model"]
    # The recorder's own text is kept as read: the recogniser may change a letter's case (ЭКГ/Экг).
    assert ho.device_text_conclusion.casefold() == t["conclusion"].casefold() and ho.phi_lines_dropped == 2
    blob = json.dumps(res.to_json_dict(), ensure_ascii=False)
    for token in [t["phi"]["id"], t["phi"]["dob"], *t["phi"]["name"].split()]:
        assert token not in blob


def test_the_app_gets_the_recorder_profiles(monkeypatch):
    from fastapi.testclient import TestClient

    from ecgscan.layout import devices
    from ecgscan_service.app import app
    from ecgscan_service.router import KEYS_ENV

    monkeypatch.setenv(KEYS_ENV, "dr:secret-1")
    prof = devices.DeviceProfile(id="ek12", manufacturer="Test", model="EK-12", layouts=("3x4+rhythm",),
                                 phi_zones=((0.0, 0.0, 0.5, 0.06),))
    monkeypatch.setattr(devices, "device_profiles", lambda: {"ek12": prof})
    r = TestClient(app).get("/scans/config/devices", headers={"X-API-Key": "secret-1"})
    assert r.status_code == 200
    assert r.json() == {"devices": [{"id": "ek12", "manufacturer": "Test", "model": "EK-12", "layouts": ["3x4+rhythm"],
                                     "phi_zones": [[0.0, 0.0, 0.5, 0.06]], "phi_zone_margin": 0.01}]}


def test_values_no_check_vouches_for_are_marked():
    """The S3 bench's only misread values: QTc taken from the QT/QTc pair's label, an axis that
    lost its minus sign or read 74° as 743. A real mismatch must not come from them."""
    m = parse_fields(["ЧСС: 72 уд/мин  QT/QTc: 380/416 мс  Ось: 45°"]).measurements
    assert (m.qt_ms, m.qtc_ms, m.axis_deg) == (380, 416, 45) and m.uncertain == ["axis_deg"]
    m = parse_fields(["ЧСС: 72 уд/мин  QT/QTc: 422/520 мс"]).measurements
    assert m.qtc_ms == 520 and "qtc_ms" in m.uncertain                 # off Bazett by > 40 ms
    m = parse_fields(["QT/QTc 383/499 ms  P-R-T axes 74 743 19"]).measurements
    assert m.axis_deg is None


def test_the_configured_ocr_engine_loads():
    """Never skipped: an engine that fails to load turns header OCR off silently (every speed
    from the header lost), as a newer rapidocr-onnxruntime did in CI and in the image (SD-098)."""
    from ecgscan.calibration.header_ocr import ocr_available
    from ecgscan.config import repo_root

    o = load_yaml("pipeline")["calibration"]["ocr"]
    ok, why = ocr_available(str(repo_root() / o["rec_model"]), o["rec_sha256"])
    assert ok, why
