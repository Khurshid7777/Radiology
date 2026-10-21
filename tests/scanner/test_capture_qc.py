"""M5: frame checks of the camera screen (TZ §6.1, SD-014, SD-046).

The reference frames in ``fixtures/capture_qc`` (1280-px previews of synthetic photos,
JPEG) carry the metrics and hints the checks must give. The Dart implementation in
the app runs the same frames against the same JSON (mobile/ecg_scanner/test).
"""

import json
from pathlib import Path

import pytest

from ecgscan.capture_qc import check_frame, config

cv2 = pytest.importorskip("cv2")
FIX = Path(__file__).parent / "fixtures" / "capture_qc"
CASES = sorted(p.stem for p in FIX.glob("*.json") if p.stem != "config")
# JPEG frames: another decoder (the app's package:image vs libjpeg) shifts pixels by
# ~0.4 grey levels, which moves the sheet share of a tilted frame by up to 0.23 and the
# tilt by 0.35 deg. The hints must still agree; the numbers are checked exactly on the
# lossless frame in exact/ (Dart and Python agree there to the last digit).
TOL = {"px_per_mm": 0.02, "sharpness": 0.10, "brightness": 0.01}
ABS = {"tilt_deg": 0.5, "sheet": 0.25, "glare": 0.005}
DECIMALS = {"brightness": 3, "glare": 5, "sharpness": 3, "tilt_deg": 3, "period_px": 3, "px_per_mm": 3, "sheet": 4}


def test_fixture_config_is_the_live_config():
    """The Dart tests read the thresholds from fixtures/capture_qc/config.json."""
    assert json.loads((FIX / "config.json").read_text()) == config()


def test_reference_frames_exist():
    assert {"clean_12", "far_6", "glare", "blur_6", "dark", "far_small_on_desk", "rotated_9",
            "grey_grid"} <= set(CASES)


@pytest.mark.parametrize("name", CASES)
def test_reference_frame(name):
    ref = json.loads((FIX / f"{name}.json").read_text())
    rgb = cv2.imread(str(FIX / f"{name}.jpg"))[..., ::-1]
    q = check_frame(rgb, scale_to_capture=ref["scale_to_capture"])
    assert set(q.hints) == set(ref["expected_hints"]), q
    m = ref["metrics"]
    for key, rel in TOL.items():
        assert q.to_dict()[key] == pytest.approx(m[key], rel=rel), key
    for key, tol in ABS.items():
        assert q.to_dict()[key] == pytest.approx(m[key], abs=tol), key


def test_exact_frame():
    ref = json.loads((FIX / "exact" / "parity.json").read_text())
    rgb = cv2.imread(str(FIX / "exact" / "parity.png"))[..., ::-1]
    got = check_frame(rgb, scale_to_capture=ref["scale_to_capture"]).to_dict()
    assert got["hints"] == ref["metrics"]["hints"]
    for key, dec in DECIMALS.items():
        assert got[key] == pytest.approx(ref["metrics"][key], abs=0.6 * 10 ** -dec), key


def test_px_per_mm_scales_to_the_captured_photo():
    rgb = cv2.imread(str(FIX / "clean_12.jpg"))[..., ::-1]
    a = check_frame(rgb, scale_to_capture=1.0).px_per_mm
    b = check_frame(rgb, scale_to_capture=3.0).px_per_mm
    assert b == pytest.approx(3 * a, rel=1e-6)


def test_blank_frame_finds_no_grid():
    import numpy as np

    q = check_frame(np.full((720, 1280, 3), 200, np.uint8))
    assert q.px_per_mm is None and q.tilt_deg is None and q.sheet == 0.0
    assert {"closer", "frame"} <= set(q.hints)


def test_config_endpoint_serves_thresholds_and_hints(monkeypatch):
    from fastapi.testclient import TestClient

    from ecgscan_service.app import app
    from ecgscan_service.router import KEYS_ENV

    monkeypatch.setenv(KEYS_ENV, "app:k1")
    body = TestClient(app).get("/scans/config/capture-qc", headers={"X-API-Key": "k1"}).json()
    assert body["config"] == config()
    assert body["hints"]["ru"]["closer"].startswith("Приблизьтесь")
    assert set(body["hints"]["uz"]) == {"closer", "glare", "steady", "level", "light", "frame", "ready"}


def test_scan_records_the_server_side_frame_check(synth_mv):
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout
    from ecgscan.pipeline import ScanHints, scan

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    res = scan(cv2.imencode(".png", img)[1].tobytes(), hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
    assert res.quality.sharpness > 15 and res.quality.glare_fraction == 0.0


def test_server_path_gives_the_same_light_metrics():
    """grid=False (the server's record of each photo) reduces the bytes first: same numbers."""
    rgb = cv2.imread(str(FIX / "glare.jpg"))[..., ::-1]
    full, light = check_frame(rgb), check_frame(rgb, grid=False)
    assert (light.width, light.height) == (full.width, full.height)
    for key in ("brightness", "glare", "sharpness"):
        assert getattr(light, key) == pytest.approx(getattr(full, key), rel=1e-4, abs=1e-5), key
    assert light.period_px is None and light.tilt_deg is None
