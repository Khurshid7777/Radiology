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
CASES = sorted(p.stem for p in FIX.glob("*.json"))
# How far an implementation (another JPEG decoder, float order) may drift.
TOL = {"px_per_mm": 0.02, "sharpness": 0.10, "brightness": 0.01}
ABS = {"tilt_deg": 0.3, "sheet": 0.05, "glare": 0.002}


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
