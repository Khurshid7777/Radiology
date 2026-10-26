"""SD-071 (TZ addendum §18.1, §18.13): how the phone took the photo travels with it, and
the original the server keeps is the bytes the phone sent."""

import hashlib
import json

import numpy as np
import pytest

from ecgscan.contract import Capture
from ecgscan.pipeline import ScanHints, scan

cv2 = pytest.importorskip("cv2")

CAPTURE = {
    "capture_mode": "custom_camera", "auto_capture": True,
    "corners": {"auto": [[10, 12], [590, 10], [592, 410], [8, 408]],
                "final": [[12, 12], [588, 12], [590, 408], [10, 406]], "edited_by_user": True},
    "camera_tilt_deg": 4.2,
    "qc": {"brightness": 171.0, "glare": 0.0, "sharpness": 640.0, "shadow": 0.06, "px_per_mm": 11.2,
           "tilt_deg": 0.3, "sheet_area": 0.61, "passed": True, "failed_checks": []},
    "mask_method": "template_zone",
}


def test_the_capture_record_reaches_the_result(synth_mv):
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout

    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10))
    jpg = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    res = scan(jpg, hints=ScanHints(client="flutter", capture=CAPTURE))
    assert res.source.capture == Capture.model_validate(CAPTURE)
    assert res.source.client_crop == CAPTURE["corners"]["final"]
    assert res.source.original_sha256 == hashlib.sha256(jpg).hexdigest()
    again = json.loads(res.model_dump_json())["source"]["capture"]
    assert again["qc"]["passed"] is True and again["corners"]["edited_by_user"] is True


def test_capture_record_is_checked():
    with pytest.raises(ValueError):
        Capture.model_validate({**CAPTURE, "capture_mode": "native_scanner"})    # variant A was declined
    with pytest.raises(ValueError):
        Capture.model_validate({**CAPTURE, "corners": {"final": [[0, 0], [1, 1]]}})
    with pytest.raises(ValueError):
        Capture.model_validate({**CAPTURE, "patient_name": "x"})                  # nothing beyond the contract


def test_the_doctors_quarter_turn_is_applied_by_the_server(synth_mv):
    """A page photographed sideways and turned on the crop screen reads like an upright one;
    the stored original is the sideways photo as sent."""
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout

    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10))
    sideways = cv2.imencode(".png", np.rot90(page, 1))[1].tobytes()       # turned counter-clockwise
    upright = scan(cv2.imencode(".png", page)[1].tobytes())
    turned = scan(sideways, hints=ScanHints(capture={**CAPTURE, "rotate_deg": 90}))
    assert turned.status == upright.status and turned.layout.name == "3x4+rhythm"
    assert turned.source.original_sha256 == hashlib.sha256(sideways).hexdigest()
    a, b = upright.leads["II"].array(), turned.leads["II"].array()
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.corrcoef(a[ok], b[ok])[0, 1] > 0.99
