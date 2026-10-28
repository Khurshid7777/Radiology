"""SD-085: speed and gain printed in a footer close to the rhythm row and the page edge
(ECG-Image-Kit: ~6 mm under the trace, 3 mm from the edge) are read."""

import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, scan

cv2 = pytest.importorskip("cv2")
pytest.importorskip("rapidocr_onnxruntime")


def test_settings_in_a_tight_footer_are_read(synth_mv):
    """Before SD-085 the footer band started 0.6 row pitch under the rhythm row, past the
    page's edge: no band, speed not set (SPEED_UNCERTAIN)."""
    page, gt = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header_settings=False))
    rb = max(t.baseline_px for t in gt.traces)
    page = page[:int(rb + 190)].copy()                   # the page ends 19 mm under the rhythm baseline
    y = int(rb + 150)
    cv2.putText(page, "25mm/s", (300, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(page, "10mm/mV", (700, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    dbg = ScanDebug()
    res = scan(cv2.imencode(".png", page)[1].tobytes(), debug=dbg)
    assert "25mm/s" in dbg.ocr_texts
    assert res.calibration.paper_speed_mm_s == 25 and res.calibration.speed_source == "header_ocr"
    assert "SPEED_UNCERTAIN" not in res.warnings
