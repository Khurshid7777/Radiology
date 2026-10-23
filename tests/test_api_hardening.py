"""API hardening tests: bounded store, upload limit, temp cleanup."""

import os

from fastapi.testclient import TestClient

from cardiolens.api import service
from cardiolens.api.app import app

client = TestClient(app, headers={"X-API-Key": "test-key"})


def test_store_is_bounded_lru():
    # Fill well past the cap; store must not grow unbounded and must keep the
    # most-recent ids retrievable.
    ids = []
    for _ in range(service.STORE_MAX_ENTRIES + 25):
        r = client.post("/api/analyze", data={"case": "normal", "mode": "A"})
        ids.append(r.json()["analysis_id"])
    assert len(service._STORE) <= service.STORE_MAX_ENTRIES
    # The newest analysis is still available; a long-evicted one is gone.
    assert service.get_analysis(ids[-1]) is not None
    assert service.get_analysis(ids[0]) is None


def test_upload_too_large_rejected():
    big = b"x" * (service.MAX_UPLOAD_BYTES + 1)
    r = client.post("/api/analyze", data={"mode": "A", "sampling_rate_hz": "500"},
                    files={"files": ("huge.csv", big, "text/csv")})
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]


def test_artifact_endpoints_do_not_leak_tempfiles():
    # Track dirs created via tempfile.mkdtemp during artifact requests and assert
    # each is removed afterwards (the FileResponse background cleanup ran).
    import tempfile as _t

    created: list[str] = []
    real_mkdtemp = _t.mkdtemp

    def tracking_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        created.append(d)
        return d

    aid = client.post("/api/analyze", data={"case": "hyperk", "mode": "A"}).json()["analysis_id"]
    _t.mkdtemp = tracking_mkdtemp
    try:
        for ext in ("report.pdf", "waveform.svg", "graph.png"):
            assert client.get(f"/api/analysis/{aid}/{ext}").status_code == 200
    finally:
        _t.mkdtemp = real_mkdtemp

    assert created, "expected the endpoints to create temp dirs"
    still_there = [d for d in created if os.path.isdir(d)]
    assert still_there == [], f"temp dirs leaked: {still_there}"


def test_body_cap_takes_a_page_in_parts_for_the_scanner():
    """D145: /scans takes up to max_parts photos of one page on the capture board (SD-066); a
    body the size of all of them passes the cap, one past it does not, and CardioLens' own
    uploads keep their 64-MB budget per request (test_upload_too_large_rejected)."""
    from cardiolens.api.app import _SCANS_MAX_BYTES

    assert _SCANS_MAX_BYTES > service.MAX_UPLOAD_BYTES
    over = client.post("/api/analyze", headers={"Content-Length": str(_SCANS_MAX_BYTES + 2 * 1024 * 1024)},
                       content=b"")
    assert over.status_code == 413
    within = client.post("/scans", headers={"Content-Length": str(service.MAX_UPLOAD_BYTES + 2 * 1024 * 1024),
                                            "Content-Type": "multipart/form-data; boundary=b"}, content=b"")
    assert within.status_code != 413
