"""SD-130: the real-use quality report and the doctor's labels hold numbers only."""

import json
import sys
from pathlib import Path

from ecgscan_service.store import ScanStore

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "scanner"))
import quality_report as qr

SHA = "ab" * 32


def _fill(db: Path, make_result):
    st = ScanStore(db)
    specs = [("ok", {}), ("ok", {}), ("needs_confirmation", {"warnings": ["LEAD_INCOHERENT"]}),
             ("rejected", {"status": "rejected", "reject_reasons": ["GRID_NOT_FOUND"]}),
             ("rejected", {"status": "rejected", "reject_reasons": ["LAYOUT_UNKNOWN"]})]
    for i, (status, over) in enumerate(specs):
        sid = f"scan-{i}-SECRET-ID"
        st.create(sid, owner="doctor-SECRET-KEY-NAME", sha256=SHA, hints={})
        res = make_result(**{"status": status, **over}) if status != "needs_confirmation" else make_result(status=status, **over)
        st.update(sid, status=status, result=json.loads(res.model_dump_json()))
    return st


def test_the_report_counts_and_leaks_nothing(tmp_path, result_factory):
    db, fb = tmp_path / "scans.db", tmp_path / "feedback.jsonl"
    _fill(db, result_factory)
    fb.write_text("\n".join(json.dumps(x) for x in [
        {"scan_id": "scan-2-SECRET-ID", "sha256": SHA, "by": "doctor-SECRET-KEY-NAME", "at": "t", "decision": "corrected",
         "changes": {"before": {"layout": "3x4+rhythm", "paper_speed_mm_s": None, "status": "needs_confirmation",
                                "warnings": ["LAYOUT_UNCERTAIN"]},
                     "after": {"layout": "6x2", "paper_speed_mm_s": 25.0}}, "comment": "PATIENT-NAME-COMMENT"},
        {"scan_id": "scan-0-SECRET-ID", "sha256": SHA, "by": "x", "at": "t", "decision": "accepted",
         "changes": {"before": {}, "after": {}}, "comment": None}]) + "\n")
    rep = qr.report(qr._results(db), qr._feedback(fb))
    assert rep["scans"] == 5 and rep["status"]["ok"]["n"] == 2 and rep["status"]["rejected"]["n"] == 2
    assert rep["reject_reasons"]["GRID_NOT_FOUND"]["n"] == 1 and rep["warnings_on_read"]["LEAD_INCOHERENT"]["n"] == 1
    assert rep["doctor_decisions"]["corrected"]["n"] == 1 and rep["doctor_changed_fields"] == {"layout": 1, "paper_speed_mm_s": 1}
    assert rep["layout_labels"] == {"n": 1, "scanner_right": 0, "share_right": 0.0}
    lab = qr.labels(qr._feedback(fb))
    assert {x["field"] for x in lab} == {"layout", "paper_speed_mm_s"}
    blob = json.dumps(rep) + json.dumps(lab)
    for secret in ("SECRET-ID", SHA, "SECRET-KEY-NAME", "PATIENT-NAME-COMMENT", "scan-2"):
        assert secret not in blob, secret


def test_the_tool_reads_the_store_read_only(tmp_path, result_factory):
    db = tmp_path / "scans.db"
    _fill(db, result_factory)
    before = db.read_bytes()
    qr._results(db)
    assert db.read_bytes() == before


def test_the_capture_tips_exist_in_both_languages_and_in_the_app_copy():
    root = Path(__file__).resolve().parents[2]
    for base in ("configs/scanner/i18n", "mobile/ecg_scanner/assets/i18n"):
        for lang in ("ru", "uz"):
            d = json.loads((root / base / f"{lang}.json").read_text(encoding="utf-8"))
            assert d["capture.tips_title"] and d["capture.tips_body"].count("•") >= 5
            assert "6×2+" in d["reject.LAYOUT_UNKNOWN"] or "6×2+ritm" in d["reject.LAYOUT_UNKNOWN"]
    app_src = (root / "mobile/ecg_scanner/lib/ui/camera_screen.dart").read_text()
    assert "capture_tips" in app_src and "capture.tips_body" in app_src
