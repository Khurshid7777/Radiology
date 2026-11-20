"""Real-use quality of the scanner service and the doctors' labels (SD-130). Derived numbers only.

    python tools/scanner/quality_report.py --db data/service/scans.db --feedback data/service/feedback.jsonl \
        [--out report.json] [--labels labels.jsonl]

Reads the service's SQLite store and ``feedback.jsonl`` read-only. The report holds counts and quantiles: statuses,
refusal reasons, warnings, how the speed / gain / layout were established, the scale, the doctor's decisions and what
the doctor changed. Never an image, a scan id, a hash, a comment, a user name or an API key. ``--labels`` writes one
line per correction: the field, the scanner's value, the doctor's value, the scanner's status and warning codes: the
truth labels for a future check of the scanner on real photos (layout detection, speed and gain).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

FIELDS = ("layout", "paper_speed_mm_s", "gain_mm_mv")


def _results(db: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT status, result FROM scans WHERE status IN "
                           "('ok','needs_confirmation','rejected','failed')").fetchall()
    finally:
        con.close()
    out = []
    for status, raw in rows:
        r = json.loads(raw) if raw else {}
        r["_status"] = status
        out.append(r)
    return out


def _feedback(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rate(c: Counter, n: int) -> dict:
    return {k: {"n": v, "share": round(v / n, 3)} for k, v in c.most_common()} if n else {}


def report(results: list[dict], feedback: list[dict]) -> dict:
    n = len(results)
    st = Counter(r["_status"] for r in results)
    reasons = Counter(x for r in results for x in r.get("reject_reasons", []))
    warns = Counter(x for r in results if r["_status"] != "rejected" for x in r.get("warnings", []))
    cal = [r.get("calibration") or {} for r in results if r["_status"] != "rejected"]
    layouts = Counter((r.get("layout") or {}).get("name") or "none" for r in results if r["_status"] != "rejected")
    ppm = [r["quality"]["px_per_mm"] for r in results if (r.get("quality") or {}).get("px_per_mm")]
    clients = Counter((r.get("source") or {}).get("client") or "unknown" for r in results)
    modality = Counter((r.get("source") or {}).get("modality") or "unknown" for r in results)
    decisions = Counter(f["decision"] for f in feedback)
    changed = Counter(k for f in feedback if f["decision"] == "corrected"
                      for k in (f.get("changes") or {}).get("after", {}))
    lay_fb = [f for f in feedback if "layout" in (f.get("changes") or {}).get("after", {})]
    lay_hit = sum(1 for f in lay_fb if (f["changes"].get("before") or {}).get("layout") == f["changes"]["after"]["layout"])
    ok_q = [r["quality"]["overall"] for r in results if r["_status"] == "ok" and (r.get("quality") or {}).get("overall") is not None]
    return {
        "scans": n,
        "status": _rate(st, n),
        "reject_reasons": _rate(reasons, st["rejected"]),
        "warnings_on_read": _rate(warns, n - st["rejected"]),
        "speed_source": _rate(Counter(c.get("speed_source") or "none" for c in cal), len(cal)),
        "gain_source": _rate(Counter(c.get("gain_source") or "none" for c in cal), len(cal)),
        "layout_found": _rate(layouts, n - st["rejected"]),
        "px_per_mm": ({"n": len(ppm), **{f"p{p}": round(float(np.percentile(ppm, p)), 1) for p in (10, 50, 90)}} if ppm else {}),
        "overall_quality_of_ok": ({"n": len(ok_q), "median": round(float(np.median(ok_q)), 3)} if ok_q else {}),
        "client": _rate(clients, n),
        "modality": _rate(modality, n),
        "doctor_decisions": _rate(decisions, len(feedback)),
        "doctor_changed_fields": dict(changed),
        "layout_labels": {"n": len(lay_fb), "scanner_right": lay_hit,
                          "share_right": round(lay_hit / len(lay_fb), 3) if lay_fb else None},
    }


def labels(feedback: list[dict]) -> list[dict]:
    """One line per field a doctor corrected: no hash, no id, no comment."""
    out = []
    for f in feedback:
        ch = f.get("changes") or {}
        before, after = ch.get("before") or {}, ch.get("after") or {}
        for k in FIELDS:
            if k in after:
                out.append({"field": k, "scanner": before.get(k), "doctor": after[k],
                            "scanner_status": before.get("status"), "scanner_warnings": sorted(before.get("warnings") or [])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--feedback", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--labels", type=Path)
    a = ap.parse_args()
    fb = _feedback(a.feedback)
    rep = report(_results(a.db), fb)
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if a.labels:
        a.labels.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels(fb)), encoding="utf-8")


if __name__ == "__main__":
    main()
