"""S4, TZ 18.10/18.14: a report PDF from a scanned page, checked by three PDF engines.

    python reports/scanner/s4/make_report.py <out_dir>

A synthetic 3x4+rhythm page under a Russian recorder header (no real patient) is scanned, sent
through the CardioLens core the way the service does it (``to_signal_representation`` + the
overlay preview) and written by the core's ``write_pdf``. ``ecgscan.bench.pdf_check`` then renders
it with PDFium, Poppler and MuPDF and looks for every scanner flag, in Russian and Uzbek, in the
text each engine extracts. Writes ``report.pdf``, ``check.json`` and the rendered pages.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.outputs.pdf_report import OVERLAY_NOTE_RU, OVERLAY_NOTE_UZ, write_pdf
from cardiolens.pipeline import run_pipeline
from ecgscan.bench.headers import render_header
from ecgscan.bench.pdf_check import check
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.bench.sources import derive_limb_leads
from ecgscan.contract.cardiolens_adapter import to_signal_representation
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanHints, scan
from ecgscan.report_block import scanner_rows


def page_photo() -> bytes:
    s = generate_synthetic_ecg(heart_rate_bpm=72, seed=7)
    mv = derive_limb_leads({n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names})
    page, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    h = render_header(5, style="ru", px_per_mm=10.0)
    top = np.full((h.image.shape[0], page.shape[1], 3), 255, np.uint8)
    top[:, :min(page.shape[1], h.image.shape[1])] = h.image[:, :page.shape[1]]
    return cv2.imencode(".png", np.vstack([top, page]))[1].tobytes()


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    art = out / "artifacts"
    res = scan(page_photo(), hints=ScanHints(client="api"), artifacts_dir=art)
    sig = to_signal_representation(res)
    sig.provenance.overlay_jpeg = (art / "overlay_preview.jpg").read_bytes()
    result = run_pipeline(sig)
    pdf = out / "report.pdf"
    write_pdf(result, str(pdf))
    flags = {lang: scanner_rows(res, lang) for lang in ("ru", "uz")}
    expect = {"overlay": [OVERLAY_NOTE_RU[:60], OVERLAY_NOTE_UZ[:60]], "conclusion": ["Заключение (RU)", "Xulosa (UZ)"]}
    for lang, rows in flags.items():
        expect[f"flags_{lang}"] = [s for row in rows for s in row]
    report = check(pdf, [s for v in expect.values() for s in v], out / "pages")
    missing = {e for eng in report["engines"].values() for e in eng.get("missing", [])}
    report["groups"] = {k: {"expected": len(v), "missing": sorted(set(v) & missing)} for k, v in expect.items()}
    report["scan"] = {"status": res.status, "warnings": res.warnings, "flags": flags}
    (out / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "groups", "fonts")}, ensure_ascii=False, indent=1))
    for name, eng in report["engines"].items():
        print(name, {k: v for k, v in eng.items() if k != "missing"}, "missing", len(eng.get("missing", [])))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
