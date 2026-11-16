"""R8 (SD-111): aggregate the PM-ECG-ID bench rows into tables (no images, no patient data are read or written).

    python -I tools/scanner/pmecgid_report.py --root data/pmecgid --rows DIR [DIR ...] --out reports/scanner/r8
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

PMC_PCC, PMC_RMSE = 0.91, 0.10          # PMcardio's published claim (vendor, their 6000 images, their method)
HR_TOL = 5.0


def _load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        rows += [json.loads(x) for x in (d / "cases.jsonl").read_text().splitlines() if x.strip()]
    return rows


def _fmt_of(root: Path) -> dict[str, str]:
    meta = csv.DictReader((root / "final_data" / "metadata.csv").open())
    return {r["Image relative path"].rsplit(".", 1)[0]: r["ECG format"] for r in meta}


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    st = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] != "rejected" and r["traces"]]
    rs, ms = [], []                     # per lead, over read pages
    for r in ok:
        for t in r["traces"]:
            if t.get("r") is not None:
                rs.append(t["r"])
                ms.append(t["rmse_mv"])
    page_r = [r["r_median"] for r in ok if r.get("r_median") is not None]
    page_m = [r["rmse_median"] for r in ok if r.get("rmse_median") is not None]
    # "all images": a refusal / unmeasured lead counts as r = 0 (a page without a trace gives nothing)
    all_r = [r["r_median"] if (r["status"] != "rejected" and r.get("r_median") is not None) else 0.0 for r in rows]
    bad = [r for r in ok if r.get("bad")]
    bad_unflag = [r for r in bad if r["status"] == "ok"]
    ok_rows = [r for r in rows if r["status"] == "ok"]

    def err(key):
        v = [r["interval_err_ms"][key] for r in ok if r.get("interval_err_ms") and r["interval_err_ms"].get(key) is not None]
        return (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if v else (None, None, 0)

    hr = []
    for r in ok:
        ref = (r.get("reference") or {}).get("rr_ms")
        e = (r.get("interval_err_ms") or {}).get("rr_ms")
        if ref and e is not None:
            hr.append(abs(60000 / ref - 60000 / (ref + e)) <= HR_TOL or abs(60000 / ref - 60000 / max(ref - e, 1)) <= HR_TOL)
    return {
        "n": n, "ok": st["ok"], "needs_confirmation": st["needs_confirmation"], "rejected": st["rejected"],
        "lead_r_mean": float(np.mean(rs)) if rs else None, "lead_r_median": float(np.median(rs)) if rs else None,
        "lead_rmse_mean": float(np.mean(ms)) if ms else None, "lead_rmse_median": float(np.median(ms)) if ms else None,
        "page_r_median": float(np.median(page_r)) if page_r else None,
        "share_page_r_ge_091": float(np.mean([x >= PMC_PCC for x in page_r])) if page_r else None,
        "share_lead_r_ge_091": float(np.mean([x >= PMC_PCC for x in rs])) if rs else None,
        "all_images_r_mean": float(np.mean(all_r)) if all_r else None,
        "page_rmse_median": float(np.median(page_m)) if page_m else None,
        "hr_within_5bpm": float(np.mean(hr)) if hr else None,
        "rr": err("rr_ms"), "qrs": err("qrs_ms"), "qt": err("qt_ms"), "pr": err("pr_ms"),
        "bad_read": len(bad), "bad_unflagged": len(bad_unflag), "ok_n": len(ok_rows),
        "bad_unflagged_ids": [r["case_id"] for r in bad_unflag],
        "reject_reasons": Counter(x for r in rows if r["status"] == "rejected" for x in (r["reject_reasons"] or ["(none)"])),
        "median_s": float(np.median([r["elapsed_s"] for r in rows])) if rows else None,
    }


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _table(groups: dict[str, list[dict]]) -> list[str]:
    head = ("| группа | N | ok | подтв. | отказ | r на отв. (медиана по отв.) | r≥0.91 (стр.) | RMSE мВ (мед.) | "
            "r по всем (отказ=0) | ЧСС ±5 | QRS Δмс мед/p90 | QT Δмс мед/p90 | PR Δмс мед/p90 | плохие без пометки |")
    lines = [head, "|" + "---|" * 14]
    for g, rows in groups.items():
        s = _stats(rows)
        q = lambda k, s=s: f"{_f(s[k][0], 0)}/{_f(s[k][1], 0)}" 
        lines.append(f"| {g} | {s['n']} | {s['ok']} | {s['needs_confirmation']} | {s['rejected']} | {_f(s['page_r_median'])} "
                     f"| {_f(s['share_page_r_ge_091'], 2)} | {_f(s['page_rmse_median'])} | {_f(s['all_images_r_mean'])} "
                     f"| {_f(s['hr_within_5bpm'], 2)} | {q('qrs')} | {q('qt')} | {q('pr')} | {s['bad_unflagged']} из {s['ok_n']} ok |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load([Path(d) for d in a.rows])
    fm = _fmt_of(root)
    main_rows = [r for r in rows if r["slice"].startswith("real_")]
    aug_rows = [r for r in rows if r["slice"].startswith("aug_")]
    by = lambda rs, key: {k: [r for r in rs if key(r) == k] for k in sorted({key(r) for r in rs})} 
    folder = lambda r: r["slice"]                                                                     
    fmt = lambda r: fm.get(r["case_id"], "?")                                                         
    lines = ["# R8: сканер на PM-ECG-ID (SD-111)", "", "## Фото и сканы (одностраничные записи)", ""]
    lines += _table({"ВСЕ фото": main_rows, **by(main_rows, folder)})
    lines += ["", "## По формату листа (фото)", ""] + _table(by(main_rows, fmt))
    lines += ["", "## Искажённые (augmentation) — отдельно", ""] + _table({"ВСЕ augmented": aug_rows, **by(aug_rows, folder)})
    s = _stats(main_rows)
    lines += ["", "## Причины отказов (фото)", ""] + [f"- {k}: {v}" for k, v in s["reject_reasons"].most_common()]
    sa = _stats(aug_rows)
    lines += ["", "## Причины отказов (augmented)", ""] + [f"- {k}: {v}" for k, v in sa["reject_reasons"].most_common()]
    lines += ["", f"## Плохие без пометки (фото): {s['bad_unflagged']} из {s['ok_n']} «ok»", ""]
    lines += [f"- {c}" for c in s["bad_unflagged_ids"]]
    lines += ["", f"## Плохие без пометки (augmented): {sa['bad_unflagged']} из {sa['ok_n']} «ok»", ""]
    lines += [f"- {c}" for c in sa["bad_unflagged_ids"]]
    lines += ["", "## Для сравнения (заявление PMcardio, их метод и 6000 снимков): средний PCC > 0.91, RMSE < 0.10 мВ.",
              (f"Здесь: r по отведениям на прочитанных — среднее {_f(s['lead_r_mean'])}, медиана {_f(s['lead_r_median'])}; "
              f"RMSE среднее {_f(s['lead_rmse_mean'])}, медиана {_f(s['lead_rmse_median'])} мВ; "
              f"доля отведений с r≥0.91: {_f(s['share_lead_r_ge_091'], 2)}.")]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("case_id", "slice", "status", "reject_reasons", "r_median", "rmse_median",
                                                       "interval_err_ms", "bad", "pr_ms", "p_onset_ratio", "elapsed_s")}) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
