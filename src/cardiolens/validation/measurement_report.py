"""Measurement accuracy report modelled on IEC 60601-2-25 (D103; plan step 4).

The standard tests global durations against the CSE database: the mean difference from
the referee estimates and its standard deviation must stay inside per-parameter limits
(table 201.105). CSE and CTS are licensed and not available here, so this report uses
the open references instead and says so — it is an *approximation* of that test, not a
conformance claim:

* LUDB (200 records; P/QRS/T boundaries marked by cardiologists on every lead) for PR,
  QRS and QT;
* GE Marquette 12SL global measurements shipped with PTB-XL+ for HR, PR, QRS, QT, P
  duration and the frontal P / QRS axes.

Limits used (mean difference, SD) — from secondary sources on table 201.105; verify
against the standard text before any regulatory use: P duration ±10 / 15 ms, PR ±10 / 10,
QRS ±6 / 10, QT ±25 / 30. HR and axes have no such limit and are reported descriptively.

CLI::

    python -m cardiolens.validation.measurement_report --ludb data/ludb \\
        --ptbxl data/ptbxl --ptbxl-list ids.txt --twelve-sl 12sl_features.csv --out report.md
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from multiprocessing import Pool
from pathlib import Path

IEC_LIMITS = {"P duration": (10, 15), "PR": (10, 10), "QRS": (6, 10), "QT": (25, 30)}
_TWELVE_SL = {"HR": "HR_Ventr_Global", "PR": "PR_Int_Global", "QRS": "QRS_Dur_Global", "QT": "QT_Int_Global",
              "P duration": "P_Dur_Global", "QRS axis": "R_AxisFrontal_Global", "P axis": "P_AxisFront_Global"}


def _ours(m) -> dict[str, float | None]:
    iv = m.intervals
    return {"HR": m.heart_rate_bpm, "PR": iv.pr_ms, "QRS": iv.qrs_ms, "QT": iv.qt_ms,
            "P duration": m.p_wave_duration_ms, "QRS axis": m.axes.qrs_deg, "P axis": m.axes.p_deg}


def _run_wfdb(path: str) -> dict[str, float | None] | None:
    from cardiolens.layer1_ingestion.parsers import load_wfdb
    from cardiolens.layer6_interpret import StubProvider
    from cardiolens.pipeline import run_pipeline

    try:
        return _ours(run_pipeline(load_wfdb(path), mode="A", provider=StubProvider()).measurements)
    except Exception:
        return None


def _run_ludb(args: tuple[str, str]) -> tuple[dict, dict] | None:
    from cardiolens.validation.harness import ludb_intervals_from_annotations

    root, name = args
    ours = _run_wfdb(f"{root}/{name}")
    if ours is None:
        return None
    gold = ludb_intervals_from_annotations(Path(root), name, 500.0)
    return ours, {"PR": gold["pr"], "QRS": gold["qrs"], "QT": gold["qt"]}


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def summarise(pairs: list[tuple[dict, dict]], params: list[str]) -> list[dict]:
    rows = []
    for p in params:
        d = [(_angle_diff(o[p], g[p]) if "axis" in p else o[p] - g[p])
             for o, g in pairs if o.get(p) is not None and g.get(p) is not None]
        if len(d) < 3:
            continue
        row = {"param": p, "n": len(d), "bias": st.mean(d), "sd": st.stdev(d),
               "mae": st.mean(abs(x) for x in d)}
        lim = IEC_LIMITS.get(p)
        if lim:
            row["limit"] = lim
            row["passes"] = abs(row["bias"]) <= lim[0] and row["sd"] <= lim[1]
            row["within"] = sum(abs(x) <= lim[0] for x in d) / len(d)
        elif "axis" in p:
            row["within"] = sum(abs(x) <= 15 for x in d) / len(d)
        rows.append(row)
    return rows


def to_markdown(title: str, rows: list[dict]) -> str:
    out = [f"### {title}", "", "| параметр | n | смещение | SD | MAE | предел (смещение / SD) | в пределе | доля в допуске |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lim = r.get("limit")
        lim_s = f"±{lim[0]} / {lim[1]}" if lim else ("±15° (описательно)" if "axis" in r["param"] else "—")
        ok = ("да" if r["passes"] else "**нет**") if lim else "—"
        within = f"{r['within']:.0%}" if "within" in r else "—"
        out.append(f"| {r['param']} | {r['n']} | {r['bias']:+.1f} | {r['sd']:.1f} | {r['mae']:.1f} | {lim_s} | {ok} | {within} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ludb", type=Path)
    ap.add_argument("--ptbxl", type=Path)
    ap.add_argument("--ptbxl-list", type=Path)
    ap.add_argument("--twelve-sl", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    parts = []
    with Pool(a.workers) as pool:
        if a.ludb:
            names = sorted((p.stem for p in a.ludb.glob("*.hea")), key=lambda x: int(x) if x.isdigit() else x)
            res = [r for r in pool.map(_run_ludb, [(str(a.ludb), n) for n in names]) if r]
            parts.append(to_markdown(f"LUDB, разметка кардиологов ({len(res)} записей)",
                                     summarise(res, ["PR", "QRS", "QT"])))
        if a.ptbxl and a.ptbxl_list and a.twelve_sl:
            ref = {str(int(float(r["ecg_id"]))): r for r in csv.DictReader(a.twelve_sl.open())}
            items = [ln.split() for ln in a.ptbxl_list.read_text().splitlines() if ln.strip()]
            items = [(i, p) for i, p in items if i in ref]
            ours = pool.map(_run_wfdb, [str(a.ptbxl / p) for _, p in items])
            pairs = []
            for (i, _), o in zip(items, ours, strict=True):
                if o is None:
                    continue
                g = {k: (float(ref[i][c]) if ref[i].get(c) not in (None, "") else None) for k, c in _TWELVE_SL.items()}
                pairs.append((o, g))
            parts.append(to_markdown(f"PTB-XL против GE 12SL ({len(pairs)} записей)",
                                     summarise(pairs, list(_TWELVE_SL))))
    a.out.write_text("\n".join(parts))
    print(a.out.read_text())


if __name__ == "__main__":
    main()
