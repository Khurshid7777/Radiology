"""SD-140: why does the frame check sometimes find twice the real px/mm? Aggregate diagnostics only.

For each phone photo (PM-ECG-ID, a test set) the autocorrelation of the grid profile at fractions and multiples of the found
period P is taken. Photos are put in groups by the scanner's own scale (doubled: check / scan >= 1.7; normal: 0.85..1.2) read from the
SD-139 run, and only per-group quantiles are reported. Nothing here changes the frame check or tunes it.

    python -I tools/scanner/period_diag.py run --root data/pmecgid --hc <hint_check shard files...> --out <file.jsonl> [--shard k/n]
    python -I tools/scanner/period_diag.py report <file.jsonl> [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHONES = ("photos_iphone", "photos_samsung", "photos_doogee")
CHECK_WIDTH = 1280
# Lags (in units of the found period P) whose autocorrelation is compared with the one at P.
FRACTIONS = {"p_over_5": 0.2, "p_over_2": 0.5, "p_x1_5": 1.5, "p_x2": 2.0, "p_x3": 3.0}


def group_of(ratio: float | None) -> str | None:
    """'doubled' / 'normal' / 'half' / None by check px/mm over the scanner's."""
    if ratio is None:
        return None
    if ratio >= 1.7:
        return "doubled"
    if ratio < 0.6:
        return "half"
    return "normal" if 0.85 <= ratio <= 1.2 else None


def acorr(profiles: np.ndarray, top: int) -> np.ndarray:
    n = profiles.shape[1]
    return np.array([float((profiles[:, : n - lag] * profiles[:, lag:]).sum()) / n for lag in range(top + 2)])


def features(a: np.ndarray, period: float) -> dict:
    """Autocorrelation at fractions and multiples of the period over the one at the period (the best of 3 lags near each)."""
    def at(lag: float) -> float | None:
        m = int(round(lag))
        return None if m < 1 or m + 2 > len(a) else float(a[max(m - 1, 0): m + 2].max())
    base = at(period)
    if not base or base <= 0:
        return {}
    out = {}
    for name, f in FRACTIONS.items():
        v = at(period * f)
        out[name] = None if v is None else round(v / base, 4)
    out["p_over_a0"] = round(base / a[0], 4)
    return out


def quantiles(values: list[float]) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "p10": round(v[len(v) // 10], 3), "median": round(v[len(v) // 2], 3), "p90": round(v[9 * len(v) // 10], 3)}


def report(rows: list[dict]) -> dict:
    out: dict = {}
    for g in ("doubled", "normal", "half"):
        part = [r for r in rows if r["group"] == g]
        out[g] = {"n": len(part), **{k: quantiles([r[k] for r in part if r.get(k) is not None])
                                      for k in (*FRACTIONS, "p_over_a0", "sheet_ratio", "check_fill", "scan_fill")}}
    return out


def run(root: Path, hc_files: list[Path], out_path: Path, shard: tuple[int, int]) -> None:
    import cv2

    import ecgscan.capture_qc as Q

    seen: dict = {}
    orig = Q.grid_period

    def wrap(profiles, **kw):
        p = orig(profiles, **kw)
        seen["profiles"], seen["period"] = profiles, p
        return p
    Q.grid_period = wrap
    # the SD-139 rows are in file order inside each of its 3 shards (file i went to shard i % 3)
    hc = [[json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in hc_files]
    files = sorted(p for ph in PHONES for p in (root / "final_data" / "visual_data" / ph).glob("*.*"))
    with out_path.open("w") as out:
        for i, p in enumerate(files):
            if i % shard[1] != shard[0]:
                continue
            row = hc[i % len(hc)][i // len(hc)]
            ratio = row["ppm_check"] / row["ppm_scan"] if row.get("ppm_check") and row.get("ppm_scan") else None
            g = group_of(ratio)
            if g is None:
                continue
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            h, w = bgr.shape[:2]
            k = min(1.0, CHECK_WIDTH / w)
            small = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
            seen.clear()
            qc = Q.check_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale_to_capture=w / small.shape[1])
            if not seen.get("period") or qc.corners is None:
                continue
            fill = max(w, h) / 297.0              # px/mm if an A4 sheet filled the frame's long side
            extra = {"check_fill": round(row["ppm_check"] / fill, 3), "scan_fill": round(row["ppm_scan"] / fill, 3),
                     "status": row["status"]}
            a = acorr(seen["profiles"], int(0.4 * seen["profiles"].shape[1]))
            f = features(a, seen["period"])
            tl, tr, _, bl = qc.corners
            long_px = max(np.hypot(*np.subtract(tr, tl)), np.hypot(*np.subtract(bl, tl)))
            # the sheet's long side against the grid: A4 is 297 mm; a ratio near 2 means the period reads as 10 mm
            f["sheet_ratio"] = round(float(qc.px_per_mm / (long_px / 297.0)), 3)
            out.write(json.dumps({"group": g, "phone": p.parent.name, **f, **extra}) + "\n")
            out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, default=Path("data/pmecgid"))
    r.add_argument("--hc", nargs="+", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--shard", default="0/1")
    g = sub.add_parser("report")
    g.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        k, n = (int(x) for x in a.shard.split("/"))
        run(a.root, a.hc, a.out, (k, n))
    else:
        rows = [json.loads(line) for p in a.files for line in p.read_text().splitlines() if line.strip()]
        print(json.dumps(report(rows), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
