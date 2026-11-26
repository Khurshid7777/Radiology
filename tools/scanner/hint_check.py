"""SD-139: how often the frame check's hints disagree with what the scanner does with the same photo.

Only aggregates are reported (per hint, per phone): no per-image values are kept in git, and no threshold is
changed here. The photos are a test set (PM-ECG-ID, data/pmecgid, gitignored).

    python -I tools/scanner/hint_check.py run --root data/pmecgid --out <file.jsonl> [--shard k/n]
    python -I tools/scanner/hint_check.py report <file.jsonl> [<file.jsonl> ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PHONES = ("photos_iphone", "photos_samsung", "photos_doogee")
HINTS = ("closer", "glare", "steady", "level", "light", "bright", "shadow", "frame", "grid")   # grid: SD-157
CHECK_WIDTH = 1280                      # as the Mini App page's server side (miniapp.check_width)


def outcome(status: str) -> str:
    """'reads' / 'confirm' / 'refused' by the scanner's status."""
    return {"ok": "reads", "needs_confirmation": "confirm"}.get(status, "refused")


def rates(rows: list[dict]) -> dict:
    """Per hint: share of photos it is named on, of those the share the scanner reads (false hint);
    for 'any hint': the same; and the share of photos with no hint that the scanner refuses (missed)."""
    n = len(rows)
    out: dict = {"n": n}
    for hint in (*HINTS, "any"):
        named = [r for r in rows if (r["hints"] if hint == "any" else hint in r["hints"])]
        false = [r for r in named if outcome(r["status"]) == "reads"]
        out[hint] = {"named": len(named), "named_share": round(len(named) / n, 4) if n else None,
                     "false": len(false), "false_share": round(len(false) / len(named), 4) if named else None}
    clear = [r for r in rows if not r["hints"]]
    missed = [r for r in clear if outcome(r["status"]) == "refused"]
    out["no_hint"] = {"n": len(clear), "refused": len(missed),
                      "refused_share": round(len(missed) / len(clear), 4) if clear else None}
    out["reads"] = sum(1 for r in rows if outcome(r["status"]) == "reads")
    return out


def run(root: Path, out_path: Path, shard: tuple[int, int]) -> None:
    import cv2

    from ecgscan.capture_qc import check_frame
    from ecgscan.pipeline import ScanHints, scan

    files = sorted(p for ph in PHONES for p in (root / "final_data" / "visual_data" / ph).glob("*.*"))
    with out_path.open("w") as out:
        for i, p in enumerate(files):
            if i % shard[1] != shard[0]:
                continue
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            k = min(1.0, CHECK_WIDTH / w)
            small = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
            qc = check_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale_to_capture=w / small.shape[1])
            res = scan(str(p), hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
            ppm = res.calibration.px_per_mm_x
            out.write(json.dumps({"phone": p.parent.name, "hints": qc.hints, "ok": qc.ok, "status": res.status,
                                  "reasons": list(res.reject_reasons or []), "ppm_check": qc.px_per_mm,
                                  "ppm_scan": ppm}) + "\n")
            out.flush()


def check_hints(bgr, width: int, min_strength: float | None) -> tuple[list[str], float | None, float | None, float]:
    """The upload check's hints at a copy ``width`` px wide; SD-157: with ``min_strength``, a weak grid adds 'grid'.
    Returns (hints, px/mm, period strength, seconds)."""
    import time

    import cv2

    from ecgscan.capture_qc import check_frame

    t = time.perf_counter()
    h, w = bgr.shape[:2]
    k = min(1.0, width / w)
    small = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
    qc = check_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale_to_capture=w / small.shape[1])
    hints = list(qc.hints)
    if min_strength is not None and qc.px_per_mm is not None and (qc.period_strength or 0.0) < min_strength:
        hints.append("grid")
    return hints, qc.px_per_mm, qc.period_strength, time.perf_counter() - t


def recheck(root: Path, old: Path, out_path: Path, shard: tuple[int, int], width: int, min_strength: float | None) -> None:
    """SD-157: the check again (another copy width, the weak-grid rule) on the photos of an earlier run's shard; the
    scanner's statuses are taken from that run (the scanner is not run again). Rows are matched by order and phone."""
    import cv2

    rows = [json.loads(line) for line in old.read_text().splitlines() if line.strip()]
    files = sorted(p for ph in PHONES for p in (root / "final_data" / "visual_data" / ph).glob("*.*"))
    mine = [p for i, p in enumerate(files) if i % shard[1] == shard[0]]
    with out_path.open("w") as out:
        it = iter(rows)
        for p in mine:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            row = next(it)
            if row["phone"] != p.parent.name:
                raise SystemExit("the earlier run's rows do not match the photos")
            hints, ppm, strength, sec = check_hints(bgr, width, min_strength)
            _, _, _, sec_old = check_hints(bgr, CHECK_WIDTH, None)
            out.write(json.dumps({**row, "hints": hints, "ok": not hints, "ppm_check": ppm, "strength": strength,
                                  "sec": round(sec, 3), "sec_old": round(sec_old, 3)}) + "\n")
            out.flush()


def doubled_silent(rows: list[dict]) -> int:
    """SD-157: photos where the check's px/mm is at least 1.7 times the scanner's and no hint is named."""
    return sum(1 for r in rows if not r["hints"] and r.get("ppm_check") and r.get("ppm_scan")
               and r["ppm_check"] / r["ppm_scan"] >= 1.7)


def report(paths: list[Path]) -> dict:
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    by: dict = defaultdict(list)
    for r in rows:
        by[r["phone"]].append(r)
    res = {"all": rates(rows), **{ph: rates(v) for ph, v in sorted(by.items())}}
    pairs = [(r["ppm_check"], r["ppm_scan"]) for r in rows if r["ppm_check"] and r["ppm_scan"]]
    if pairs:
        ratio = sorted(c / s for c, s in pairs)
        res["ppm_check_over_scan"] = {"n": len(pairs), "p10": round(ratio[len(ratio) // 10], 3),
                                      "median": round(ratio[len(ratio) // 2], 3), "p90": round(ratio[9 * len(ratio) // 10], 3)}
    res["doubled_silent"] = doubled_silent(rows)
    secs = sorted(r["sec"] for r in rows if "sec" in r)
    if secs:
        old = sorted(r["sec_old"] for r in rows)
        res["seconds"] = {"median": round(secs[len(secs) // 2], 3), "median_before": round(old[len(old) // 2], 3)}
    res["refusal_reasons"] = dict(sorted(defaultdict(int, {}).items()))
    reasons: dict = defaultdict(int)
    for r in rows:
        for c in r["reasons"]:
            reasons[c] += 1
    res["refusal_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, default=Path("data/pmecgid"))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--shard", default="0/1")
    c = sub.add_parser("recheck", help="SD-157: the check again on an earlier run's photos, statuses kept")
    c.add_argument("--root", type=Path, default=Path("data/pmecgid"))
    c.add_argument("--old", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--shard", default="0/1")
    c.add_argument("--width", type=int, default=1600)
    c.add_argument("--min-strength", type=float, default=0.15)
    g = sub.add_parser("report")
    g.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        k, n = (int(x) for x in a.shard.split("/"))
        run(a.root, a.out, (k, n))
    elif a.cmd == "recheck":
        k, n = (int(x) for x in a.shard.split("/"))
        recheck(a.root, a.old, a.out, (k, n), a.width, a.min_strength)
    else:
        print(json.dumps(report(a.files), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
