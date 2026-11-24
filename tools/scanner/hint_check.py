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
HINTS = ("closer", "glare", "steady", "level", "light", "bright", "shadow", "frame")
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
    g = sub.add_parser("report")
    g.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        k, n = (int(x) for x in a.shard.split("/"))
        run(a.root, a.out, (k, n))
    else:
        print(json.dumps(report(a.files), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
