"""SD-077 run: paper lying over the rulers, beyond the zone, up to the markers.
Usage: python reports/scanner/board-beyond-sd077/run_board_beyond.py out.json [case ...]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "scanner"))
from conftest import synth_mv as _smv
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.board import load_board, render_board
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, scan

mv = _smv.__wrapped__()
TILT = np.float32([[120, 80], [0.95, 30], [1.0, 0.98], [60, 0.9]])


def shoot(board):
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [w * TILT[1, 0], TILT[1, 1]], [w * TILT[2, 0], h * TILT[2, 1]], [TILT[3, 0], h * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(w * 1.05), int(h * 1.02)), borderValue=(90, 80, 70))


def photo(case, ppm=12.0):
    spec = load_board(case["board"])
    board = cv2.cvtColor(render_board(spec, ppm), cv2.COLOR_GRAY2BGR)
    if case.get("layout"):
        pg, _ = render_page(mv, 500.0, get_layout(case["layout"]), RenderSpec(px_per_mm=ppm, **case.get("render", {})))
        t, b = case.get("crop_tb", (0, 0))
        pg = pg[int(t * ppm):pg.shape[0] - int(b * ppm)]
        x0, y0 = case["at"]
        X, Y = int(x0 * ppm), int(y0 * ppm)
        board[Y:Y + pg.shape[0], X:X + pg.shape[1]] = pg
    return shoot(board)


PAGE = dict(layout="6x2", render=dict(row_pitch_mm=33), crop_tb=(2, 16))      # 274 x 216 mm, rows to the edge
CASES = {
    "inside (test_board page)": dict(board="a3", layout="3x4+rhythm", at=(50, 50)),
    "6x2 inside the zone": dict(board="a3", **PAGE, at=(55, 40.5)),
    "6x2 over the bottom ruler": dict(board="a3", **PAGE, at=(55, 49.5)),
    "6x2 over the bottom ruler, grey grid": dict(board="a3", **{**PAGE, "render": dict(row_pitch_mm=33, grid="gray")}, at=(55, 49.5)),
    "bare board": dict(board="a3"),
}


def run(names):
    out = {}
    for name in names:
        dbg = ScanDebug()
        res = scan(cv2.imencode(".png", photo(CASES[name]))[1].tobytes(), debug=dbg)
        rs = {}
        for lead, tr in res.leads.items():
            y = tr.array(); k = int(round(tr.start_s * 500)); t = mv[lead][k:k + len(y)]
            n = min(len(t), len(y)); ok = np.isfinite(y[:n])
            if ok.sum() > 100:
                rs[lead] = round(float(np.corrcoef(t[:n][ok], y[:n][ok])[0, 1]), 3)
        r = dbg.rectified
        out[name] = dict(status=res.status, reject=res.reject_reasons, warnings=res.warnings,
                         method=r.method if r else None, zone=list(r.bgr.shape[:2]) if r else None,
                         layout=res.layout.name if res.layout else None, leads=len(res.leads),
                         r_med=round(float(np.median(list(rs.values()))), 3) if rs else None, r=rs)
        print(name, json.dumps(out[name]), flush=True)
    return out


if __name__ == "__main__":
    o = run(sys.argv[2:] or list(CASES))
    with open(sys.argv[1], "w") as f:
        json.dump(o, f, indent=1)
