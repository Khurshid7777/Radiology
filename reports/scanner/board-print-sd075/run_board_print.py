"""SD-075 run: a board printed with a feed error along x, a true-scale page on it.
Usage: python reports/scanner/board-print-sd075/run_board_print.py out.json [case ...]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "scanner"))
from conftest import synth_mv as _smv
from ecgscan.bench.metrics import clinical_measurements
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.board import load_boards, render_board
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, scan

mv = _smv.__wrapped__()
TILT = np.float32([[120, 80], [0.95, 30], [1.0, 0.98], [60, 0.9]])
spec = load_boards(["a3", "a4"])[0]

def feed(board, ppm, err):
    """Board printed with x_true = x + err(x) (mm)."""
    h, w = board.shape[:2]
    xb = np.arange(w * 2) / 2 / ppm
    xt = xb + err(xb, spec.size_mm[0])
    xs = np.arange(w) / ppm
    src = np.interp(xs, xt, xb) * ppm
    mapx = np.tile(src.astype(np.float32), (h, 1)); mapy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(board, mapx, mapy, cv2.INTER_LINEAR, borderValue=255)

def photo(ppm, err, bent=False):
    board = render_board(spec, ppm) if bent else feed(render_board(spec, ppm), ppm, err)
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    page, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=ppm))
    x0, y0 = int(50 * ppm), int(50 * ppm)
    board[y0:y0 + page.shape[0], x0:x0 + page.shape[1]] = page
    if bent:                       # the board and the page on it stretched alike: not a print error
        board = feed(board, ppm, err)
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [w * TILT[1, 0], TILT[1, 1]], [w * TILT[2, 0], h * TILT[2, 1]], [TILT[3, 0], h * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(w * 1.05), int(h * 1.02)), borderValue=(90, 80, 70))

CASES = {
    "ideal": lambda x, W: 0 * x,
    "scale_0.99": lambda x, W: -0.01 * x,
    "scale_1.01": lambda x, W: 0.01 * x,
    "sine_1.5": lambda x, W: 1.5 * np.sin(2 * np.pi * x / W),
    "owner": lambda x, W: 1.0 * (1 - x / W) + 2.0 * np.sin(np.pi * x / W),
    "0.99+sine": lambda x, W: -0.01 * x + 1.5 * np.sin(2 * np.pi * x / W),
    "1.01+owner": lambda x, W: 0.01 * x + 1.0 * (1 - x / W) + 2.0 * np.sin(np.pi * x / W),
    "owner_x2": lambda x, W: 2.0 * (1 - x / W) + 4.0 * np.sin(np.pi * x / W),
    "bent_x2": lambda x, W: 2.0 * (1 - x / W) + 4.0 * np.sin(np.pi * x / W),
}

def run(names):
    out = {}
    for name in names:
        dbg = ScanDebug()
        res = scan(cv2.imencode(".png", photo(12.0, CASES[name], bent=name.startswith("bent")))[1].tobytes(), debug=dbg)
        fit = dbg.board
        rs = []
        for lead, tr in res.leads.items():
            y = tr.array(); k = int(round(tr.start_s * 500)); t = mv[lead][k:k + len(y)]
            ok = np.isfinite(y[:len(t)])
            if ok.sum() > 100: rs.append(np.corrcoef(t[ok], y[:len(t)][ok])[0, 1])
        cm = clinical_measurements(res) if res.leads else {}
        r = dbg.rectified
        out[name] = dict(status=res.status, reject=res.reject_reasons, warnings=res.warnings,
                         method=r.method if r else None, board_ok=fit.ok if fit else None,
                         board_res=round(fit.residual_mm, 3) if fit else None, reason=fit.reason if fit else None,
                         grid_x=r and r.grid.px_per_mm_x and round(r.grid.px_per_mm_x, 4), grid_y=r and r.grid.px_per_mm_y and round(r.grid.px_per_mm_y, 4),
                         spread=r and r.tile_pitches and round(float(np.std(r.tile_pitches) / np.mean(r.tile_pitches)), 4), corners_only=fit and fit.corners_only, notes=fit and fit.notes,
                         r_med=round(float(np.median(rs)), 4) if rs else None,
                         hr=cm.get("heart_rate_bpm") and round(cm["heart_rate_bpm"], 2), qrs=cm.get("qrs_ms"), qt=cm.get("qt_ms"), pr=cm.get("pr_ms"))
        print(name, json.dumps(out[name], default=str), flush=True)
    return out

if __name__ == "__main__":
    o = run(sys.argv[2:] or list(CASES))
    with open(sys.argv[1], "w") as f:
        json.dump(o, f, indent=1, default=str)
