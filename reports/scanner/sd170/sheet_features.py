import json, sys, os
import numpy as np
os.environ.setdefault("SM", "2.5")
src = open(sys.argv[1] + "/sd170/proto_0.py").read().split("SP = Path(sys.argv[1])")[0]
src = src.replace("FIXED_SLOPE = False", "FIXED_SLOPE = True")
exec(src)
from pathlib import Path
from ecgscan.bench.cases import load_case
from ecgscan.pipeline import ScanDebug, scan
SP = Path(sys.argv[1])
for sl in sys.argv[2:]:
    for cdir in sorted((SP / "lq_std_out/cases" / sl).iterdir()):
        case = load_case(cdir); dbg = ScanDebug(); res = scan(str(case.image_path), debug=dbg)
        if res.status == "rejected" or not dbg.readings: continue
        rect = dbg.rectified; ppm = rect.px_per_mm
        xs0 = min(int(q.panel.x0) for q in dbg.readings); xs1 = max(int(q.panel.x0) + len(q.y_px) for q in dbg.readings)
        rows = sorted({round(q.panel.baseline) for q in dbg.readings})
        fields, found = [], []
        for b in rows:
            dx, frac = grid_dx2(rect.bgr, b - 8 * ppm, b + 8 * ppm, xs0, xs1, ppm)
            fields.append(dx / ppm); found.append(frac)
        F = np.array(fields)
        p90 = float(np.percentile(np.abs(F), 90))
        cors = [float(np.corrcoef(F[i], F[i + 1])[0, 1]) for i in range(len(F) - 1) if F[i].std() > 1e-6 and F[i + 1].std() > 1e-6]
        # smoothness: energy of the field's second difference over ~4 mm against its variance
        step = int(4 * ppm)
        rough = float(np.median([np.std(f[2 * step:] - 2 * f[step:-step] + f[:-2 * step]) / (np.std(f) + 1e-9) for f in F]))
        print(json.dumps({"slice": sl, "case": cdir.name, "p90_mm": round(p90, 3), "row_corr_median": round(float(np.median(cors)), 3) if cors else None,
                          "rough": round(rough, 3), "found": round(float(np.median(found)), 3)}), flush=True)
