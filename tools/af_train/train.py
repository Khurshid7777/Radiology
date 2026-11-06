"""D205: train the AF second-path model on Chapman/Ningbo rows, check it on the PTB-XL bench rows.

    python tools/af_train/train.py --chapman <dir> --ptbxl <dir> [--out models/cardiolens/af_second_path.json]

Logistic regression on standardised features (stored as plain JSON). A gradient-boosting model is
fitted alongside only as a reference for how much a non-linear model would add. Five-fold CV on
Chapman; PTB-XL is never used for fitting or for choosing the threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cardiolens.ml.af import AF_FEATURES, vector

RULE_AF = "atrial_fibrillation"


def load(d: Path, positive: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    rows = [r for r in rows if "f" in r]
    x = np.vstack([vector(r["f"]) for r in rows])
    y = np.array([positive in r["labels"] for r in rows], dtype=int)
    return x, y, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapman", type=Path, required=True)
    ap.add_argument("--ptbxl", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--target-spec", type=float, default=0.97,
                    help="threshold: specificity on Chapman non-AF (CV) at least this")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    xc, yc, _ = load(a.chapman, "AF")
    xp, yp, rows_p = load(a.ptbxl, "AFIB")
    sc = StandardScaler().fit(xc)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p_cv = cross_val_predict(lr, sc.transform(xc), yc, cv=cv, method="predict_proba")[:, 1]
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    p_cv_h = cross_val_predict(hgb, xc, yc, cv=cv, method="predict_proba")[:, 1]
    neg = np.sort(p_cv[yc == 0])
    thr = float(neg[int(np.ceil(a.target_spec * len(neg))) - 1])
    lr.fit(sc.transform(xc), yc)
    hgb.fit(xc, yc)
    pp = lr.predict_proba(sc.transform(xp))[:, 1]
    pp_h = hgb.predict_proba(xp)[:, 1]
    thr_h = float(np.sort(p_cv_h[yc == 0])[int(np.ceil(a.target_spec * (yc == 0).sum())) - 1])

    def report(p, t, name):
        rule = np.array([RULE_AF in r["codes"] for r in rows_p])
        combo = rule | (p >= t)
        groups = {"AFIB": yp == 1}
        for g in ("PAC", "SARRH", "PVC", "AFLT", "STACH", "SBRAD", "SVTAC", "PSVT"):
            groups[g] = np.array([g in r["labels"] and "AFIB" not in r["labels"] for r in rows_p])
        groups["controls"] = np.array([{"NORM", "SR"} <= set(r["labels"]) and len(r["labels"]) == 2 for r in rows_p])
        out = {"model": name, "threshold": round(t, 3)}
        for g, mask in groups.items():
            out[g] = {"rule": f"{int(rule[mask].sum())}/{int(mask.sum())}",
                      "rule_or_model": f"{int(combo[mask].sum())}/{int(mask.sum())}",
                      "model_only_new": int((combo & ~rule)[mask].sum())}
        sr_on_af = sum(1 for r, c in zip(rows_p, combo, strict=True)
                       if "AFIB" in r["labels"] and "sinus_rhythm" in r["codes"] and not c)
        out["sinus_rhythm_left_on_AF"] = sr_on_af
        return out

    cv_auc = {}
    from sklearn.metrics import roc_auc_score
    cv_auc["lr"] = round(float(roc_auc_score(yc, p_cv)), 3)
    cv_auc["hgb"] = round(float(roc_auc_score(yc, p_cv_h)), 3)
    res = {"chapman_rows": int(len(yc)), "chapman_af": int(yc.sum()), "cv_auc": cv_auc,
           "ptbxl_auc": {"lr": round(float(roc_auc_score(yp, pp)), 3), "hgb": round(float(roc_auc_score(yp, pp_h)), 3)},
           "lr": report(pp, thr, "logistic"), "hgb_reference": report(pp_h, thr_h, "hgb")}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        a.out.write_text(json.dumps({
            "name": "af_second_path", "decision": "D205", "features": list(AF_FEATURES),
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]), "threshold": thr,
            "trained_on": "Chapman/Ningbo adults (tools/af_train/dump.py chapman)", "checked_on": "PTB-XL bench",
            "cv_auc": cv_auc, "target_spec_chapman": a.target_spec}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
