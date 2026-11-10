"""D227: signs of ventricular origin on a wide QRS template (Brugada, Vereckei aVR, Pava lead II), from the
owner's books (D216). Each sign is computed on the median beat; which of them the report uses is decided by
DECISIONS D227 (specificity against bundle-branch-block aberrancy).
"""

from __future__ import annotations

import numpy as np

PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
AMP_MIN_UV = 100.0          # a deflection smaller than this is not counted as an R or an S


def _lead(X: np.ndarray, names: list[str], name: str) -> np.ndarray | None:
    up = [n.upper() for n in names]
    return X[up.index(name.upper())] if name.upper() in up else None


def vt_signs(X: np.ndarray, names: list[str], on: int, off: int, fs: float) -> dict[str, bool | None]:
    """{sign: True / False / None (not measurable)} for one median beat; ``on`` / ``off`` are QRS bounds."""
    b0 = max(0, on - int(0.03 * fs))
    base = {n: float(np.median(X[i, b0:on])) if on > b0 else float(X[i, on]) for i, n in enumerate(names)}

    def q(name):
        x = _lead(X, names, name)
        return None if x is None else x[on:off + 1] - base[names[[m.upper() for m in names].index(name.upper())]]

    out: dict[str, bool | None] = {}
    width_ms = (off - on) / fs * 1000.0
    out["qrs_ge_160"] = width_ms >= 160.0
    pre = {n: q(n) for n in PRECORDIAL}
    if all(v is not None and v.size for v in pre.values()):
        net = [float(v.max() + v.min()) for v in pre.values()]          # dominant polarity
        pos = [float(v.max()) for v in pre.values()]
        neg = [float(-v.min()) for v in pre.values()]
        # Brugada: all precordial QRS monophasic R (no S of a quarter of the R) or all QS (no such R).
        mono_r = [p > 0 and g < 0.25 * p for p, g in zip(pos, neg, strict=True)]
        mono_qs = [g > 0 and p < 0.25 * g for p, g in zip(pos, neg, strict=True)]
        out["concordance"] = (all(mono_r) or all(mono_qs)) and all(abs(n) >= AMP_MIN_UV for n in net)
        rs = [p >= AMP_MIN_UV and g >= AMP_MIN_UV and int(np.argmax(v)) < int(np.argmin(v))
              for v, p, g in zip(pre.values(), pos, neg, strict=True)]
        out["no_rs_precordial"] = not any(rs)
        rs_ms = [int(np.argmin(v)) / fs * 1000.0 for v, ok in zip(pre.values(), rs, strict=True) if ok]
        out["rs_gt_100"] = bool(rs_ms) and max(rs_ms) > 100.0
    else:
        out["concordance"] = out["no_rs_precordial"] = out["rs_gt_100"] = None
    avr = q("aVR")
    if avr is not None and avr.size:
        k = int(0.04 * fs)
        first = avr[:k]
        init_r = float(first.max()) >= AMP_MIN_UV and int(np.argmax(first)) <= int(np.argmin(first) if first.min() < -AMP_MIN_UV else k)
        out["avr_initial_r"] = bool(init_r)       # Vereckei: an INITIAL R in aVR (a terminal R is RBBB)
    else:
        out["avr_initial_r"] = None
    ii = q("II")
    if ii is not None and ii.size:
        dom = ii if abs(ii.max()) >= abs(ii.min()) else -ii
        # time to the first local peak of the dominant deflection that reaches half its height
        half = 0.5 * float(dom.max())
        idx = next((i for i in range(1, dom.size - 1) if dom[i] >= half and dom[i] >= dom[i - 1] and dom[i] >= dom[i + 1]),
                   int(np.argmax(dom)))
        out["ii_rwpt_ge_50"] = idx / fs * 1000.0 >= 50.0
    else:
        out["ii_rwpt_ge_50"] = None
    i_lead, avf = q("I"), q("aVF")
    if i_lead is not None and avf is not None:
        out["northwest_axis"] = float(i_lead.max() + i_lead.min()) < 0 and float(avf.max() + avf.min()) < 0
    else:
        out["northwest_axis"] = None
    return out


SIGN_TEXT_RU = {
    "concordance": "конкордантность V1–V6", "no_rs_precordial": "нет RS в V1–V6", "rs_gt_100": "RS > 100 мс",
    "avr_initial_r": "начальный R в aVR", "ii_rwpt_ge_50": "время до пика R в II ≥ 50 мс",
    "northwest_axis": "ось «северо-запад»", "qrs_ge_160": "QRS ≥ 160 мс",
}
