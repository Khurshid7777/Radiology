"""Internal consistency of a digitised paper ECG (D114, plan step 7).

A page's leads within one time window were recorded at the same moment, so two
checks need no reference signal:

* **panel lag** — every lead's QRS lands where the lead-II rhythm strip has it.
  Residual perspective shifts rows in time against each other; the QRS read off
  all leads then widens (70 -> 114 ms, a false VT) and the RR wanders (sinus read
  as AF). The lag is the cross-correlation peak of |slope| against lead II over
  the lead's window;
* **Einthoven** — II = I + III inside the window the three limb leads share.

Synthetic pages (8 records x 7 degradations x 2 seeds): clean, mild, blurred and
noisy photos have a median lag <= 7 ms and r(II, I + III) >= 0.85; phone, hard
and wrecked photos — where every false critical alert of the benchmark sat —
have 25-80 ms. A lag the search finds is undone (``align_panels``, D115): each
lead's window is shifted onto the strip, which recovers phone photos whose trace
shapes survived; what is left is judged by Einthoven after the shift.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation

LAG_SEARCH_S = 0.08


def _envelope(x: np.ndarray) -> np.ndarray:
    e = np.abs(np.diff(x, prepend=x[:1]))
    return e - e.mean()


def _lag_ms(x: np.ndarray, ref: np.ndarray, fs: float) -> float | None:
    a, b = _envelope(x), _envelope(ref)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    n = len(a)
    lim = min(int(LAG_SEARCH_S * fs), n // 4)
    cc = [float(np.dot(a[max(0, k):n + min(0, k)], b[max(0, -k):n - max(0, k)])) for k in range(-lim, lim + 1)]
    return 1000.0 * (int(np.argmax(cc)) - lim) / fs


def _window(sig: SignalRepresentation, lead: str, t0: float, t1: float) -> np.ndarray:
    fs = sig.sampling_rate_hz
    return sig.lead(lead)[int(round(t0 * fs)):int(round(t1 * fs))].astype(np.float64)


def page_consistency(sig: SignalRepresentation) -> dict | None:
    """``{"panel_lag_ms": median |lag|, "einthoven_r": r}`` for a digitised page with a
    full-length lead-II strip; None when the page gives nothing to compare."""
    windows = getattr(sig.provenance, "lead_windows", None) or {}
    if not sig.provenance.digitized or "II" not in windows or "II" not in sig.lead_names:
        return None
    ii0, ii1 = windows["II"]
    fs = sig.sampling_rate_hz
    lags = []
    for lead, (t0, t1) in windows.items():
        if lead == "II" or lead not in sig.lead_names or t0 < ii0 or t1 > ii1 or (t1 - t0) < 1.0:
            continue
        lag = _lag_ms(_window(sig, lead, t0, t1), _window(sig, "II", t0, t1), fs)
        if lag is not None:
            lags.append(abs(lag))
    ein = None
    if all(ln in windows and ln in sig.lead_names for ln in ("I", "III")):
        t0 = max(windows["I"][0], windows["III"][0], ii0)
        t1 = min(windows["I"][1], windows["III"][1], ii1)
        if t1 - t0 >= 1.0:
            a = _window(sig, "II", t0, t1)
            b = _window(sig, "I", t0, t1) + _window(sig, "III", t0, t1)
            if np.std(a) > 0 and np.std(b) > 0:
                ein = float(np.corrcoef(a, b)[0, 1])
    if not lags and ein is None:
        return None
    return {"panel_lag_ms": float(np.median(lags)) if lags else None,
            "einthoven_r": None if ein is None else round(ein, 3)}


def page_unreliable(raw: dict | None, aligned: dict | None, max_lag_ms: float,
                    min_einthoven_r: float) -> bool:
    """True when the lag before alignment reached the search limit (not found) or
    the aligned page breaks II = I + III."""
    if raw is None:
        return False
    lag = raw["panel_lag_ms"]
    ein = (aligned or raw)["einthoven_r"]
    return bool((lag is not None and lag >= max_lag_ms) or (ein is not None and ein < min_einthoven_r))


def align_panels(sig: SignalRepresentation, passes: int = 2) -> SignalRepresentation:
    """Shift every lead's window onto the lead-II strip by its QRS lag (D115)."""
    windows = getattr(sig.provenance, "lead_windows", None) or {}
    if not sig.provenance.digitized or "II" not in windows or "II" not in sig.lead_names:
        return sig
    fs = sig.sampling_rate_hz
    x = np.array(sig.signal, dtype=np.float32, copy=True)
    ii = sig.lead_names.index("II")
    ii0, ii1 = windows["II"]
    for _ in range(passes):
        for lead, (t0, t1) in windows.items():
            if lead == "II" or lead not in sig.lead_names or t0 < ii0 or t1 > ii1 or (t1 - t0) < 1.0:
                continue
            i = sig.lead_names.index(lead)
            a, b = int(round(t0 * fs)), int(round(t1 * fs))
            lag = _lag_ms(x[i, a:b].astype(np.float64), x[ii, a:b].astype(np.float64), fs)
            k = 0 if lag is None else int(round(lag * fs / 1000.0))
            if k == 0:
                continue
            seg = x[i, a:b]
            x[i, a:b] = (np.concatenate([seg[k:], np.repeat(seg[-1:], k)]) if k > 0
                         else np.concatenate([np.repeat(seg[:1], -k), seg[:k]]))
    return sig.model_copy(update={"signal": x})
