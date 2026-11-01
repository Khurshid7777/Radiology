"""Real-use quality log (D190): one line per analysis, derived numbers only.

What the site and the bot actually receive is measured here, not on a benchmark: the route
a photo took, why it was gated or refused, our rate and intervals, and — when the recorder
printed them in its header (D183) — the recorder's own values, a free reference.

No image, no file name, no patient field, no conclusion text and no user goes into a line.
Writing never breaks an analysis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

GATE_CODES = ("photo_unreliable", "photo_coarse", "no_ecg_signal", "amplitude_implausible",
              "poor_quality_signal", "screen_photo_limited")
# A refusal is kept as a kind, never as its text (a message may quote an upload's name).
REFUSAL_KINDS = (("разрешение", "low_resolution"), ("resolution", "low_resolution"),
                 ("вертикальн", "portrait"), ("кусок", "strip_piece"), ("too large", "too_large"),
                 ("не найден", "no_ecg"), ("not an ecg", "no_ecg"), ("формат", "format"))


def refusal_kind(message: str) -> str:
    low = (message or "").lower()
    return next((kind for key, kind in REFUSAL_KINDS if key in low), "other")


def quality_row(result, client: str | None = None) -> dict:
    """The derived-only line for one finished analysis."""
    from cardiolens.pipeline import device_mismatches

    sig, m = result.signal, result.measurements
    prov = sig.provenance
    notes = prov.notes or []
    scanner = next((n for n in notes if n.startswith("ecgscan scan_id=")), None)
    warn = next((n[10:] for n in notes if n.startswith("warnings: ")), "")
    codes = sorted(result.findings.codes())
    dev = {k: v for k, v in (prov.device_measurements or {}).items() if isinstance(v, (int, float))}
    return {
        "ts": int(time.time()),
        "client": client or "web",
        "source": sig.source,
        "digitized": bool(prov.digitized),
        "route": "scanner" if scanner else ("readers" if prov.digitized else "file"),
        "scanner_status": scanner.split("status=")[1].split()[0] if scanner and "status=" in scanner else None,
        "scanner_warnings": sorted(set(warn.replace(",", " ").split())),
        "px_per_mm": round(sig.sampling_rate_hz / 25.0, 1) if prov.digitized else None,
        "leads": len(sig.lead_names),
        "duration_s": round(sig.duration_s, 1),
        "gates": [c for c in codes if c in GATE_CODES],
        "hr_bpm": m.heart_rate_bpm, "pr_ms": m.intervals.pr_ms,
        "qrs_ms": m.intervals.qrs_ms, "qt_ms": m.intervals.qt_ms,
        "device": dev or None,
        "device_mismatch": [k for k, _, _ in device_mismatches(m, dev)] if dev else None,
        "findings": [c for c in codes if c not in GATE_CODES],
    }


def refusal_row(message: str, client: str | None = None) -> dict:
    return {"ts": int(time.time()), "client": client or "web", "refused": refusal_kind(message)}


def append(path: str | Path | None, row: dict) -> None:
    """Append one line; any failure is logged and swallowed."""
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:          # the log is a by-product; the analysis has already succeeded
        log.exception("quality log not written")


DEVICE_KEYS = ("hr_bpm", "pr_ms", "qrs_ms", "qt_ms")


def read(path: str | Path | None) -> list[dict]:
    p = Path(path) if path else None
    if not p or not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize(rows: list[dict], since_ts: int = 0) -> dict:
    """Counts per route (read / gated / refused) and agreement with the recorder's printed
    values where it printed them — the real-use counterpart of the ecgdig table (D185)."""
    rows = [r for r in rows if r.get("ts", 0) >= since_ts]
    out: dict = {"analyses": len(rows), "refused": {}, "routes": {}, "gates": {}, "device": {}}
    for r in rows:
        if "refused" in r:
            out["refused"][r["refused"]] = out["refused"].get(r["refused"], 0) + 1
            continue
        d = out["routes"].setdefault(r.get("route", "?"), {"read": 0, "gated": 0})
        d["read"] += 1
        d["gated"] += bool(r.get("gates") and set(r["gates"]) & {"photo_unreliable", "photo_coarse", "no_ecg_signal"})
        for g in r.get("gates") or []:
            out["gates"][g] = out["gates"].get(g, 0) + 1
        dev = r.get("device") or {}
        for k in DEVICE_KEYS:
            if dev.get(k) is None or r.get(k) is None:
                continue
            s = out["device"].setdefault(k, {"n": 0, "agree": 0, "gated": 0, "abs_err": []})
            s["n"] += 1
            s["agree"] += k not in (r.get("device_mismatch") or [])
            s["gated"] += bool(r.get("gates"))
            s["abs_err"].append(abs(r[k] - dev[k]))
    for s in out["device"].values():
        errs = sorted(s.pop("abs_err"))
        s["median_abs_err"] = errs[len(errs) // 2] if errs else None
    return out
