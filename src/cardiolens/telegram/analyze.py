"""Photo -> pipeline -> review card -> verdict parsing. No Telegram here; everything
is testable offline and the agent only moves bytes and messages around.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import Provenance, SignalRepresentation
from cardiolens.feedback import Feedback, FindingFeedback
from cardiolens.layer1_ingestion.digitize import estimate_grid_pitch
from cardiolens.layer1_ingestion.strip import digitize_auto_to_signal
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.pipeline import PipelineResult, run_pipeline
from cardiolens.telegram.store import Record

MAX_SIDE_PX = 2400   # photos are downscaled to this before anything else (phones give 4000+)


def save_photo(data: bytes, dest: str | Path) -> Path:
    """Re-encode the photo (drops EXIF / GPS / device metadata) and cap its size."""
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    h, w = img.shape[:2]
    scale = MAX_SIDE_PX / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dest


def is_ecg_photo(path: str | Path, min_pitch_px: float = 3.0) -> tuple[bool, float | None]:
    """An image is an ECG candidate when the digitiser finds a millimetre grid in it, or
    when it is a photo of a recorder's own screen (D120-D121: its grid is not paper)."""
    import cv2

    from cardiolens.layer1_ingestion.screen import is_device_screen

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, None
    pitch = estimate_grid_pitch(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    ok = (pitch is not None and pitch >= min_pitch_px) or is_device_screen(bgr)
    return bool(ok), None if pitch is None else float(pitch)


def analyse_photo(path: str | Path, *, mode: str = "A", provider=None) -> tuple[PipelineResult | None, str | None]:
    """Digitise the photo (3x4 page or 12x1 strip, chosen from the ink — D72) and
    run the pipeline; (result, None) or (None, error text)."""
    try:
        sig = digitize_auto_to_signal(path)
        return run_pipeline(sig, mode=mode, provider=provider or StubProvider()), None
    except Exception as e:  # digitiser raises on pages it cannot read — that is data too
        return None, f"{type(e).__name__}: {e}"


def analyse_photos(paths: list[str | Path], *, mode: str = "A",
                   provider=None) -> tuple[PipelineResult | None, str | None]:
    """Several photos sent together (a Telegram album). Photos of one recorder's pages —
    TeleECG's limb page and chest page — become one record (D123); anything else is
    analysed from the first photo that digitises."""
    if len(paths) == 1:
        return analyse_photo(paths[0], mode=mode, provider=provider)
    from cardiolens.layer1_ingestion.screen import combine_screen_signals

    sigs, errors = [], []
    for p in paths:
        try:
            sigs.append(digitize_auto_to_signal(p))
        except Exception as e:  # noqa: BLE001 — a page that does not digitise is data too
            errors.append(f"{Path(p).name}: {type(e).__name__}: {e}")
    screens = [s for s in sigs if any(n.startswith("device screen photo") for n in s.provenance.notes)]
    try:
        if len(screens) >= 2:
            sig = combine_screen_signals(screens)
        elif sigs:
            sig = sigs[0]
        else:
            return None, "; ".join(errors)
    except ValueError:            # the pages repeat leads: two photos of the same page
        sig = sigs[0]
    try:
        return run_pipeline(sig, mode=mode, provider=provider or StubProvider()), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def save_signal(sig: SignalRepresentation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, signal=sig.signal.astype(np.float32), fs=float(sig.sampling_rate_hz),
        leads=np.array(sig.lead_names), duration_s=float(sig.duration_s),
        windows=json.dumps(sig.provenance.lead_windows or {}),
        confidence=-1.0 if sig.provenance.confidence is None else float(sig.provenance.confidence),
        notes=json.dumps(sig.provenance.notes),
    )
    return path


def load_signal(path: str | Path) -> SignalRepresentation:
    z = np.load(path, allow_pickle=False)
    conf = float(z["confidence"])
    windows = {k: tuple(v) for k, v in json.loads(str(z["windows"])).items()}
    return SignalRepresentation(
        signal=z["signal"].astype(np.float64), sampling_rate_hz=float(z["fs"]),
        lead_names=[str(x) for x in z["leads"]], duration_s=float(z["duration_s"]),
        source="telegram-photo",
        provenance=Provenance(digitized=True, confidence=None if conf < 0 else conf,
                              lead_windows=windows or None, notes=json.loads(str(z["notes"]))),
    )


def render_review_png(result: PipelineResult, path: str | Path, *, seconds: float = 5.0) -> Path:
    """Four leads on a red millimetre grid — what the digitiser read, for the reviewer's eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = result.signal
    fs = sig.sampling_rate_hz
    leads = [ln for ln in ("I", "II", "V1", "V5") if ln in sig.lead_names] or sig.lead_names[:4]
    n = int(min(seconds * fs, sig.n_samples))
    t = np.arange(n) / fs
    fig, axes = plt.subplots(len(leads), 1, figsize=(10, 1.9 * len(leads)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ln in zip(axes, leads, strict=False):
        y = sig.signal[sig.lead_names.index(ln), :n] / 1000.0
        ax.plot(t, y, color="#1a1f26", linewidth=0.9)
        ax.set_ylabel(f"{ln}\n(mV)", rotation=0, labelpad=18, va="center", fontsize=9)
        ax.set_xticks(np.arange(0, seconds + 0.01, 0.2), minor=True)
        ax.set_xticks(np.arange(0, seconds + 0.01, 1.0))
        ax.set_yticks(np.arange(-2, 2.01, 0.5))
        ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
        ax.grid(which="major", color="#e57373", linewidth=0.6)
        ax.grid(which="minor", color="#ffcdd2", linewidth=0.3)
        ax.set_ylim(-1.5, 2.0)
        ax.set_xlim(0, seconds)
    codes = ", ".join(sorted(result.findings.codes())) or "—"
    axes[0].set_title(f"CardioLens — оцифровано с фото · находки: {codes}", fontsize=10)
    axes[-1].set_xlabel("с (25 мм/с, 10 мм/мВ)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def summarize(result: PipelineResult) -> dict:
    m = result.measurements
    iv = m.intervals
    prov = result.signal.provenance
    calib = next((n for n in prov.notes if n.startswith("calibration:")), None)
    layout = next((n.split(":")[0] for n in prov.notes if "strip" in n or "page" in n), None)
    return {
        "layout": layout,
        "heart_rate_bpm": m.heart_rate_bpm, "pr_ms": iv.pr_ms, "qrs_ms": iv.qrs_ms,
        "qt_ms": iv.qt_ms, "qtc_bazett_ms": iv.qtc_bazett_ms, "qrs_axis_deg": m.axes.qrs_deg,
        "confidence": result.report.confidence_value, "digitiser_coverage": prov.confidence,
        "calibration": calib,
    }


def _fmt(v, unit: str = "", nd: int = 0) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}{unit}"


VERDICT_HELP = (
    "Ответьте на эту карточку:\n"
    "  +  — верно (все находки подтверждены)\n"
    "  -  — неверно (ни одна находка не подтверждена)\n"
    "  -код +код — убрать / добавить находки, например: -st_elevation +lbbb\n"
    "  любой другой текст — комментарий (без метки)\n"
    "/stats — счётчики, /help — коды находок"
)


def format_card(rec: Record, result: PipelineResult | None, error: str | None = None) -> str:
    head = f"📷 ЭКГ из «{rec.chat_title or rec.chat_id}» · {rec.date[:16]} · id {rec.id}"
    if rec.caption:
        head += f"\n💬 {rec.caption[:300]}"
    if result is None:
        return (f"{head}\n\n⚠️ Не удалось оцифровать: {error}\n"
                f"Фото сохранено в набор. Если на нём есть диагноз, добавьте его метками (+код).\n\n{VERDICT_HELP}")
    s = summarize(result)
    alerts = [a.label_ru for a in result.safety.alerts]
    findings = [f.label_ru for f in result.findings.findings] or ["— (ничего не найдено)"]
    lines = [head, ""]
    if alerts:
        lines.append("🚨 КРИТИЧЕСКОЕ: " + "; ".join(alerts))
    lines.append(f"ЧСС {_fmt(s['heart_rate_bpm'])} /мин · PR {_fmt(s['pr_ms'])} · QRS {_fmt(s['qrs_ms'])} · "
                 f"QTc {_fmt(s['qtc_bazett_ms'])} мс · ось {_fmt(s['qrs_axis_deg'], '°')}")
    lines.append("Находки: " + "; ".join(findings))
    lines.append("Коды: " + (", ".join(sorted(result.findings.codes())) or "—"))
    lines.append(f"Уверенность {s['confidence']:.2f} · оцифровка {_fmt(s['digitiser_coverage'], '', 2)}"
                 f"{' · раскладка ' + s['layout'] if s.get('layout') else ''} · нарратив: {result.report.provider}")
    lines.append("")
    lines.append(VERDICT_HELP)
    return "\n".join(lines)


_CONFIRM = {"+", "✅", "👍", "ok", "ок", "да", "верно", "yes"}
_REJECT = {"-", "−", "—", "❌", "👎", "нет", "неверно", "no"}
_TOKEN = re.compile(r"([+\-−])\s*([a-z][a-z0-9_]*)")


@dataclass
class VerdictParse:
    kind: str                         # confirmed | rejected | partial | comment
    gold: list[str] | None            # None for comment-only replies
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    comment: str | None = None


def parse_verdict(text: str, predicted: list[str]) -> VerdictParse:
    """Reviewer reply -> label. ``+`` confirms all predicted codes, ``-`` rejects all,
    ``-code`` / ``+code`` edit the set, anything else is a comment without a label."""
    raw = (text or "").strip()
    low = raw.lower()
    if low in _CONFIRM:
        return VerdictParse("confirmed", sorted(predicted))
    if low in _REJECT:
        return VerdictParse("rejected", [])
    tokens = _TOKEN.findall(low)
    if tokens:
        added = sorted({c for s, c in tokens if s == "+"})
        removed = sorted({c for s, c in tokens if s != "+"})
        gold = sorted((set(predicted) - set(removed)) | set(added))
        comment = _TOKEN.sub("", raw).strip(" ,;.") or None
        kind = "confirmed" if gold == sorted(predicted) else "partial"
        return VerdictParse(kind, gold, added, removed, comment)
    return VerdictParse("comment", None, comment=raw or None)


def apply_verdict(rec: Record, parse: VerdictParse) -> Record:
    rec.verdict = parse.kind
    rec.gold = parse.gold
    rec.comment = parse.comment
    rec.verdict_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def feedback_from_verdict(rec: Record, parse: VerdictParse, reviewer: str | None = None) -> Feedback | None:
    """Verdict -> TZ §13 feedback entry (confidence calibration samples)."""
    if parse.gold is None:
        return None
    gold = set(parse.gold)
    items = [
        FindingFeedback(code=c, verdict="confirmed" if c in gold else "rejected",
                        confidence=float(rec.confidences.get(c, 0.5)))
        for c in rec.predicted
    ]
    items += [FindingFeedback(code=c, verdict="corrected", confidence=0.0, corrected_code=c,
                              comment="missed by the pipeline")
              for c in gold - set(rec.predicted)]
    return Feedback(analysis_id=rec.id, reviewer=reviewer, mode=rec.analysis_mode, findings=items,
                    overall_agreement=(parse.kind == "confirmed"))
