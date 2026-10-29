"""Photo -> pipeline -> review card -> verdict parsing. No Telegram here; everything
is testable offline and the agent only moves bytes and messages around.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import Provenance, SignalRepresentation
from cardiolens.feedback import Feedback, FindingFeedback
from cardiolens.layer1_ingestion.digitize import estimate_grid_pitch
from cardiolens.layer1_ingestion.strip import (
    LEADS_INCONSISTENT_NOTE,
    SCALE_CONFLICT_NOTE,
    digitize_auto_to_signal,
)
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
    when it is a photo of a recorder's own screen (D120-D121: its grid is not paper) or
    of a three-channel strip on a patterned desk (D125)."""
    import cv2

    from cardiolens.layer1_ingestion.screen import is_device_screen
    from cardiolens.layer1_ingestion.sequential import is_sequential_strip

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, None
    pitch = estimate_grid_pitch(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    ok = (pitch is not None and pitch >= min_pitch_px) or is_device_screen(bgr) or is_sequential_strip(bgr)
    return bool(ok), None if pitch is None else float(pitch)


# Scanner refusals about the picture itself (D139): the CardioLens readers would read the
# same picture worse, so the refusal stands. Refusals about the layout (a screen, a report
# sheet, part of a strip: GRID_NOT_FOUND, LAYOUT_UNKNOWN, PAGE_INCOMPLETE, SCREEN_PHOTO, ...)
# go on to the CardioLens readers, which read those formats behind their photo gates.
SCANNER_FINAL_REFUSALS = frozenset({"LOW_RESOLUTION", "GLARE", "BLUR", "TRACE_BROKEN", "LIMB_LEADS_INCONSISTENT",
                                    "LOW_CONTRAST", "IMAGE_TOO_LARGE", "CALIBRATION_MISSING"})


PHOTO_GATES = frozenset({"photo_unreliable", "no_ecg_signal", "amplitude_implausible"})
COARSE_WARNING = "LOW_RESOLUTION_COARSE"     # the scanner's coarse mode (D170, SD-074)

# A paper ECG photographed sideways (D171): the scanner says LAYOUT_UNKNOWN, but the
# CardioLens readers then "read" the tall picture as a 12x1 strip — time runs down the
# page, and the record is garbage. Turning it automatically was tried and rejected on the
# scanner side (SD-074: the pulse and the QRS polarity pick the upside-down page too), so
# a portrait photo that the readers took for paper is refused with the advice to turn it.
PORTRAIT_HINT = ("Снимок вертикальный: строки ЭКГ должны идти слева направо. Поверните фото и "
                 "загрузите снова")
_PAPER_NOTES = ("12x1 strip", "3x4 page", "three-channel strip", "3x4+rhythm strip", "report sheet")


def _portrait(path) -> bool:
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            # As the phone held it (EXIF): the scanner and OpenCV read it so (D180).
            w, h = ImageOps.exif_transpose(im).size
    except Exception:
        return False
    return h > 1.15 * w


def _is_device_screen(path) -> bool:
    """The photo shows a device screen by the CardioLens detector (a lit panel in a dark
    frame); a file OpenCV cannot open (HEIC, PDF) is not called one."""
    try:
        import cv2

        from cardiolens.layer1_ingestion.screen import is_device_screen

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:
        return False
    return img is not None and bool(is_device_screen(img))


def _guard_orientation(path, sig):
    """``sig`` as read by the CardioLens readers, unless the photo stands on end and they
    took it for a paper format — then the read is refused (D171)."""
    if _portrait(path) and any(n.startswith(_PAPER_NOTES) for n in sig.provenance.notes):
        raise ValueError(PORTRAIT_HINT)
    return sig


def scale_unknown(result) -> bool:
    """The scanner read the page but asks to confirm it, trusts neither its calibration nor
    its paper speed, and finds its own rhythm row inconsistent with the panels (D144). The
    scale doubt alone is common on good reads (13 of the scanner's benchmark pages: strips,
    photos without a header); with the rhythm inconsistency it marked only bad reads there.
    Limb-lead inconsistency alone is not enough: the synthetic generator's limb leads are
    only roughly consistent, and a clean page of them carries it. With broken traces as well
    it is (D181: ECG-Image-Kit pages read as atrial fibrillation with no gate)."""
    cal = getattr(result, "calibration", None)
    warnings = set(result.warnings or [])
    return (result.status == "needs_confirmation"
            and ((cal is not None and cal.confidence == 0) or "SPEED_UNCERTAIN" in warnings)
            and ("RHYTHM_INCONSISTENT" in warnings
                 or {"LIMB_LEADS_INCONSISTENT", "TRACE_GAPS"} <= warnings))


def attach_device_header(sig: SignalRepresentation, result) -> None:
    """The recorder's printed measurements and model from the scanner's header read (S3,
    D183): a reference the core compares ours with. Its text conclusion is not carried:
    it is the recorder's, never ours, and never goes to the narrative."""
    h = getattr(result, "header_ocr", None)
    if h is None:
        return
    dm = h.device_measurements
    if dm is not None:
        # Values no check of the scanner vouches for (QTc off Bazett, axes) are left out
        # (scanner SD-091): a misread one must not even make an advisory mismatch.
        d = dm.model_dump()
        unsure = set(d.pop("uncertain", None) or [])
        vals = {k: int(v) for k, v in d.items() if v is not None and k not in unsure}
        sig.provenance.device_measurements = vals or None
    info = {k: getattr(h, k) for k in ("device_model", "filter_hp_hz", "filter_lp_hz", "mains_hz")
            if getattr(h, k, None) is not None}
    sig.provenance.device_info = info or None


def leads_inconsistent(result) -> bool:
    """The scanner asks to confirm a read whose own checks disagree: its rhythm row against
    the panels, or the limb leads with broken traces. With the scale read from the page
    (header and pulse) this still marks bad reads: ECG-Image-Kit pages read rate 124 as 89
    and 48 as 61 with both (D181); pages with only limb-lead doubt read right."""
    warnings = set(result.warnings or [])
    return (result.status == "needs_confirmation"
            and ("RHYTHM_INCONSISTENT" in warnings or {"LIMB_LEADS_INCONSISTENT", "TRACE_GAPS"} <= warnings))


def scan_keeping_overlay(scan, image, hints=None):
    """``scan()`` with its confirmation overlay kept in memory (D163): the traces drawn over
    the rectified page, header and footer bands blanked (SD-050, D136), as JPEG bytes — or
    None when the scanner refused (it draws nothing then). The artifact files live only
    for the call."""
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_overlay_"))
    try:
        result = scan(image, artifacts_dir=tmp, hints=hints)
        p = (result.artifacts or {}).get("trace_overlay_preview")
        overlay = Path(p).read_bytes() if p and Path(p).exists() else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result, overlay


def digitize_photo(path: str | Path | list[str | Path], *, paper_speed_mm_s: float | None = None,
                   gain_mm_mv: float | None = None) -> SignalRepresentation:
    """The bot's photo reader (D139): the ECG scanner first — on its benchmark it reads paper
    pages at r 0.98 with no unflagged bad case, where the CardioLens readers have 23 % —
    then, when the scanner refuses the layout, the CardioLens readers (device screens,
    report sheets, parts of strips). Raises ValueError with the reason otherwise.
    A list: one page photographed in parts on the capture board (SD-066, D145), which only
    the scanner composes; a set it does not compose raises ValueError."""
    in_parts = isinstance(path, list)
    try:
        from ecgscan.contract.cardiolens_adapter import to_signal_representation
        from ecgscan.i18n import t as scanner_text
        from ecgscan.pipeline import scan
    except ImportError:                 # the scanner extra is not installed
        if in_parts:
            raise ValueError("photos in parts need the scanner") from None
        return apply_paper_scale(digitize_auto_to_signal(path), paper_speed_mm_s, gain_mm_mv)
    from ecgscan.pipeline import ScanHints

    # D179: the speed and gain printed on the strip, given by the user, go to the scanner as
    # its hints and are applied to the CardioLens readers' record (they assume 25 / 10).
    hints = ScanHints(paper_speed_mm_s=paper_speed_mm_s, gain_mm_mv=gain_mm_mv)
    result, overlay = scan_keeping_overlay(scan, [str(p) for p in path] if in_parts else path, hints)
    readers = None if in_parts else (lambda: apply_paper_scale(
        _guard_orientation(path, digitize_auto_to_signal(path)), paper_speed_mm_s, gain_mm_mv))
    if result.status != "rejected":
        sig = to_signal_representation(result)
        sig.provenance.overlay_jpeg = overlay      # what was read, for the doctor's eye (D163)
        attach_device_header(sig, result)
        # A compressed photo read in the scanner's coarse mode (D170) stays with the scanner:
        # the CardioLens readers read 3-4 px/mm worse, and the core withholds what the
        # scale cannot give (photo_coarse).
        if COARSE_WARNING in (result.warnings or []):
            # D172: a device screen (a dark lit panel, no paper) whose lattice the coarse
            # estimator took for a 5-mm grid: the screen readers read it whole, the coarse
            # read has the traces in pieces. The scanner's own screen test is its; this is
            # the route's belt and braces.
            if readers is not None and _is_device_screen(path):
                return readers()
            return sig
        unknown = scale_unknown(result)
        if not unknown and not leads_inconsistent(result):
            return sig
        # D144: read, but with no time or gain scale it trusts and its own rhythm row at
        # odds with the panels (on a strip turned by 2 deg: a rate of 55 read as 100.6). The
        # CardioLens readers go first; if they too read it only behind a photo gate, the
        # scanner's record stands with its scale marked unknown, so the rate and intervals
        # are withheld (D128).
        if readers is not None:
            try:
                alt = readers()
                if not run_pipeline(alt, mode="A").findings.codes() & PHOTO_GATES:
                    return alt
            except ValueError:
                pass
        sig.provenance.notes.append(SCALE_CONFLICT_NOTE if unknown else LEADS_INCONSISTENT_NOTE)
        return sig
    final = [r for r in result.reject_reasons if r in SCANNER_FINAL_REFUSALS]
    if final or readers is None:
        raise ValueError("; ".join(scanner_text(f"reject.{r}") for r in (final or result.reject_reasons)))
    return readers()


USER_SCALE_NOTE = "scale given by the user"


def apply_paper_scale(sig: SignalRepresentation, paper_speed_mm_s: float | None,
                      gain_mm_mv: float | None) -> SignalRepresentation:
    """The CardioLens readers read time at 25 mm/s and amplitude at 10 mm/mV (or at the
    calibration pulse they found). A strip printed at 50 mm/s read at 25 doubles every
    interval and halves the rate; at 20 mm/mV every amplitude doubles (D179: the owner's
    CardioTouch strip, 50 mm/s and 20 mm/mV). The user's values correct the record."""
    notes = list(sig.provenance.notes)
    upd: dict = {}
    k_t = (paper_speed_mm_s / 25.0) if paper_speed_mm_s and paper_speed_mm_s != 25 else None
    pulse = any("'pulse_found': True" in n or "(calibration_pulse)" in n for n in notes)
    k_a = (10.0 / gain_mm_mv) if gain_mm_mv and gain_mm_mv != 10 and not pulse else None
    if k_t:
        upd["sampling_rate_hz"] = sig.sampling_rate_hz * k_t
        upd["duration_s"] = sig.duration_s / k_t
        w = sig.provenance.lead_windows
        prov = sig.provenance.model_copy(update={"lead_windows": None if w is None else
                                                 {ld: (a / k_t, b / k_t) for ld, (a, b) in w.items()}})
        upd["provenance"] = prov
    if k_a:
        upd["signal"] = sig.signal * np.float32(k_a)
    if not (k_t or k_a):
        return sig
    out = sig.model_copy(update=upd)
    out.provenance.overlay_jpeg = sig.provenance.overlay_jpeg
    out.provenance.notes = [*notes, f"{USER_SCALE_NOTE}: {paper_speed_mm_s or 25:g} mm/s, "
                                    f"{gain_mm_mv or 10:g} mm/mV" + (" (gain from the pulse kept)" if pulse and gain_mm_mv else "")]
    return out


def on_capture_board(paths: list[str | Path]) -> bool:
    """Every photo shows the scanner's capture board (its ArUco markers, SD-065)."""
    try:
        import cv2

        from ecgscan.board import detect_board, load_boards
        from ecgscan.pipeline import pipeline_config
    except ImportError:
        return False
    b_cfg = pipeline_config().get("board") or {}
    if not b_cfg.get("enabled", True):
        return False
    specs = load_boards(list(b_cfg.get("specs", [])))
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None or detect_board(img, specs, min_markers=int(b_cfg.get("min_markers", 3)),
                                       whole_zone=False) is None:
            return False
    return True


# Readers' refusals the card says in words (D142); anything else is shown as raised.
_ERROR_TEXT = [
    ("cannot be named", ("отведения на этом куске ленты не опознать: на нём нет ни начала ленты с калибровочным "
                         "импульсом, ни групп I–III / aVR–aVF. Снимите ленту целиком или пришлите все куски "
                         "одним альбомом")),
    ("no trace detected", ("кривая ЭКГ на снимке не найдена. Снимите ближе, прямо сверху, без бликов, "
                           "и отправьте файлом")),
]


_PPM_NOTE = " На снимке ≈"


def error_text(e: Exception) -> str:
    """What the card says about a photo that was not read: the scanner's and the readers'
    Russian advice as is, known English refusals in words, the rest with its type."""
    from cardiolens.layer1_ingestion.sequential import RHYTHM_LOW_RES

    msg = str(e)
    if RHYTHM_LOW_RES in msg:
        head, _, advice = msg.partition("): ")
        ppm = head.rpartition("(")[2].split()[0]
        return f"{advice}{_PPM_NOTE}{ppm} точки на мм."
    for key, text in _ERROR_TEXT:
        if key in msg:
            return text[0].upper() + text[1:] + "."
    if isinstance(e, ValueError) and any("а" <= ch <= "я" for ch in msg.lower()):
        return msg                      # already said in Russian (the scanner's refusals)
    return f"{type(e).__name__}: {msg}"


def analyse_photo(path: str | Path, *, mode: str = "A", provider=None) -> tuple[PipelineResult | None, str | None]:
    """Digitise the photo (``digitize_photo``, D139) and run the pipeline; (result, None)
    or (None, error text)."""
    try:
        sig = digitize_photo(path)
        return run_pipeline(sig, mode=mode, provider=provider or StubProvider()), None
    except Exception as e:  # digitiser raises on pages it cannot read — that is data too
        return None, error_text(e)


def digitize_photos(paths: list[str | Path], *, paper_speed_mm_s: float | None = None,
                    gain_mm_mv: float | None = None) -> SignalRepresentation:
    """Photos sent together as one record — the bot's album, the web page's upload (D155).
    Photos of one recorder's pages — TeleECG's limb page and chest page — become one record
    (D123), as do photos of the parts of one three-channel strip (D125); one page in parts on
    the capture board is composed by the scanner (D145); anything else is read from the
    first photo that digitises. Raises ValueError with the reason in words (D142)."""
    scale = {"paper_speed_mm_s": paper_speed_mm_s, "gain_mm_mv": gain_mm_mv}
    if len(paths) == 1:
        try:
            return digitize_photo(paths[0], **scale)
        except Exception as e:  # digitiser raises on pages it cannot read — that is data too
            raise ValueError(error_text(e)) from e
    from cardiolens.layer1_ingestion.screen import combine_screen_signals

    # A set the scanner refuses is said as such; the parts are not read one by one (each is a piece).
    if on_capture_board(paths):
        try:
            return digitize_photo(list(paths), **scale)
        except Exception as e:
            raise ValueError(error_text(e)) from e
    sigs, errors = [], []
    for p in paths:
        try:
            sigs.append(digitize_photo(p, **scale))
        except Exception as e:
            errors.append((Path(p).name, error_text(e)))
    screens = [s for s in sigs if any(n.startswith("device screen photo") for n in s.provenance.notes)]
    strips = [s for s in sigs if any(n.startswith(("three-channel strip photo", "3x4+rhythm strip photo"))
                                     for n in s.provenance.notes)]
    try:
        if len(screens) >= 2:
            return combine_screen_signals(screens)
        if len(strips) >= 2:       # parts of one strip, in the order sent (D125)
            return combine_screen_signals(strips, drop_repeats=True)
    except ValueError:            # the pages repeat leads: two photos of the same page
        return sigs[0]
    if sigs:
        return sigs[0]
    # One reason for all the photos is said once (D142: three parts of one strip
    # repeated the same refusal three times).
    reasons = {t.split(_PPM_NOTE)[0] for _, t in errors}
    if len(reasons) == 1:
        ppm = sorted({t.split(_PPM_NOTE)[1].split()[0] for _, t in errors if _PPM_NOTE in t})
        raise ValueError(reasons.pop() + (f"{_PPM_NOTE}{'–'.join(ppm[::max(1, len(ppm) - 1)])} точки на мм."
                                          if ppm else ""))
    raise ValueError("; ".join(f"{name}: {t}" for name, t in errors))


def analyse_photos(paths: list[str | Path], *, mode: str = "A",
                   provider=None) -> tuple[PipelineResult | None, str | None]:
    """Several photos sent together (a Telegram album) -> (result, None) or (None, error text);
    ``digitize_photos`` says which photos make one record."""
    if len(paths) == 1:
        return analyse_photo(paths[0], mode=mode, provider=provider)
    try:
        sig = digitize_photos(paths)
    except ValueError as e:
        return None, str(e)
    try:
        return run_pipeline(sig, mode=mode, provider=provider or StubProvider()), None
    except Exception as e:
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
    axes[0].set_title(f"CardioVision — оцифровано с фото · находки: {codes}", fontsize=10)
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
    scanned = next((n.split()[1] for n in prov.notes if n.startswith("layout: ")), None)   # ecgscan (D139)
    # The reader's own note names the layout; the calibration dict and the album note mention
    # "strip" and "page" too (D148: an FK album's card read "раскладка calibration").
    layout = (f"{scanned} (сканер)" if scanned else
              next((n.split(":")[0] for n in prov.notes
                    if not n.startswith(("calibration:", "photo pages combined"))
                    and any(w in n for w in ("strip", "page", "screen", "sheet"))), None))
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
    unreliable = next((f for f in result.findings.findings if f.code == "photo_unreliable"), None)
    scale_unknown = unreliable is not None and any(e.feature == "scale_conflict" and e.value == "yes"
                                                    for e in unreliable.evidence)
    if scale_unknown:      # D128: the time scale is not known, so no number read from it is
        lines.append("ЧСС и интервалы не измерены: масштаб снимка не определён")
    else:
        lines.append(f"ЧСС {_fmt(s['heart_rate_bpm'])} /мин · PR {_fmt(s['pr_ms'])} · QRS {_fmt(s['qrs_ms'])} · "
                     f"QTc {_fmt(s['qtc_bazett_ms'])} мс · ось {_fmt(s['qrs_axis_deg'], '°')}"
                     + (" (по ненадёжному снимку)" if unreliable is not None else ""))
    scanner = next((n for n in result.signal.provenance.notes if n.startswith("ecgscan scan_id=")), None)
    if scanner and "status=needs_confirmation" in scanner:   # D139: the scanner read it but asks to check
        warn = next((n for n in result.signal.provenance.notes if n.startswith("warnings: ")), "")
        lines.append("⚠️ Сканер просит проверить масштаб и раскладку по снимку" + (f" ({warn[10:]})" if warn else ""))
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
