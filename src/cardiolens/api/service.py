"""Request handling helpers for the API — build a signal, run the pipeline.

Kept separate from the FastAPI wiring so it is unit-testable without HTTP.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from cardiolens.layer1_ingestion import (
    generate_synthetic_ecg,
    is_dicom,
    is_scp,
    load_csv,
    load_dicom,
    load_edf,
    load_scp,
    load_wfdb,
)
from cardiolens.pipeline import PipelineResult, run_pipeline

# Guardrails for the public-facing service.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024   # 64 MB total across uploaded files

# A photo or scan of a paper ECG (D155): told by the file's magic, else by its name. iPhone
# photos (HEIC/HEIF) and PDF scans or device exports count too (D158): OpenCV reads neither,
# so ``as_photo`` renders them to PNG first.
_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"BM", b"II*\x00", b"MM\x00*")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
                   ".heic", ".heif", ".pdf"}
_HEIF_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heif", b"mif1", b"msf1")


def _is_heif(content: bytes) -> bool:
    return content[4:8] == b"ftyp" and content[8:12] in _HEIF_BRANDS


def _is_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF")


def is_image(content: bytes, name: str) -> bool:
    head = content[:12]
    return (head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
            or _is_heif(content) or _is_pdf(content)
            or Path(name).suffix.lower() in _IMAGE_SUFFIXES)


# The scanner's and the readers' refusals were written for Telegram ("send the photo as a
# file"); on the web page the same advice is the original from the gallery (D159).
_WEB_WORDING = (
    ("отправьте фото файлом, без сжатия", "выберите оригинал снимка из галереи, а не копию из мессенджера"),
    ("и отправьте файлом", "и выберите оригинал из галереи"),
    ("пришлите все куски одним альбомом", "выберите все куски вместе"),
)


def web_wording(text: str) -> str:
    for bot, web in _WEB_WORDING:
        text = text.replace(bot, web)
    return text


def _upright_copy(path: Path, content: bytes) -> Path | None:
    """A JPEG the phone stored on its side with an EXIF turn: a PNG turned upright, so every
    reader (the scanner, the CardioLens readers, the portrait check) sees the same picture
    (D180). None when there is nothing to turn."""
    if not content.startswith(b"\xff\xd8"):
        return None
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            if (im.getexif() or {}).get(0x0112, 1) in (None, 1):
                return None
            out = path.with_name(path.stem + "_upright.png")
            ImageOps.exif_transpose(im).convert("RGB").save(out)
            return out
    except Exception:
        return None


def as_photo(path: Path, content: bytes) -> Path:
    """The file as an image OpenCV reads: itself, or a PNG rendered next to it (D158). A PDF
    gives its first page at 300 dpi (11.8 px/mm, above the readers' 8 px/mm); a HEIC photo
    is turned the way the phone held it (EXIF)."""
    if _is_heif(content):
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        out = path.with_suffix(".png")
        ImageOps.exif_transpose(Image.open(path)).convert("RGB").save(out)
        return out
    turned = _upright_copy(path, content)
    if turned is not None:
        return turned
    if _is_pdf(content):
        import pymupdf

        out = path.with_suffix(".png")
        with pymupdf.open(path) as doc:
            if doc.page_count == 0:
                raise ValueError("в PDF нет страниц")
            doc[0].get_pixmap(dpi=300).save(out)
        return out
    return path
STORE_MAX_ENTRIES = 200               # bounded in-memory analysis cache

# Synthetic demo cases exposed by the web UI (mirrors cardiolens.demo).
DEMO_CASES: dict[str, dict] = {
    "normal": dict(heart_rate_bpm=72),
    "brady": dict(heart_rate_bpm=44),
    "tachy": dict(heart_rate_bpm=125),
    "hyperk": dict(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6),
    "stemi": dict(heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]),
    "child": dict(heart_rate_bpm=95, age_years=6),
}


@dataclass
class UploadFile:
    filename: str
    content: bytes


def signal_from_case(case: str):
    if case not in DEMO_CASES:
        raise ValueError(f"unknown case '{case}'")
    return generate_synthetic_ecg(**DEMO_CASES[case])


def signal_from_uploads(
    files: list[UploadFile],
    *,
    sampling_rate_hz: float | None = None,
    unit: str = "uv",
    paper_speed_mm_s: float | None = None,
    gain_mm_mv: float | None = None,
    wording: str = "web",
):
    """Persist uploaded files to a temp dir and dispatch to a parser.

    Supports WFDB (.hea + .dat), EDF (.edf/.bdf), CSV (.csv/.txt) and DICOM waveform
    ECG (.dcm, or any file with the DICOM magic: devices export without an extension) and
    SCP-ECG (.scp, or a record whose CRC and section-0 mark hold), and photos or scans of a
    paper ECG (JPEG/PNG/WebP/BMP/TIFF, D155).
    """
    if not files:
        raise ValueError("no files uploaded")
    total = sum(len(f.content) for f in files)
    if total > MAX_UPLOAD_BYTES:
        raise ValueError(f"upload too large ({total} bytes > {MAX_UPLOAD_BYTES})")
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_api_"))
    try:
        saved: list[Path] = []
        for f in files:
            # Path(...).name strips any directory components (traversal-safe).
            p = tmp / Path(f.filename).name
            p.write_bytes(f.content)
            saved.append(p)

        # Photos of a paper ECG take the bot's route (D155): the scanner first, then the
        # CardioLens readers; several photos are one record (pages, parts of a strip, parts of
        # a page on the capture board); a refusal comes back in words (D142) as a 400.
        images = [(p, f.content) for p, f in zip(saved, files, strict=True) if is_image(f.content, p.name)]
        if images:
            if len(images) != len(saved):
                raise ValueError("photos and ECG signal files cannot be mixed in one upload")
            from cardiolens.telegram.analyze import digitize_photos

            try:
                return digitize_photos([as_photo(p, c) for p, c in images],
                                       paper_speed_mm_s=paper_speed_mm_s, gain_mm_mv=gain_mm_mv)
            except ValueError as e:
                # The Telegram bot keeps "send it as a file" (D161); the page says "the gallery".
                raise ValueError(str(e) if wording == "bot" else web_wording(str(e))) from e

        # A malformed file is the sender's error (400 with the reason), whatever the
        # parser raised underneath: truncated SCP sections, junk EDF, a 0-d CSV (D135).
        try:
            dcm = next((p for p, f in zip(saved, files, strict=True)
                        if p.suffix.lower() == ".dcm" or is_dicom(f.content)), None)
            if dcm is not None:
                return load_dicom(dcm)
            scp = next((p for p, f in zip(saved, files, strict=True)
                        if p.suffix.lower() == ".scp" or is_scp(f.content)), None)
            if scp is not None:
                return load_scp(scp)
            hea = next((p for p in saved if p.suffix.lower() == ".hea"), None)
            if hea is not None:
                return load_wfdb(hea.with_suffix(""))
            edf = next((p for p in saved if p.suffix.lower() in (".edf", ".bdf")), None)
            if edf is not None:
                return load_edf(edf)
            csv = next((p for p in saved if p.suffix.lower() in (".csv", ".txt")), None)
            if csv is not None:
                if sampling_rate_hz is None:
                    raise ValueError("sampling_rate_hz is required for CSV input")
                return load_csv(csv, sampling_rate_hz=sampling_rate_hz, unit=unit)
            raise ValueError("unsupported upload; expected WFDB(.hea+.dat), EDF, CSV, DICOM ECG or SCP-ECG")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read the uploaded ECG file ({type(exc).__name__})") from exc
    finally:
        # Parsers read the raw signal into memory, so the temp files are no
        # longer needed once parsing returns (or raises) — never leak them.
        shutil.rmtree(tmp, ignore_errors=True)


def apply_patient_overrides(
    sig,
    *,
    age_years: float | None = None,
    age_days: float | None = None,
    sex: str | None = None,
    potassium: float | None = None,
    calcium: float | None = None,
):
    if age_years is not None:
        sig.patient.age_years = age_years
    if age_days is not None:
        sig.patient.age_days = age_days
    if sex is not None:
        sig.patient.sex = sex
    if potassium is not None:
        sig.context.potassium = potassium
    if calcium is not None:
        sig.context.calcium = calcium
    return sig


# --- Bounded in-memory analysis store (MVP; not for multi-worker production) ---
# LRU-evicting so a long-running server can't grow without bound. Swap for a DB
# / object store in production (see DECISIONS.md).
_STORE: OrderedDict[str, PipelineResult] = OrderedDict()
_OWNERS: dict[str, str | None] = {}      # analysis id -> API key name that created it (D136)


def run_and_store(sig, *, mode: str | None, prior_id: str | None = None,
                  owner: str | None = None) -> tuple[str, PipelineResult]:
    prior = get_analysis(prior_id, owner=owner) if prior_id else None
    if prior_id:
        sig.context.prior_ecg_id = prior_id
    result = run_pipeline(sig, mode=mode, prior=prior)
    analysis_id = uuid.uuid4().hex[:12]
    _STORE[analysis_id] = result
    _OWNERS[analysis_id] = owner
    _STORE.move_to_end(analysis_id)
    while len(_STORE) > STORE_MAX_ENTRIES:
        old_id, _ = _STORE.popitem(last=False)  # evict least-recently-used
        _OWNERS.pop(old_id, None)
    return analysis_id, result


def get_analysis(analysis_id: str, *, owner: str | None = None) -> PipelineResult | None:
    """``owner``: the caller's key name; another key's analysis reads as absent (D136)."""
    if owner is not None and _OWNERS.get(analysis_id) not in (None, owner):
        return None
    result = _STORE.get(analysis_id)
    if result is not None:
        _STORE.move_to_end(analysis_id)  # mark as recently used
    return result
