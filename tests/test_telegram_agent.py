"""Telegram collector (D71): the offline parts — photo gate, analysis, cards, verdicts,
dataset store, feedback entries, export to the harness — and the collector with a fake client."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardiolens.feedback import FeedbackStore
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import render_ecg_page
from cardiolens.telegram import (
    DatasetStore,
    Record,
    TelegramConfig,
    analyse_photo,
    apply_verdict,
    feedback_from_verdict,
    format_card,
    is_ecg_photo,
    load_signal,
    parse_verdict,
    render_review_png,
    save_photo,
    save_signal,
)
from cardiolens.telegram.export import agreement_summary, build_telegram_dataset, summary_lines
from cardiolens.validation import run_validation


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """A 3x4 page rendered from a synthetic sinus ECG at 10 px/mm, as a photo sent as a file
    arrives (D139: the scanner reads it; below 8 px/mm it refuses)."""
    root = tmp_path_factory.mktemp("page")
    sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=5)
    png = root / "page.png"
    render_ecg_page(sig, png, px_per_mm=10, calibration_pulse=True)
    return png


def test_photo_gate_finds_the_grid_and_rejects_a_plain_image(page, tmp_path):
    import cv2

    ok, pitch = is_ecg_photo(page)
    assert ok and 9.0 <= pitch <= 11.0      # the fixture page is drawn at 10 px/mm
    plain = tmp_path / "plain.png"
    cv2.imwrite(str(plain), np.full((400, 600), 230, dtype=np.uint8))
    assert is_ecg_photo(plain)[0] is False


def test_save_photo_reencodes_and_caps_size(page, tmp_path):
    import cv2

    big = cv2.resize(cv2.imread(str(page)), (4800, 3200))
    _ok, buf = cv2.imencode(".jpg", big)
    out = save_photo(buf.tobytes(), tmp_path / "p" / "x.jpg")
    h, w = cv2.imread(str(out)).shape[:2]
    assert max(h, w) == 2400 and out.suffix == ".jpg"
    with pytest.raises(ValueError):
        save_photo(b"not an image", tmp_path / "y.jpg")


def test_analyse_photo_round_trip_card_png_and_signal(page, tmp_path):
    result, error = analyse_photo(page)
    assert error is None and result is not None
    assert "sinus_rhythm" in result.findings.codes()
    rec = Record(id="1_2", chat_id=1, msg_id=2, chat_title="ЭКГ чат", date="2026-10-07T10:00:00",
                 caption="что тут?", predicted=sorted(result.findings.codes()))
    card = format_card(rec, result)
    assert "ЧСС" in card and "Коды: " in card and "+  — верно" in card and "ЭКГ чат" in card
    png = render_review_png(result, tmp_path / "r.png")
    assert png.exists() and png.stat().st_size > 10_000
    npz = save_signal(result.signal, tmp_path / "s.npz")
    back = load_signal(npz)
    assert back.lead_names == result.signal.lead_names and back.provenance.digitized
    assert np.allclose(back.signal, result.signal.signal, atol=0.01)
    assert back.provenance.lead_windows == result.signal.provenance.lead_windows


def test_failed_digitisation_still_yields_a_card(tmp_path):
    import cv2

    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((300, 300), 255, dtype=np.uint8))
    result, error = analyse_photo(blank)
    rec = Record(id="1_3", chat_id=1, msg_id=3)
    card = format_card(rec, result, error)
    assert result is None or "Коды" in card
    if result is None:
        assert "Не удалось оцифровать" in card and error


def test_parse_verdict_syntax():
    pred = ["sinus_rhythm", "st_elevation"]
    assert parse_verdict("+", pred).kind == "confirmed"
    assert parse_verdict("ok", pred).gold == pred
    assert parse_verdict("-", pred).kind == "rejected" and parse_verdict("—", pred).gold == []
    p = parse_verdict("-st_elevation +lbbb  там блокада", pred)
    assert (p.kind, p.gold, p.added, p.removed, p.comment) == \
        ("partial", ["lbbb", "sinus_rhythm"], ["lbbb"], ["st_elevation"], "там блокада")
    assert parse_verdict("+sinus_rhythm", pred).kind == "confirmed"     # no change to the set
    c = parse_verdict("мне кажется, артефакт", pred)
    assert c.kind == "comment" and c.gold is None and c.comment.startswith("мне")


def test_verdict_to_feedback_and_calibration_samples(tmp_path):
    rec = Record(id="1_4", chat_id=1, msg_id=4, predicted=["sinus_rhythm", "st_elevation"],
                 confidences={"sinus_rhythm": 0.9, "st_elevation": 0.6}, analysis_mode="A")
    parse = parse_verdict("-st_elevation +lbbb", rec.predicted)
    apply_verdict(rec, parse)
    assert rec.gold == ["lbbb", "sinus_rhythm"] and rec.verdict == "partial" and rec.verdict_time
    fb = feedback_from_verdict(rec, parse, reviewer="me")
    verdicts = {f.code: f.verdict for f in fb.findings}
    assert verdicts == {"sinus_rhythm": "confirmed", "st_elevation": "rejected", "lbbb": "corrected"}
    store = FeedbackStore.at(tmp_path / "fb.jsonl")
    store.append(fb)
    assert sorted(store.calibration_dataset()) == [(0.6, 0), (0.9, 1)]
    assert feedback_from_verdict(rec, parse_verdict("комментарий", rec.predicted)) is None


def test_dataset_store_upsert_lookup_and_stats(tmp_path):
    store = DatasetStore(tmp_path / "tg")
    rec = Record(id="7_1", chat_id=7, msg_id=1, status="analysed", predicted=["sinus_rhythm"], review_msg_ids=[100, 101])
    store.upsert(rec)
    store.upsert(Record(id="7_2", chat_id=7, msg_id=2, status="not_ecg"))
    rec.verdict, rec.gold = "confirmed", ["sinus_rhythm"]
    store.upsert(rec)                                         # second line for the same id
    again = DatasetStore(tmp_path / "tg")                     # reload: last line wins
    assert again.has(7, 1) and not again.has(7, 3)
    assert again.get("7_1").gold == ["sinus_rhythm"] and again.by_review_msg(101).id == "7_1"
    assert again.by_review_msg(999) is None
    s = again.stats()
    assert (s["total"], s["analysed"], s["not_ecg"], s["verified"], s["awaiting_review"]) == (2, 1, 1, 1, 0)
    assert len((tmp_path / "tg" / "dataset.jsonl").read_text().splitlines()) == 3


def test_export_verified_records_to_the_harness(page, tmp_path):
    result, _ = analyse_photo(page)
    store = DatasetStore(tmp_path / "tg")
    sigpath = save_signal(result.signal, store.signals / "1_1.npz")
    pred = sorted(result.findings.codes())
    store.upsert(Record(id="1_1", chat_id=1, msg_id=1, status="analysed", predicted=pred,
                        signal_path=str(sigpath), verdict="confirmed", gold=pred))
    store.upsert(Record(id="1_2", chat_id=1, msg_id=2, status="analysed", predicted=pred))   # unreviewed: skipped
    cases = build_telegram_dataset(store.root)
    assert len(cases) == 1 and cases[0].gold_findings == set(pred)
    report = run_validation(cases)
    assert report.n_cases == 1 and report.per_class["sinus_rhythm"].sensitivity == 1.0
    agg = agreement_summary(store.all())
    assert agg["verified"] == 1 and agg["exact_match"] == 1
    assert any("exact match" in line for line in summary_lines(store))


class _FakeMessage:
    def __init__(self, mid, data=None, text="", reply_to=None, sender=42):
        self.id, self._data, self.message, self.reply_to_msg_id = mid, data, text, reply_to
        self.photo = data is not None
        self.sender_id, self.chat_id, self.date = sender, 1, datetime(2026, 10, 7, 10, 0, tzinfo=UTC)
        self.replies = []

    async def download_media(self, _):
        return self._data

    async def reply(self, text):
        self.replies.append(text)


class _FakeClient:
    def __init__(self):
        self.sent = []
        self._next = 500

    async def send_file(self, chat, files, caption=""):
        msgs = [SimpleNamespace(id=self._next + i) for i in range(len(files))]
        self._next += len(files)
        self.sent.append((chat, list(files), caption))
        return msgs


def test_dialog_lines_keep_groups_and_channels_only():
    from cardiolens.telegram.agent import dialog_lines

    dialogs = [
        SimpleNamespace(id=-1001234567890, is_group=True, is_channel=False, title="ЭКГ разбор",
                        entity=SimpleNamespace(username="ecg_group")),
        SimpleNamespace(id=-1009876543210, is_group=False, is_channel=True, title="Кардио канал",
                        entity=SimpleNamespace(username=None)),
        SimpleNamespace(id=42, is_group=False, is_channel=False, title="Иван", entity=SimpleNamespace(username="ivan")),
    ]
    lines = dialog_lines(dialogs)
    assert len(lines) == 2
    assert lines[0].lstrip().startswith("-1001234567890  group") and lines[0].endswith("ЭКГ разбор  @ecg_group")
    assert "channel  Кардио канал" in lines[1] and "@" not in lines[1]


def test_collector_end_to_end_with_a_fake_client(page, tmp_path, monkeypatch):
    from cardiolens.telegram.agent import Collector

    cfg = TelegramConfig(store_dir=str(tmp_path / "tg"), feedback_path=str(tmp_path / "fb.jsonl"))
    store = DatasetStore(cfg.store_dir)
    client = _FakeClient()
    col = Collector(client, cfg, store, "me")
    chat = SimpleNamespace(id=1, title="ЭКГ группа")
    photo = _FakeMessage(10, data=page.read_bytes(), text="посмотрите")
    rec = asyncio.run(col.handle_photo(photo, chat))
    assert rec.status == "analysed" and "sinus_rhythm" in rec.predicted and rec.sender_hash
    assert len(client.sent) == 1 and len(client.sent[0][1]) == 2 and "ЧСС" in client.sent[0][2]
    assert rec.review_msg_ids == [500, 501]
    assert asyncio.run(col.handle_photo(photo, chat)) is None           # seen before: not re-sent

    verdict = _FakeMessage(11, text="+", reply_to=501)
    reply = asyncio.run(col.handle_review(verdict))
    assert reply.startswith("Записано") and store.get(rec.id).gold == rec.predicted
    assert json.loads((tmp_path / "fb.jsonl").read_text().splitlines()[0])["overall_agreement"] is True
    stats = asyncio.run(col.handle_review(_FakeMessage(12, text="/stats")))
    assert "верно 1" in stats
    assert asyncio.run(col.handle_review(_FakeMessage(13, text="+", reply_to=999))) is None

    blank = tmp_path / "blank.png"
    import cv2

    cv2.imwrite(str(blank), np.full((300, 300), 230, dtype=np.uint8))
    skipped = asyncio.run(col.handle_photo(_FakeMessage(14, data=blank.read_bytes()), chat))
    assert skipped.status == "not_ecg" and len(client.sent) == 1


def test_album_of_two_screen_pages_gives_one_card(tmp_path):
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer1_ingestion.screen import CHEST_6, LIMB_6, render_device_screen
    from cardiolens.telegram.agent import Collector

    s = generate_synthetic_ecg(heart_rate_bpm=80, duration_s=10.0, seed=5)
    limb = render_device_screen(s, tmp_path / "limb.png", leads=LIMB_6)
    chest = render_device_screen(s, tmp_path / "chest.png", leads=CHEST_6)
    cfg = TelegramConfig(store_dir=str(tmp_path / "tg"), feedback_path=str(tmp_path / "fb.jsonl"))
    store = DatasetStore(cfg.store_dir)
    client = _FakeClient()
    col = Collector(client, cfg, store, "me")
    chat = SimpleNamespace(id=1, title="ЭКГ группа")
    msgs = [_FakeMessage(20, data=Path(limb).read_bytes()), _FakeMessage(21, data=Path(chest).read_bytes())]
    rec = asyncio.run(col.handle_album(msgs, chat))
    assert rec.status == "analysed" and len(client.sent) == 1 and len(client.sent[0][1]) == 3   # 2 photos + review
    other = store.get(DatasetStore.record_id(1, 21))
    assert other.status == "merged" and rec.id in other.error
    assert store.stats()["awaiting_review"] == 1


def test_image_sent_as_a_file_is_read_like_a_photo(page, tmp_path):
    from cardiolens.telegram.agent import IMAGE_DOC_MAX_BYTES, Collector, is_image_message

    doc = SimpleNamespace(photo=None, document=SimpleNamespace(mime_type="image/jpeg", size=3_000_000))
    pdf = SimpleNamespace(photo=None, document=SimpleNamespace(mime_type="application/pdf", size=100_000))
    huge = SimpleNamespace(photo=None, document=SimpleNamespace(mime_type="image/png", size=IMAGE_DOC_MAX_BYTES + 1))
    assert is_image_message(doc) and not is_image_message(pdf) and not is_image_message(huge)
    assert not is_image_message(SimpleNamespace(photo=None, document=None))

    cfg = TelegramConfig(store_dir=str(tmp_path / "tg"), feedback_path=str(tmp_path / "fb.jsonl"))
    client = _FakeClient()
    col = Collector(client, cfg, DatasetStore(cfg.store_dir), "me")
    chat = SimpleNamespace(id=1, title="ЭКГ группа")
    msg = _FakeMessage(30, data=page.read_bytes())
    msg.photo, msg.document = None, SimpleNamespace(mime_type="image/png", size=len(page.read_bytes()))
    assert is_image_message(msg)
    rec = asyncio.run(col.handle_photo(msg, chat))
    assert rec.status == "analysed" and len(client.sent) == 1
    broken = _FakeMessage(31, data=b"not an image at all")
    assert asyncio.run(col.handle_photo(broken, chat)) is None and len(client.sent) == 1


def test_bot_reader_scanner_first_then_cardiolens(tmp_path):
    """D139: a paper page goes to the scanner; a low-resolution page is refused with the
    scanner's reason; a device screen (scanner: no grid) goes on to the CardioLens readers."""
    from cardiolens.layer1_ingestion.screen import CHEST_6, render_device_screen
    from cardiolens.telegram.analyze import digitize_photo

    sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=5)
    page10 = render_ecg_page(sig, tmp_path / "p10.png", px_per_mm=10, calibration_pulse=True)
    assert digitize_photo(page10).provenance.notes[0].startswith("ecgscan scan_id=")
    page6 = render_ecg_page(sig, tmp_path / "p6.png", px_per_mm=6, calibration_pulse=True)
    with pytest.raises(ValueError, match="низкое разрешение"):
        digitize_photo(page6)
    screen = render_device_screen(sig, tmp_path / "s.png", leads=CHEST_6)
    assert any(n.startswith("device screen photo") for n in digitize_photo(screen).provenance.notes)


def test_refusals_are_said_in_words():
    """D142: the card showed the readers' English refusals with their exception type."""
    from cardiolens.layer1_ingestion.sequential import RHYTHM_LOW_RES
    from cardiolens.telegram.analyze import error_text

    low = error_text(ValueError(f"{RHYTHM_LOW_RES} (5.0 px/mm): Слишком низкое разрешение."))
    assert low.startswith("Слишком низкое разрешение.") and "≈5.0" in low and "ValueError" not in low
    named = error_text(ValueError("three-channel strip: ... the leads cannot be named; photograph the whole strip"))
    assert named.startswith("Отведения на этом куске ленты не опознать")
    assert error_text(ValueError("Снимок пересвечен")) == "Снимок пересвечен"
    assert error_text(RuntimeError("boom")) == "RuntimeError: boom"


def test_scanner_scale_doubt_with_inconsistency_is_not_trusted():
    """D144: a doubtful scale alone is common on the scanner's good reads; with its own
    rhythm inconsistency it marked only bad ones."""
    from types import SimpleNamespace as NS

    from cardiolens.telegram.analyze import scale_unknown

    def res(status="needs_confirmation", conf=0.0, warnings=()):
        return NS(status=status, calibration=NS(confidence=conf), warnings=list(warnings))

    assert scale_unknown(res(warnings=["SPEED_UNCERTAIN", "RHYTHM_INCONSISTENT"]))
    assert scale_unknown(res(conf=0.6, warnings=["SPEED_UNCERTAIN", "RHYTHM_INCONSISTENT"]))
    assert not scale_unknown(res(warnings=["SPEED_UNCERTAIN", "LIMB_LEADS_INCONSISTENT"]))
    assert not scale_unknown(res(warnings=["SPEED_UNCERTAIN"]))
    assert not scale_unknown(res(conf=0.8, warnings=["RHYTHM_INCONSISTENT"]))
    assert not scale_unknown(res(status="ok", warnings=["SPEED_UNCERTAIN", "RHYTHM_INCONSISTENT"]))
