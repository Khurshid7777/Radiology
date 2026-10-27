"""M4: the Telegram client's behaviour without Telegram (TZ §6.5, SD-009)."""

import numpy as np
import pytest

from ecgscan_bot.logic import BotConfig, ScanApi, allowed, handle_file, intake, lang_of, verdict

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cryptography")
from fastapi.testclient import TestClient

from ecgscan_service.app import app
from ecgscan_service.router import KEYS_ENV, get_service
from ecgscan_service.service import ScanService, Settings

PNG = cv2.imencode(".png", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()


def _cfg(**over):
    c = dict(api_url="http://test", max_mb=20, poll_interval_s=0.05, poll_timeout_s=10, send_overlay=False,
             public_url="https://clinic.example/", allowed={111}, api_key="bot-key")
    c.update(over)
    return BotConfig(**c)


def test_whitelist_and_language():
    cfg = _cfg()
    assert allowed(cfg, 111) and not allowed(cfg, 222) and not allowed(cfg, None)
    assert lang_of("uz") == "uz" and lang_of("uz-Latn") == "uz" and lang_of("ru") == "ru" and lang_of(None) == "ru"


def test_compressed_photo_is_refused_with_advice():
    cfg = _cfg()
    assert "файлом" in intake(cfg, kind="photo", mime=None, size=None, lang="ru")
    assert "fayl" in intake(cfg, kind="photo", mime=None, size=None, lang="uz")
    assert intake(_cfg(mode="scan"), kind="document", mime="application/pdf", size=10, lang="ru").startswith("Нужен файл")
    assert intake(cfg, kind="document", mime="application/pdf", size=10, lang="ru") is None   # analyze mode reads PDF (D158)
    assert "20" in intake(cfg, kind="document", mime="image/jpeg", size=30 * 1024 * 1024, lang="ru")
    assert intake(cfg, kind="document", mime="image/jpeg", size=1000, lang="ru") is None


@pytest.fixture
def api(tmp_path, monkeypatch, result_factory):
    monkeypatch.setenv(KEYS_ENV, "bot:bot-key")
    results = {}

    def fake_scan(data, *, hints, artifacts_dir):
        return results["next"].model_copy(deep=True)

    svc = ScanService(Settings(root=tmp_path / "svc", keep_originals=False, retention_days=30, max_bytes=10**6,
                               mime=("image/png",), workers=1, feedback_path=tmp_path / "fb.jsonl",
                               public_url=None), scan_fn=fake_scan, vault=None)
    app.dependency_overrides[get_service] = lambda: svc
    yield ScanApi(_cfg(), http=TestClient(app)), results, result_factory
    app.dependency_overrides.clear()
    svc.close()


def test_ok_verdict_in_uzbek_with_link(api):
    client, results, make = api
    results["next"] = make(status="ok")
    text, sid = handle_file(_cfg(), client, PNG, "ecg.png", "uz")
    assert text.startswith("Tayyor") and "25 mm/s" in text and "10 mm/mV" in text
    assert f"https://clinic.example/scans/{sid}" in text and "fotosuratidan" in text


def test_rejection_explains_what_to_do(api):
    client, results, make = api
    results["next"] = make(status="rejected", reject_reasons=["GLARE", "LOW_RESOLUTION"], leads={})
    text, _ = handle_file(_cfg(), client, PNG, "ecg.png", "ru")
    assert text.startswith("Снимок не обработан") and "Блик" in text and "файлом" in text
    assert "clinic.example" not in text                  # nothing to open for a rejected photo


def test_needs_confirmation_lists_the_warnings(api):
    client, results, make = api
    results["next"] = make(status="needs_confirmation", warnings=["BEATS_INCONSISTENT"])
    text, _ = handle_file(_cfg(), client, PNG, "ecg.png", "ru")
    assert text.startswith("Нужно подтверждение") and "ЧСС" in text


def test_server_trouble_is_answered_not_raised(api):
    client, _, _ = api
    client.headers = {"X-API-Key": "wrong"}
    text, sid = handle_file(_cfg(), client, PNG, "ecg.png", "ru")
    assert sid is None and "недоступен" in text


def test_verdict_without_public_url_shows_the_id(result_factory):
    body = {"scan_id": "abc", "status": "ok", "result": result_factory(status="ok").model_dump(mode="json"),
            "messages": {"ru": {"status": "Готово", "reasons": [], "warnings": []}}}
    assert "scan_id: abc" in verdict(_cfg(public_url=None), body, "ru")


# --- analyze mode: the CardioVision conclusion and the PDF (D161) ---------------------------

def test_caption_gives_age_and_sex():
    from ecgscan_bot.logic import Patient, parse_caption

    assert parse_caption("54 м") == Patient(age_years=54.0, sex="male")
    assert parse_caption("ж, 7 лет") == Patient(age_years=7.0, sex="female")
    assert parse_caption("erkak 63") == Patient(age_years=63.0, sex="male")
    assert parse_caption("ayol") == Patient(sex="female")
    assert parse_caption("3 мес м") == Patient(age_days=pytest.approx(91.2), sex="male")   # an infant: days
    assert parse_caption("10 kun a") == Patient(age_days=10.0, sex="female")
    assert parse_caption("ЭКГ 2026") == Patient()                      # a year is not an age
    assert parse_caption(None) == Patient() and not Patient(age_years=5).given
    assert Patient(age_years=5, sex="male").given


def _page_png(tmp_path, px_per_mm=10):
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer1_ingestion.digitize import render_ecg_page

    png = tmp_path / f"page{px_per_mm}.png"
    render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), png,
                    px_per_mm=px_per_mm, calibration_pulse=True)
    return png.read_bytes()


@pytest.fixture
def analyze_api():
    from cardiolens.api.app import app as cardiolens_app
    from ecgscan_bot.logic import AnalyzeApi

    return AnalyzeApi(_cfg(api_key="test-key", mode="analyze"), http=TestClient(cardiolens_app))


def test_analyze_mode_answers_with_the_conclusion_and_a_pdf(analyze_api, tmp_path):
    from ecgscan_bot.logic import handle_analysis

    page = _page_png(tmp_path)
    text, aid = handle_analysis(_cfg(), analyze_api, [("page.png", page)], "ru", "54 м")
    assert aid and "Не заменяет врача" in text and "Критические" not in text, text
    assert "Возраст и пол не указаны" not in text
    assert analyze_api.pdf(aid)[:4] == b"%PDF"
    assert analyze_api.overlay(aid)[:3] == b"\xff\xd8\xff"      # the traces over the page (D163)
    uz, _ = handle_analysis(_cfg(), analyze_api, [("page.png", page)], "uz", "54 e")
    assert "Shifokor" in uz
    # No caption: the analysis still runs, the reply says the norms were an adult's.
    bare, aid2 = handle_analysis(_cfg(), analyze_api, [("page.png", page)], "ru", None)
    assert aid2 and "Возраст и пол не указаны" in bare


def test_analyze_mode_takes_an_album_as_one_record(analyze_api, tmp_path):
    from ecgscan_bot.logic import handle_analysis

    page = _page_png(tmp_path)
    text, aid = handle_analysis(_cfg(), analyze_api, [("a.png", page), ("b.png", page)], "ru", "40 ж")
    assert aid and "Не заменяет врача" in text, text


def test_analyze_mode_refusal_keeps_the_telegram_wording(analyze_api, tmp_path):
    # D170: a camera-sized photo with a coarse grid is still refused (a small copy is read coarse).
    import cv2

    from ecgscan_bot.logic import handle_analysis

    page = cv2.imdecode(np.frombuffer(_page_png(tmp_path, px_per_mm=6), np.uint8), cv2.IMREAD_COLOR)
    big = cv2.copyMakeBorder(page, 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    data = cv2.imencode(".png", big)[1].tobytes()
    text, aid = handle_analysis(_cfg(), analyze_api, [("page.png", data)], "ru")
    assert aid is None and "файлом" in text and "галере" not in text, text
    small, aid2 = handle_analysis(_cfg(), analyze_api, [("page.png", _page_png(tmp_path, px_per_mm=4))], "ru", "54 м")
    assert aid2 and "сжат" in small.lower(), small
