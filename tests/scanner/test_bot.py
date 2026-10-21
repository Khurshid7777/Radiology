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
    assert intake(cfg, kind="document", mime="application/pdf", size=10, lang="ru").startswith("Нужен файл")
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
