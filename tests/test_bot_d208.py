"""D208: the Telegram bot — invite codes, symptoms, a link for a colleague, the reply language."""

import json
import os
import stat

import pytest

from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import (
    BotConfig,
    PaperScale,
    allowed,
    analyse,
    is_admin,
    parse_caption,
    parse_symptoms,
    render,
    symptom_buttons,
    toggle_symptom,
)

pytest.importorskip("cv2")
from fastapi.testclient import TestClient

ADMIN = 111


def _cfg(**over):
    c = dict(api_url="http://test", max_mb=20, poll_interval_s=0.05, poll_timeout_s=10, send_overlay=False,
             public_url="https://cardio.example", allowed={ADMIN}, api_key="test-key", mode="analyze")
    c.update(over)
    return BotConfig(**c)


def test_invite_codes_open_and_revoke_access(tmp_path):
    store = AccessStore(tmp_path / "access.json")
    cfg = _cfg()
    assert allowed(cfg, ADMIN, "private", store) and is_admin(cfg, ADMIN)
    assert not allowed(cfg, 222, "private", store)
    code = store.new_code("clinic A", ADMIN)
    assert len(code) == 8 and code.isalnum() and code.isupper()
    assert not store.join(222, "WRONG123") and not allowed(cfg, 222, "private", store)
    assert store.join(222, code.lower())                      # typed in lower case still works
    assert allowed(cfg, 222, "private", store) and not is_admin(cfg, 222)
    assert not allowed(cfg, 222, "group", store)              # never in a group (D136)
    assert store.codes()[0]["users"] == 1
    assert store.revoke(code) and not allowed(cfg, 222, "private", store)
    assert not store.join(333, code)                          # a revoked code lets no one in
    assert allowed(cfg, ADMIN, "private", store)              # an administrator cannot be revoked


def test_the_store_keeps_ids_only_and_is_private(tmp_path):
    path = tmp_path / "bot" / "access.json"
    store = AccessStore(path)
    store.join(222, store.new_code("x", ADMIN))
    store.set_lang(222, "uz")
    data = json.loads(path.read_text())
    assert set(data["users"]["222"]) == {"code", "at", "lang"}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    again = AccessStore(path)                                  # survives a restart
    assert again.is_member(222) and again.lang(222) == "uz"


def test_symptoms_from_a_caption_and_not_the_age():
    assert parse_symptoms("54 м, боль в груди 3 ч") == (["chest_pain"], "lt12h")
    assert parse_symptoms("обморок, одышка") == (["dyspnea", "syncope"], None)
    assert parse_symptoms("ko'krakda og'riq 30 soat") == (["chest_pain"], "12to48h")
    assert parse_symptoms("54 м") == ([], None)               # a lone number is the age, not the onset
    assert parse_caption("54 м, боль в груди 3 ч").age_years == 54


def test_symptom_buttons_toggle():
    sel, onset = toggle_symptom([], None, "y=syncope")
    sel, onset = toggle_symptom(sel, onset, "o=lt12h")
    assert (sel, onset) == (["syncope"], "lt12h")
    rows = symptom_buttons(sel, onset, "ru")
    labels = [label for row in rows for label, _ in row]
    assert "✓ Обморок" in labels and "✓ < 12 ч" in labels and rows[-1][0][1] == "go"
    assert toggle_symptom(sel, onset, "y=syncope") == ([], "lt12h")
    assert toggle_symptom(sel, onset, "y=<script>") == (["syncope"], "lt12h")


def _page(tmp_path):
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer1_ingestion.digitize import render_ecg_page

    png = tmp_path / "page.png"
    render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), png, px_per_mm=10, calibration_pulse=True)
    return png.read_bytes()


@pytest.fixture
def api():
    from cardiolens.api.app import app as cardiolens_app
    from ecgscan_bot.logic import AnalyzeApi

    return AnalyzeApi(_cfg(), http=TestClient(cardiolens_app))


def test_symptoms_reach_the_analysis_and_the_reply_carries_the_advice(api, tmp_path):
    page = _page(tmp_path)
    plain, aid, _ = analyse(_cfg(), api, [("page.png", page)], "ru", "54 м")
    assert aid and "Симптомы" not in plain
    text, _, body2 = analyse(_cfg(), api, [("page.png", page)], "ru", "54 м", symptoms=["chest_pain"],
                                onset="lt12h")
    assert "боль в груди" in text and body2["symptom_advice"] and body2["symptom_advice"][0]["urgent"]
    text3, _, _ = analyse(_cfg(), api, [("page.png", page)], "ru", "54 м обморок")
    assert "обморок" in text3.lower()


def test_the_reply_switches_language_without_a_new_analysis(api, tmp_path):
    _, _, body = analyse(_cfg(), api, [("page.png", _page(tmp_path))], "ru", "54 м")
    uz = render(body, "uz", PaperScale(), True)
    ru = render(body, "ru", PaperScale(), True)
    assert uz != ru and "Shifokor" in uz


def test_a_link_for_a_colleague_opens_the_pdf_without_a_key(api, tmp_path, monkeypatch):
    monkeypatch.setenv("ECGSCAN_BOT_PUBLIC_URL", "https://cardio.vision-ai.uz")
    _, aid, _ = analyse(_cfg(), api, [("page.png", _page(tmp_path))], "ru", "54 м")
    url = api.share(aid)
    assert url.startswith("https://cardio.vision-ai.uz/s/")
    from cardiolens.api.app import app as cardiolens_app

    r = TestClient(cardiolens_app).get(url.removeprefix("https://cardio.vision-ai.uz"))
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert api.share("000000000000") is None                  # unknown or gone: no link, no exception


# --- D209: a plain photo, questions after the file, the doctor's note ---

def test_a_plain_photo_is_taken_in_analyze_mode():
    from ecgscan_bot.logic import intake

    assert intake(_cfg(), kind="photo", mime=None, size=None, lang="ru") is None


def test_questions_start_with_what_the_caption_did_not_say():
    from ecgscan_bot.logic import first_question, patient_answer

    assert first_question(None) == "patient" and first_question("54") == "patient"
    assert first_question("54 м") == "note"
    assert patient_answer("54 м").age_years == 54 and patient_answer("ж").sex == "female"
    assert patient_answer("не знаю") is None and patient_answer("") is None


def test_the_note_is_cleaned():
    from cardiolens.api.service import NOTE_MAX, clean_note

    assert clean_note("нет") is None and clean_note("Yo'q") is None and clean_note("  ") is None
    assert clean_note("ИБС,\nстенокардия\x07") == "ИБС, стенокардия"
    assert len(clean_note("а" * 1000)) == NOTE_MAX


def test_the_note_is_in_the_reply_and_never_reaches_the_narrative(api, tmp_path, monkeypatch):
    import cardiolens.pipeline as pl

    seen = {}
    real = pl.interpret

    def spy(data, *a, **k):
        seen["ctx"] = data.context
        return real(data, *a, **k)

    monkeypatch.setattr(pl, "interpret", spy)
    text, aid, _ = analyse(_cfg(), api, [("page.png", _page(tmp_path))], "ru", "54 м",
                           note="ИБС, постинфарктный кардиосклероз")
    assert aid and "Диагноз / описание со слов врача: ИБС, постинфарктный кардиосклероз" in text
    assert seen["ctx"].clinical_note is None
    plain, _, _ = analyse(_cfg(), api, [("page.png", _page(tmp_path))], "ru", "54 м")
    assert "со слов врача" not in plain
