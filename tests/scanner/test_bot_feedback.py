"""SD-108: the doctor's verdict under a reply — "right" / "wrong" and, after "wrong", what it really is —
goes to the server's feedback log; no photo and no patient field travel with it."""

import json

import pytest

pytest.importorskip("aiogram")

from aiogram.methods import SendMessage  # noqa: E402
from aiogram.types import Update  # noqa: E402
from test_bot_group import ADMINS, GROUP, NOW, FakeApi, cfg, chats, photo_update, run, text_update  # noqa: E402

from ecgscan_bot.access import AccessStore  # noqa: E402
from ecgscan_bot.logic import clean_comment, feedback_buttons, parse_feedback  # noqa: E402


class Api(FakeApi):
    def __init__(self, known=("a1",)):
        super().__init__()
        self.known, self.fb = set(known), []

    def feedback(self, aid, agree, reviewer, comment=None, source=None):
        if aid not in self.known:
            return False
        self.fb.append({"aid": aid, "agree": agree, "reviewer": reviewer, "comment": comment, "source": source})
        return True


def press(user_id: int, data: str, uid: int = 9) -> Update:
    return Update.model_validate({"update_id": uid, "callback_query": {
        "id": "1", "chat_instance": "x", "data": data, "from": {"id": user_id, "is_bot": False, "first_name": "A"},
        "message": {"message_id": 5, "date": NOW, "chat": {"id": user_id, "type": "private"}}}})


def buttons(sent) -> list[str]:
    return [b.callback_data for m in sent if isinstance(m, SendMessage) and m.reply_markup
            for row in m.reply_markup.inline_keyboard for b in row]


def test_the_reply_carries_the_verdict_buttons_and_a_group_photo_is_marked(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    store.set_group(GROUP, "ЭКГ клуб", True)
    sent = run(cfg(), Api(), store, photo_update(GROUP, "supergroup", 999))
    assert buttons(sent).count("fb:ok:a1:g") == len(ADMINS) and "fb:no:a1:g" in buttons(sent)
    sent = run(cfg(), Api(), store, photo_update(111, "private", 111, title=None), press(111, "q:skip", 2),
               press(111, "q:no", 3))
    assert "fb:ok:a1" in buttons(sent) and "fb:no:a1" in buttons(sent)


def test_right_and_wrong_with_the_real_diagnosis_are_recorded(tmp_path):
    api, store = Api(), AccessStore(tmp_path / "a.json")
    sent = run(cfg(), api, store, press(111, "fb:ok:a1"))
    assert api.fb == [{"aid": "a1", "agree": True, "reviewer": "tg:111", "comment": None, "source": "bot"}]
    assert chats(sent) == [111]
    api.fb.clear()
    sent = run(cfg(), api, store, press(222, "fb:no:a1:g", 1),
               text_update(222, "private", 222, "  Нижний ИМ,\nподъём ST в II, III, aVF  ", 2),
               text_update(222, "private", 222, "ещё текст", 3))
    assert [(f["agree"], f["comment"], f["source"]) for f in api.fb] == [
        (False, None, "bot-group"), (False, "Нижний ИМ, подъём ST в II, III, aVF", "bot-group")]
    assert len(api.fb) == 2                                           # the later text is not a second comment


def test_only_who_may_use_the_bot_gives_a_verdict_and_bad_buttons_do_nothing(tmp_path):
    api, store = Api(), AccessStore(tmp_path / "a.json")
    run(cfg(), api, store, press(999, "fb:ok:a1"))                    # not allowed
    run(cfg(), api, store, press(111, "fb:maybe:a1"), press(111, "fb:ok:../x"), press(111, "fb:ok:a1:zz"))
    assert api.fb == []
    sent = run(cfg(), api, store, press(111, "fb:ok:gone1"))          # the server no longer keeps it
    assert api.fb == [] and "не удалось" in sent[-1].text


def test_the_buttons_and_the_comment_cleaning():
    assert [d for _, d in feedback_buttons("ab12", "uz", group=True)] == ["fb:ok:ab12:g", "fb:no:ab12:g"]
    assert parse_feedback("fb:no:ab12") == ("ab12", False, "bot")
    assert parse_feedback("fb:ok:ab12:g") == ("ab12", True, "bot-group")
    assert parse_feedback("re:ab12:x") is None and parse_feedback(None) is None
    assert clean_comment("  a\x00b\n\tc  ") == "a b c" and clean_comment(" \n ") is None
    assert len(clean_comment("я" * 1000)) == 300


def test_the_verdict_lands_in_the_servers_feedback_log(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app as cardiolens_app
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer1_ingestion.digitize import render_ecg_page
    from ecgscan_bot.logic import AnalyzeApi

    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(log_path))
    png = tmp_path / "page.png"
    render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), png, px_per_mm=10, calibration_pulse=True)
    api = AnalyzeApi(cfg(api_key="test-key"), http=TestClient(cardiolens_app))
    aid = api.analyze([("page.png", png.read_bytes())])["analysis_id"]
    assert api.feedback(aid, False, "tg:111", "Нижний ИМ", "bot-group") is True
    assert api.feedback("0" * 12, True, "tg:111") is False
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["analysis_id"] == aid
    assert rows[0]["overall_agreement"] is False and rows[0]["reviewer"] == "tg:111"
