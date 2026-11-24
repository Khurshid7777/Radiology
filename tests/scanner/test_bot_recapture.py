"""SD-144: a compressed photo in a private chat gets the warning with the full-quality capture button."""

import re

import pytest

pytest.importorskip("aiogram")

from aiogram.methods import EditMessageReplyMarkup, EditMessageText, SendMessage
from aiogram.types import Update
from test_bot_group import ADMINS, NOW, photo_update, text_update
from test_bot_scan import PHOTO, Api, Reply, Server, run, scan_cfg

from ecgscan.i18n import t
from ecgscan_bot.logic import PUBLIC_URL_ENV
from ecgscan_bot.texts import tr
from ecgscan_service.miniapp import verify_ticket

ADMIN = min(ADMINS)


def of(sent, kind):
    return [m for m in sent if isinstance(m, kind)]


def doc_update(uid=1):
    return Update.model_validate({"update_id": uid, "message": {
        "message_id": 40 + uid, "date": NOW, "chat": {"id": ADMIN, "type": "private"},
        "from": {"id": ADMIN, "is_bot": False, "first_name": "D"},
        "document": {"file_id": "d", "file_unique_id": "du", "file_name": "ecg.jpg", "mime_type": "image/jpeg",
                     "file_size": 3000}}})


def album_photo(uid):
    u = photo_update(ADMIN, "private", ADMIN, uid).model_dump()
    u["message"]["media_group_id"] = "alb1"
    return Update.model_validate(u)


def test_the_warning_carries_one_capture_button_and_the_copy_is_still_read(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    from ecgscan_bot.access import AccessStore

    api = Api(Server())
    sent = run(scan_cfg(), api, AccessStore(tmp_path / "a.json"), photo_update(ADMIN, "private", ADMIN, 1),
               text_update(ADMIN, "private", ADMIN, "54 м", 2), text_update(ADMIN, "private", ADMIN, "-", 3))
    warn = [m for m in of(sent, SendMessage) if m.text.startswith(tr("compressed", "ru"))]
    assert len(warn) == 1 and t("bot.recapture", "ru") in warn[0].text
    ((button,),) = warn[0].reply_markup.inline_keyboard
    assert button.text == t("bot.recapture_btn", "ru")
    m = re.fullmatch(r"https://site\.example/miniapp/capture\?lang=ru&t=([0-9a-f.]+)", button.web_app.url)
    assert m and verify_ticket(m.group(1), {"k": "bot"}, max_minutes=10) is not None
    assert api.calls == 1                                              # the compressed copy is read as before


class AllApi(Api):
    def __init__(self, server):
        super().__init__(server)
        self.all = []

    def analyze(self, files, **kw):
        self.all.append(files)
        return super().analyze(files, **kw)


def test_a_photo_taken_by_the_button_comes_the_way_of_scan(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    from ecgscan_bot.access import AccessStore

    api = AllApi(Server(Reply(204), Reply(200, PHOTO)))
    sent = run(scan_cfg(), api, AccessStore(tmp_path / "a.json"), photo_update(ADMIN, "private", ADMIN, 1),
               text_update(ADMIN, "private", ADMIN, "54 м", 2), text_update(ADMIN, "private", ADMIN, "-", 3),
               everything=True)
    texts = [m.text for m in of(sent, SendMessage)]
    assert t("bot.scan_got", "ru") in texts
    assert [("ecg.jpg", PHOTO)] in api.all                            # the original is analysed as a file
    # the warning keeps its text; only its spent button goes
    assert not [m for m in of(sent, EditMessageText) if m.text == t("bot.scan_expired", "ru")]
    assert of(sent, EditMessageReplyMarkup)


def test_an_album_gets_one_warning_and_one_button(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    from ecgscan_bot.access import AccessStore

    sent = run(scan_cfg(), Api(Server()), AccessStore(tmp_path / "a.json"), album_photo(1), album_photo(2), album_photo(3),
               settle=0.05)
    warn = [m for m in of(sent, SendMessage) if m.text.startswith(tr("compressed", "ru"))]
    assert len(warn) == 1 and warn[0].reply_markup is not None


def test_without_a_site_the_old_warning_and_a_file_gets_none(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    from ecgscan_bot.access import AccessStore

    sent = run(scan_cfg(public_url=None), Api(Server()), AccessStore(tmp_path / "a.json"),
               photo_update(ADMIN, "private", ADMIN, 1), settle=0.05)
    warn = [m for m in of(sent, SendMessage) if m.text.startswith(tr("compressed", "ru"))]
    assert len(warn) == 1 and warn[0].text == tr("compressed", "ru") and warn[0].reply_markup is None
    sent = run(scan_cfg(), Api(Server()), AccessStore(tmp_path / "b.json"), doc_update(), settle=0.05)
    assert not [m for m in of(sent, SendMessage) if m.text.startswith(tr("compressed", "ru"))]


def test_texts_in_both_languages():
    for k in ("bot.recapture", "bot.recapture_btn"):
        assert t(k, "ru") and t(k, "uz") and t(k, "ru") != t(k, "uz")
