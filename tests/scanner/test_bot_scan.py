"""SD-137: ``/scan`` opens the capture page as a Mini App with a one-time ticket; the photo taken there
comes from the server as the original and goes the way of a file sent to the chat."""

import asyncio
import re

import pytest

pytest.importorskip("aiogram")

from aiogram import Bot
from aiogram.methods import EditMessageReplyMarkup, EditMessageText, SendMessage
from test_bot_group import ADMINS, GROUP, FakeApi, Silent, cfg, text_update

from ecgscan.i18n import t
from ecgscan_bot.__main__ import build, setup_logging
from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import PUBLIC_URL_ENV, miniapp_button_url, miniapp_take, photo_name
from ecgscan_bot.texts import tr
from ecgscan_service.miniapp import verify_ticket

ADMIN = min(ADMINS)
PHOTO = b"\xff\xd8\xff" + b"original bytes of the phone's photo" * 50


class Reply:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


class Server:
    """The server's inbox as the bot sees it: a list of answers, then 204 for ever."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def get(self, path, headers=None):
        self.asked.append((path, dict(headers or {})))
        got = self.answers.pop(0) if self.answers else Reply(204)
        if isinstance(got, Exception):
            raise got
        return got


class Api(FakeApi):
    def __init__(self, server):
        super().__init__()
        self.http, self.headers, self.files = server, {"X-API-Key": "k"}, None

    def analyze(self, files, **kw):
        self.files = files
        return super().analyze(files, **kw)


def run(c, api, store, *updates, settle: float = 0.3, everything: bool = False):
    session = Silent()
    bot = Bot("42:TEST", session=session)

    async def go():
        dp = build(c, bot, api, store)
        for u in updates:
            await dp.feed_update(bot, u)
            await asyncio.sleep(settle)            # the waiting task works between the doctor's messages

    asyncio.run(go())
    return [m for m in session.sent if everything or isinstance(m, SendMessage)]


def scan_cfg(**kw):
    return cfg(**{"public_url": "https://site.example", "miniapp_poll_s": 0.01, **kw})


def test_scan_sends_a_web_app_button_with_a_fresh_ticket_signed_with_the_bots_key(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server()
    sent = run(scan_cfg(), Api(server), AccessStore(tmp_path / "a.json"), text_update(ADMIN, "private", ADMIN, "/scan"),
               settle=0.05)
    assert len(sent) == 1 and sent[0].chat_id == ADMIN and "10" in sent[0].text
    ((button,),) = sent[0].reply_markup.inline_keyboard
    m = re.fullmatch(r"https://site\.example/miniapp/capture\?lang=ru&t=([0-9a-f.]+)", button.web_app.url)
    assert m, button.web_app.url
    ticket = verify_ticket(m.group(1), {"k": "bot"}, max_minutes=10)
    assert ticket is not None                                    # the server accepts it with the bot's key
    assert str(ADMIN) not in button.web_app.url                  # no Telegram id leaves the bot
    assert server.asked and server.asked[0] == (f"/scans/miniapp/inbox/{ticket.id}", {"X-API-Key": "k"})


def test_the_photo_from_the_page_goes_the_way_of_a_file_sent_to_the_chat(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server(Reply(204), RuntimeError("network"), Reply(204), Reply(200, PHOTO))
    api = Api(server)
    sent = run(scan_cfg(), api, AccessStore(tmp_path / "a.json"),
               text_update(ADMIN, "private", ADMIN, "/scan", 1),
               text_update(ADMIN, "private", ADMIN, "54 м", 2),          # the answer to "sex and age" (D209)
               text_update(ADMIN, "private", ADMIN, "боль в груди", 3),  # the doctor's note
               everything=True)
    gone = [m for m in sent if isinstance(m, EditMessageReplyMarkup)]
    assert len(gone) == 1 and gone[0].reply_markup is None and gone[0].chat_id == ADMIN       # the spent button goes
    assert not [m for m in sent if isinstance(m, EditMessageText)]
    sent = [m for m in sent if isinstance(m, SendMessage)]
    texts = [m.text for m in sent]
    assert texts[1] == t("bot.scan_got", "ru") and texts[2] == tr("ask_patient", "ru")
    assert api.calls == 1 and api.files == [("ecg.jpg", PHOTO)]     # the same bytes, one analysis
    assert any("Синусовый ритм" in x for x in texts)
    assert len(server.asked) == 4                                    # the bot stops asking once it has the photo


def test_scan_is_refused_to_a_stranger_silent_in_a_group_and_plain_without_a_site(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server()
    stranger = run(scan_cfg(), Api(server), AccessStore(tmp_path / "a.json"), text_update(999, "private", 999, "/scan"),
                   settle=0.05)
    assert len(stranger) == 1 and stranger[0].reply_markup is None and "/code" in stranger[0].text
    assert not run(scan_cfg(), Api(server), AccessStore(tmp_path / "b.json"),
                   text_update(GROUP, "supergroup", ADMIN, "/scan"), settle=0.05)
    for c in (scan_cfg(public_url=None), scan_cfg(public_url="http://plain.example"), scan_cfg(api_key=None)):
        off = run(c, Api(server), AccessStore(tmp_path / "c.json"), text_update(ADMIN, "private", ADMIN, "/scan"),
                  settle=0.05)
        assert len(off) == 1 and off[0].reply_markup is None and off[0].text == t("bot.scan_off", "ru")
    assert not server.asked


def test_the_bot_stops_asking_when_the_ticket_runs_out_or_the_photo_is_gone(tmp_path, monkeypatch):
    """... and says so in the button's own message: a photo that never came is not a silent loss."""
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server()
    sent = run(scan_cfg(miniapp_ticket_minutes=1 / 60), Api(server), AccessStore(tmp_path / "a.json"),
               text_update(ADMIN, "private", ADMIN, "/scan"), settle=2.3, everything=True)
    asked = len(server.asked)
    assert 1 <= asked < 400
    assert [type(m) for m in sent] == [SendMessage, EditMessageText]
    assert sent[1].text == t("bot.scan_expired", "ru") and sent[1].chat_id == ADMIN and sent[1].reply_markup is None
    gone = Server(Reply(404))
    sent = run(scan_cfg(), Api(gone), AccessStore(tmp_path / "b.json"), text_update(ADMIN, "private", ADMIN, "/scan"),
               everything=True)
    assert len(gone.asked) == 1 and isinstance(sent[-1], EditMessageText)


def test_earlier_buttons_keep_working_and_only_the_oldest_gives_way(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server()
    sent = run(scan_cfg(miniapp_poll_s=0.05, miniapp_max_buttons=2), Api(server), AccessStore(tmp_path / "a.json"),
               *[text_update(ADMIN, "private", ADMIN, "/scan", i) for i in (1, 2, 3)], settle=0.4, everything=True)
    order = list(dict.fromkeys(p for p, _ in server.asked))
    assert len(order) == 3
    late = {p for p, _ in server.asked[-6:]}
    assert late == set(order[1:])                                  # the first ticket is no longer asked for
    retired = [m for m in sent if isinstance(m, EditMessageText)]
    assert retired[0].text == t("bot.scan_expired", "ru") and retired[0].message_id == 1   # its message says so
    assert sent.index(retired[0]) < len(sent) - 2                  # ... at once (the rest retire as the bot stops)


def test_the_bots_log_does_not_carry_every_poll_with_the_tickets_number(monkeypatch):
    import logging

    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.NOTSET)
    setup_logging()
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    assert logging.getLogger("httpx").isEnabledFor(logging.WARNING)


def test_helpers():
    assert miniapp_button_url(cfg(public_url=None), "ru") is None
    url, tid, expiry = miniapp_button_url(scan_cfg(), "uz", now=1_800_000_000)
    assert url.startswith("https://site.example/miniapp/capture?lang=uz&t=" + tid) and expiry == 1_800_000_600
    assert miniapp_take(Api(Server(Reply(200, b"x"))), "t") == b"x"
    assert miniapp_take(Api(Server(Reply(204))), "t") is None
    assert miniapp_take(Api(Server(Reply(500))), "t") is None
    assert miniapp_take(Api(Server(Reply(404))), "t") is False
    assert photo_name(b"\xff\xd8\xff\xe0") == "ecg.jpg" and photo_name(b"\x89PNG\r\n\x1a\n") == "ecg.png"
    assert photo_name(b"\x00\x00\x00\x18ftypheic") == "ecg.heic" and photo_name(b"RIFF\x00\x00\x00\x00WEBP") == "ecg.webp"
    assert photo_name(b"MM\x00*") == "ecg.tiff" and photo_name(b"?") == "ecg.jpg"
