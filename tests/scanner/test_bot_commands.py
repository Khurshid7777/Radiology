"""SD-146: the bot's command list in private chats, and t.me/<bot>?start=scan."""

import asyncio
import re

import pytest

pytest.importorskip("aiogram")

from aiogram import Bot
from aiogram.methods import SendMessage, SetMyCommands
from aiogram.types import BotCommandScopeAllPrivateChats
from test_bot_group import ADMINS, Silent, text_update
from test_bot_scan import Api, Server, run, scan_cfg

from ecgscan.i18n import t
from ecgscan_bot.__main__ import set_commands
from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import PUBLIC_URL_ENV, command_list
from ecgscan_bot.texts import tr

ADMIN = min(ADMINS)


def commands_sent(session):
    async def go():
        await set_commands(Bot("42:TEST", session=session))
    asyncio.run(go())
    return [m for m in session.sent if isinstance(m, SetMyCommands)]


def test_the_command_list_is_set_for_private_chats_in_both_languages():
    sent = commands_sent(Silent())
    assert [m.language_code for m in sent] == [None, "uz"]
    for m, lang in zip(sent, ("ru", "uz"), strict=True):
        assert isinstance(m.scope, BotCommandScopeAllPrivateChats)
        assert [c.command for c in m.commands] == ["scan", "app", "probe"]
        assert [c.description for c in m.commands] == [d for _, d in command_list(lang)]


def test_a_refused_command_list_does_not_stop_the_bot(monkeypatch):
    async def refuse(self, bot, method, timeout=None):
        raise ConnectionError("Telegram said no")

    monkeypatch.setattr(Silent, "make_request", refuse)
    assert commands_sent(Silent()) == []                              # no exception


def test_command_texts():
    for lang in ("ru", "uz"):
        for c, d in command_list(lang):
            assert 3 <= len(d) <= 256 and re.fullmatch(r"[a-z]+", c)
    assert command_list("ru") != command_list("uz")


def test_start_scan_gives_the_capture_button_at_once(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    server = Server()
    sent = run(scan_cfg(), Api(server), AccessStore(tmp_path / "a.json"), text_update(ADMIN, "private", ADMIN, "/start scan"),
               settle=0.05)
    msgs = [m for m in sent if isinstance(m, SendMessage)]
    assert len(msgs) == 1 and msgs[0].text == t("bot.scan", "ru", minutes="10")
    ((button,),) = msgs[0].reply_markup.inline_keyboard
    assert "/miniapp/capture?lang=ru&t=" in button.web_app.url and server.asked     # the wait has begun


def test_start_scan_to_a_stranger_is_the_usual_refusal(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = run(scan_cfg(), Api(Server()), AccessStore(tmp_path / "a.json"), text_update(999, "private", 999, "/start scan"),
               settle=0.05)
    msgs = [m for m in sent if isinstance(m, SendMessage)]
    assert len(msgs) == 1 and "/code" in msgs[0].text and tr("code_bad", "ru") not in msgs[0].text


def test_start_scan_without_a_site_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = run(scan_cfg(public_url=None), Api(Server()), AccessStore(tmp_path / "a.json"),
               text_update(ADMIN, "private", ADMIN, "/start scan"), settle=0.05)
    msgs = [m for m in sent if isinstance(m, SendMessage)]
    assert len(msgs) == 1 and msgs[0].text == t("bot.scan_off", "ru") and msgs[0].reply_markup is None
