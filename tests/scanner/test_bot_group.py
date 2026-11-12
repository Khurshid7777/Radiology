"""SD-107: the bot in a group — it reads ECG photos, says nothing there, and sends the result to the
administrators in private. The dispatcher is fed Telegram updates; the bot's session talks to no one."""

import asyncio
import datetime as dt

import pytest

pytest.importorskip("aiogram")

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetFile, SendDocument, SendMessage, SendPhoto
from aiogram.types import Chat, File, Message, Update

from ecgscan_bot.__main__ import build
from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import (
    AnalyzeRefused,
    BotConfig,
    HourlyLimit,
    group_head,
    message_link,
    watched_group,
)

ADMINS = {111, 222}
GROUP = -1001234567890
NOW = int(dt.datetime(2026, 11, 12, tzinfo=dt.UTC).timestamp())


class Silent(BaseSession):
    """Records what the bot would send; answers what the handlers need."""

    def __init__(self):
        super().__init__()
        self.sent = []

    async def make_request(self, bot, method, timeout=None):
        self.sent.append(method)
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="photos/x.jpg")
        if isinstance(method, (SendMessage, SendPhoto, SendDocument)):
            return Message(message_id=len(self.sent), date=dt.datetime.now(dt.UTC),
                           chat=Chat(id=method.chat_id, type="private"))
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield b"\xff\xd8jpeg"

    async def close(self):
        pass


class FakeApi:
    def __init__(self, refuse: str | None = None):
        self.refuse, self.calls = refuse, 0

    def analyze(self, files, **kw):
        self.calls += 1
        if self.refuse:
            raise AnalyzeRefused(self.refuse)
        return {"analysis_id": "a1", "conclusion_ru": "Синусовый ритм, ЧСС 72.", "conclusion_uz": "Sinus ritmi, YUCH 72.",
                "critical_alerts": []}

    def overlay(self, aid):
        return b"jpg"

    def pdf(self, aid):
        return b"%PDF"

    def share(self, aid):
        return None


def cfg(**kw) -> BotConfig:
    base = dict(api_url="http://x", max_mb=20, poll_interval_s=0.01, poll_timeout_s=1, send_overlay=False,
                public_url=None, allowed=set(ADMINS), api_key="k", mode="analyze", group_recipients=set(ADMINS))
    return BotConfig(**{**base, **kw})


def photo_update(chat_id: int, chat_type: str, user_id: int, uid: int = 1, title: str | None = "ЭКГ клуб") -> Update:
    return Update.model_validate({"update_id": uid, "message": {
        "message_id": 40 + uid, "date": NOW, "chat": {"id": chat_id, "type": chat_type, "title": title},
        "from": {"id": user_id, "is_bot": False, "first_name": "D"},
        "photo": [{"file_id": "f", "file_unique_id": "u", "width": 1280, "height": 900}]}})


def text_update(chat_id: int, chat_type: str, user_id: int, text: str, uid: int = 1) -> Update:
    entities = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}] if text.startswith("/") else []
    return Update.model_validate({"update_id": uid, "message": {
        "message_id": 40 + uid, "date": NOW, "chat": {"id": chat_id, "type": chat_type, "title": "ЭКГ клуб"},
        "from": {"id": user_id, "is_bot": False, "first_name": "D"}, "text": text, "entities": entities}})


def run(c: BotConfig, api, store, *updates):
    session = Silent()
    bot = Bot("42:TEST", session=session)

    async def go():
        dp = build(c, bot, api, store)
        for u in updates:
            await dp.feed_update(bot, u)

    asyncio.run(go())
    return session.sent


def chats(sent, kind=SendMessage):
    return [m.chat_id for m in sent if isinstance(m, kind)]


def test_a_group_photo_is_answered_to_the_administrators_in_private_only(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    store.set_group(GROUP, "ЭКГ клуб", True, by=111)
    api = FakeApi()
    sent = run(cfg(), api, store, photo_update(GROUP, "supergroup", user_id=999))
    assert api.calls == 1                                         # one analysis, whoever posted it
    assert GROUP not in chats(sent) + chats(sent, SendPhoto) + chats(sent, SendDocument)
    assert sorted(set(chats(sent))) == [111, 222]
    assert sorted(chats(sent, SendDocument)) == [111, 222]        # the PDF
    first = next(m for m in sent if isinstance(m, SendMessage) and m.chat_id == 111)
    assert "ЭКГ клуб" in first.text and "t.me/c/1234567890/41" in first.text and "Синусовый ритм" in first.text


def test_a_group_that_is_not_switched_on_is_not_read(tmp_path):
    api = FakeApi()
    sent = run(cfg(), api, AccessStore(tmp_path / "a.json"), photo_update(GROUP, "supergroup", user_id=111))
    assert api.calls == 0 and not chats(sent)


def test_the_bot_never_writes_in_a_group(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    sent = run(cfg(), FakeApi(), store, text_update(GROUP, "supergroup", 999, "привет", 1),
               text_update(GROUP, "supergroup", 999, "/start", 2), text_update(GROUP, "group", 111, "/invite x", 3))
    assert not chats(sent)


def test_a_refused_photo_is_reported_or_only_logged(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    store.set_group(GROUP, "ЭКГ клуб", True)
    sent = run(cfg(), FakeApi(refuse="Сетка не найдена."), store, photo_update(GROUP, "supergroup", 999))
    assert sorted(chats(sent)) == [111, 222] and not chats(sent, SendDocument)
    assert all("Сетка не найдена." in m.text for m in sent if isinstance(m, SendMessage))
    quiet = run(cfg(group_notify_refusals=False), FakeApi(refuse="Сетка не найдена."), store,
                photo_update(GROUP, "supergroup", 999))
    assert not chats(quiet)


def test_only_the_chosen_administrators_get_the_group_results(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    store.set_group(GROUP, "ЭКГ клуб", True)
    sent = run(cfg(group_recipients={222}), FakeApi(), store, photo_update(GROUP, "supergroup", 999))
    assert set(chats(sent)) == {222}


def test_an_administrator_switches_a_group_on_from_the_private_chat(tmp_path):
    store = AccessStore(tmp_path / "a.json")
    c = cfg()
    sent = run(c, FakeApi(), store, text_update(GROUP, "supergroup", 111, "/watch"))
    assert sorted(chats(sent)) == [111, 222] and not store.group_on(GROUP)        # the offer, in private
    sent = run(c, FakeApi(), store, text_update(GROUP, "supergroup", 999, "/watch"))
    assert not chats(sent)                                                        # a stranger: nothing

    def press(user_id: int, data: str) -> Update:
        return Update.model_validate({"update_id": 9, "callback_query": {
            "id": "1", "chat_instance": "x", "data": data, "from": {"id": user_id, "is_bot": False, "first_name": "A"},
            "message": {"message_id": 5, "date": NOW, "chat": {"id": user_id, "type": "private"}}}})

    run(c, FakeApi(), store, press(999, f"gr:{GROUP}:on"))
    assert not store.group_on(GROUP)                                              # not an administrator
    run(c, FakeApi(), store, press(111, f"gr:{GROUP}:on"))
    assert store.group_on(GROUP) and watched_group(c, GROUP, "supergroup", store)
    run(c, FakeApi(), store, press(111, "gr:-555:on"))
    assert not store.group_on(-555)                                               # a group the bot never met
    run(c, FakeApi(), store, press(222, f"gr:{GROUP}:off"))
    assert not store.group_on(GROUP)


def test_the_private_chat_works_as_before(tmp_path):
    sent = run(cfg(), FakeApi(), AccessStore(tmp_path / "a.json"), text_update(111, "private", 111, "/admins"))
    assert chats(sent) == [111] and "111, 222" in sent[-1].text
    sent = run(cfg(), FakeApi(), AccessStore(tmp_path / "a.json"), text_update(999, "private", 999, "/admins"))
    assert "111" not in sent[-1].text                                             # ids are for administrators


def test_links_limits_and_the_environment(monkeypatch):
    assert message_link(GROUP, 7) == "https://t.me/c/1234567890/7"
    assert message_link(GROUP, 7, "ecgclub") == "https://t.me/ecgclub/7"
    assert message_link(-4455, 7) is None                                         # a basic group has no links
    assert group_head("ЭКГ клуб", None, "uz").startswith("📥")
    limit = HourlyLimit(2)
    assert [limit.allow(1, t) for t in (0, 1, 2, 3700)] == [True, True, False, True]
    monkeypatch.setenv("ECGSCAN_BOT_ALLOWED", "111,222")
    monkeypatch.setenv("ECGSCAN_BOT_GROUPS", f"{GROUP}")
    monkeypatch.setenv("ECGSCAN_BOT_GROUP_RECIPIENTS", "222,999")                 # 999 is no administrator: dropped
    c = BotConfig.load()
    assert c.groups == {GROUP} and c.group_recipients == {222}
    assert watched_group(c, GROUP, "supergroup") and not watched_group(c, GROUP, "private")
    monkeypatch.delenv("ECGSCAN_BOT_GROUP_RECIPIENTS")
    assert BotConfig.load().group_recipients == {111, 222}


def test_a_private_photo_still_gets_its_questions_and_the_full_reply(tmp_path):
    def press(data: str, uid: int) -> Update:
        return Update.model_validate({"update_id": uid, "callback_query": {
            "id": "1", "chat_instance": "x", "data": data, "from": {"id": 111, "is_bot": False, "first_name": "A"},
            "message": {"message_id": 5, "date": NOW, "chat": {"id": 111, "type": "private"}}}})

    api = FakeApi()
    sent = run(cfg(), api, AccessStore(tmp_path / "a.json"), photo_update(111, "private", 111, title=None),
               press("q:skip", 2), press("q:no", 3))
    assert api.calls == 1 and set(chats(sent)) == {111}
    assert chats(sent, SendPhoto) == [111] and chats(sent, SendDocument) == [111]
    texts = [m.text for m in sent if isinstance(m, SendMessage)]
    assert any("Синусовый ритм" in x for x in texts) and not any("📥" in x for x in texts)
    assert sent[-1].reply_markup is not None                                      # the buttons under the reply
