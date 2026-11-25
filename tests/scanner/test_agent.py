"""SD-152: the desk agent: its key, the server's queue, the program on the laptop, the bot's side, end to end."""

import asyncio
import re
import time

import pytest

from ecgscan_agent.core import Agent, Memory, Settings
from ecgscan_service.agent import (
    AgentKey,
    AgentQueue,
    QueueFull,
    issue_agent_key,
    kind_of,
    new_agent_id,
    verify_agent_key,
)
from ecgscan_service.miniapp import issue_ticket, verify_ticket

KEYS = {"k-bot": "bot", "k-agents": "agents"}
PDF = b"%PDF-1.4\n" + b"x" * 300


# --- the key -------------------------------------------------------------------------------------------

def test_an_agent_key_is_its_own_kind_of_key():
    aid = new_agent_id()
    key, expiry = issue_agent_key("k-bot", aid, 180, now=1_800_000_000)
    assert verify_agent_key(key, KEYS, max_days=400, now=1_800_000_000) == AgentKey(aid, "bot", expiry)
    assert verify_agent_key(key, KEYS, max_days=100, now=1_800_000_000) is None        # too long a life
    assert verify_agent_key(key, KEYS, max_days=400, now=expiry) is None                # run out
    assert verify_agent_key(key, {"k-other": "other"}, max_days=400, now=1_800_000_000) is None
    agents_key, _ = issue_agent_key("k-agents", aid, 10, now=1_800_000_000)
    assert verify_agent_key(agents_key, KEYS, max_days=400, now=1_800_000_000) is None   # a read-only key signs nothing
    ticket, _, _ = issue_ticket("k-bot", 5)
    assert verify_agent_key(ticket, KEYS, max_days=400) is None                          # a ticket is no agent key
    assert verify_ticket(issue_agent_key("k-bot", "a" * 24, 1)[0], KEYS, max_minutes=10) is None
    for bad in ("", None, "x.y.z", key + "0", key.upper()):
        assert verify_agent_key(bad, KEYS, max_days=400, now=1_800_000_000) is None


def test_kinds_by_first_bytes():
    allowed = ["pdf", "xml", "dicom", "jpeg", "png", "tiff"]
    assert kind_of(PDF, allowed) == "pdf" and kind_of(b"\xef\xbb\xbf<?xml ...", allowed) == "xml"
    assert kind_of(b"\x00" * 128 + b"DICM", allowed) == "dicom" and kind_of(b"\xff\xd8\xff\xe0", allowed) == "jpeg"
    assert kind_of(b"MZ\x90\x00", allowed) is None and kind_of(PDF, ["xml"]) is None


def test_the_queue_keeps_records_in_order_for_their_owner_and_within_limits():
    q = AgentQueue(max_items=3, max_bytes=1000, per_agent=2, keep_s=60)
    a, b = AgentKey("a" * 16, "bot", 0), AgentKey("b" * 16, "bot", 0)
    q.put(a, "pdf", b"1", now=0)
    q.put(b, "pdf", b"2", now=1)
    q.put(a, "xml", b"3", now=2)
    with pytest.raises(QueueFull):
        q.put(a, "pdf", b"4", now=3)                                     # two from one agent at most
    assert not q.room(b, 10, now=3)                                        # three in all
    assert q.take("other", now=3) is None
    assert q.take("bot", now=3) == ("a" * 16, "pdf", b"1") and q.take("bot", now=3) == ("b" * 16, "pdf", b"2")
    assert q.take("bot", now=100) is None                                  # forgotten in time


# --- the program on the laptop ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def agent(tmp_path, sender, exts=None):
    st = Settings(server="https://site.example", key="a.b.c", folders=[str(tmp_path / "in")], poll_s=0, stable_s=2,
                  **({"exts": exts} if exts else {}))
    clock = Clock()
    return Agent(st, Memory(tmp_path / "home" / "sent.json"), sender=sender, clock=clock), clock


def test_a_new_record_is_sent_once_when_it_stops_growing_and_old_ones_are_not(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "old.pdf").write_bytes(PDF + b"old")
    sent = []
    ag, clock = agent(tmp_path, lambda s, d: sent.append(d) or 200)
    assert ag.baseline() == 1
    new = tmp_path / "in" / "sub" / "Ivanov 2026.pdf"
    new.parent.mkdir()
    new.write_bytes(PDF[:100])
    assert ag.round()["sent"] == 0                                         # just appeared: not yet
    new.write_bytes(PDF)                                                   # still being written
    clock.t = 1
    assert ag.round()["sent"] == 0
    clock.t = 4
    assert ag.round()["sent"] == 1 and sent == [PDF]
    clock.t = 10
    assert ag.round()["sent"] == 0                                         # once
    again, c2 = agent(tmp_path, lambda s, d: sent.append(d) or 200)        # a restart: the memory holds
    again.round()
    c2.t = 5
    assert again.round()["sent"] == 0 and len(sent) == 1


def test_no_connection_is_tried_again_and_a_refusal_is_not(tmp_path):
    (tmp_path / "in").mkdir()
    answers = [0, 503, 200]
    sent = []

    def flaky(s, d):
        sent.append(d)
        return answers.pop(0)

    ag, clock = agent(tmp_path, flaky)
    (tmp_path / "in" / "a.pdf").write_bytes(PDF)
    ag.round()
    for t in (3, 6, 9, 12):
        clock.t = t
        ag.round()
    assert len(sent) == 3                                                  # 0, 503, then sent; never again
    refused = []
    ag2, clock2 = agent(tmp_path, lambda s, d: refused.append(d) or 415)
    (tmp_path / "in" / "b.xml").write_bytes(b"<?xml version='1.0'?><x/>")
    ag2.round()
    for t in (3, 6, 9):
        clock2.t = t
        ag2.round()
    assert len(refused) == 1                                               # a refusal is final for that file


def test_only_the_chosen_kinds_of_file_are_looked_at(tmp_path):
    (tmp_path / "in").mkdir()
    sent = []
    ag, clock = agent(tmp_path, lambda s, d: sent.append(d) or 200, exts=[".xml"])
    (tmp_path / "in" / "a.pdf").write_bytes(PDF)
    (tmp_path / "in" / "b.XML").write_bytes(b"<x/>")
    (tmp_path / "in" / "c.exe").write_bytes(b"MZ")
    ag.round()
    clock.t = 5
    ag.round()
    assert sent == [b"<x/>"]


def test_settings_are_checked_and_the_log_has_no_names(tmp_path, caplog):
    import logging

    with pytest.raises(ValueError):
        Settings(server="http://site.example", key="a.b.c", folders=["x"]).check()
    with pytest.raises(ValueError):
        Settings(server="https://site.example", key="nokey", folders=["x"]).check()
    caplog.set_level(logging.DEBUG)
    (tmp_path / "in").mkdir()
    ag, clock = agent(tmp_path, lambda s, d: 200)
    (tmp_path / "in" / "Ivanov Ivan 1956.pdf").write_bytes(PDF)
    ag.round()
    clock.t = 5
    ag.round()
    assert "Ivanov" not in caplog.text and "%PDF" not in caplog.text


# --- the server ----------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from ecgscan_service import router
    from ecgscan_service.app import app

    monkeypatch.setenv("ECGSCAN_API_KEYS", "bot:k-bot,other:k-other")
    monkeypatch.setenv("ECGSCAN_SERVICE_DIR", str(tmp_path / "service"))
    router.get_agent_queue.cache_clear()
    yield TestClient(app)
    router.get_agent_queue.cache_clear()


def test_the_server_takes_a_record_with_an_agent_key_and_gives_it_to_the_bot_once(client):
    key, _ = issue_agent_key("k-bot", "c" * 16, 180)
    r = client.post("/miniapp/agent/upload", content=PDF, headers={"X-Agent-Key": key})
    assert r.status_code == 200 and r.json() == {"queued": True, "kind": "pdf"}
    assert client.get("/scans/agent/inbox", headers={"X-API-Key": "k-other"}).status_code == 204
    got = client.get("/scans/agent/inbox", headers={"X-API-Key": "k-bot"})
    assert got.status_code == 200 and got.content == PDF
    assert got.headers["x-agent-id"] == "c" * 16 and got.headers["x-agent-kind"] == "pdf"
    assert client.get("/scans/agent/inbox", headers={"X-API-Key": "k-bot"}).status_code == 204
    assert client.get("/scans/agent/inbox").status_code == 401


@pytest.mark.parametrize("headers", [{}, {"X-Agent-Key": "x.y.z"}, {"X-Agent-Key": issue_ticket("k-bot", 5)[0]}])
def test_without_a_good_agent_key_nothing_is_taken(client, headers):
    assert client.post("/miniapp/agent/upload", content=PDF, headers=headers).status_code == 401


def test_what_is_not_a_record_is_refused(client):
    key, _ = issue_agent_key("k-bot", "d" * 16, 180)
    assert client.post("/miniapp/agent/upload", content=b"MZ\x90\x00" + b"\x00" * 300,
                       headers={"X-Agent-Key": key}).status_code == 415
    assert client.get("/scans/agent/inbox", headers={"X-API-Key": "k-bot"}).status_code == 204


def test_a_bad_key_is_refused_before_the_body_is_read():
    from ecgscan_service.app import app

    sent = []

    async def receive():
        raise AssertionError("the body was read")

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "https",
             "path": "/miniapp/agent/upload", "raw_path": b"/miniapp/agent/upload", "query_string": b"", "root_path": "",
             "headers": [(b"x-agent-key", b"forged"), (b"content-length", b"1000")],
             "client": ("203.0.113.1", 1), "server": ("site", 443)}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 401


# --- the bot -------------------------------------------------------------------------------------------

pytest.importorskip("aiogram")


def test_agent_gives_a_key_that_the_server_takes_and_off_stops_it(tmp_path, monkeypatch):
    from aiogram.methods import SendMessage
    from test_bot_group import ADMINS, GROUP, FakeApi, cfg, run, text_update

    from ecgscan_bot.access import AccessStore

    admin = min(ADMINS)
    store = AccessStore(tmp_path / "a.json")
    c = cfg(api_key="k-bot")
    sent = [m for m in run(c, FakeApi(), store, text_update(admin, "private", admin, "/agent")) if isinstance(m, SendMessage)]
    assert len(sent) == 1 and sent[0].parse_mode == "HTML"
    key = re.search(r"<code>([0-9a-f.]+)</code>", sent[0].text).group(1)
    k = verify_agent_key(key, KEYS, max_days=400)
    assert k is not None and store.agent_chat(k.id) == admin                     # no Telegram id in the key: by design
    off = [m for m in run(c, FakeApi(), store, text_update(admin, "private", admin, "/agent off")) if isinstance(m, SendMessage)]
    assert "1" in off[0].text and store.agent_chat(k.id) is None
    stranger = [m for m in run(c, FakeApi(), AccessStore(tmp_path / "b.json"), text_update(999, "private", 999, "/agent"))
                if isinstance(m, SendMessage)]
    assert "/code" in stranger[0].text and "<code>" not in stranger[0].text
    assert not run(c, FakeApi(), AccessStore(tmp_path / "c.json"), text_update(GROUP, "supergroup", admin, "/agent"))


def test_end_to_end_a_file_in_the_folder_reaches_the_doctors_chat(tmp_path, monkeypatch, client):
    """folder -> agent -> server -> bot -> the doctor's chat: «a record from the device», the patient question,
    and the analysis of the very bytes."""
    from aiogram import Bot
    from aiogram.methods import SendMessage
    from test_bot_group import ADMINS, FakeApi, Silent, cfg

    from ecgscan.i18n import t  # noqa: F401
    from ecgscan_bot.__main__ import build
    from ecgscan_bot.access import AccessStore
    from ecgscan_bot.texts import tr

    admin = min(ADMINS)
    store = AccessStore(tmp_path / "a.json")
    aid = new_agent_id()
    store.add_agent(admin, aid)
    key, _ = issue_agent_key("k-bot", aid, 180)
    (tmp_path / "in").mkdir()

    def to_server(s, data):
        return client.post("/miniapp/agent/upload", content=data, headers={"X-Agent-Key": s.key}).status_code

    st = Settings(server="https://site.example", key=key, folders=[str(tmp_path / "in")], poll_s=0, stable_s=0)
    ag = Agent(st, Memory(tmp_path / "home" / "sent.json"), sender=to_server, clock=time.monotonic)
    (tmp_path / "in" / "record.pdf").write_bytes(PDF)
    ag.round()
    assert ag.round()["sent"] == 1

    class Api(FakeApi):
        def __init__(self):
            super().__init__()
            self.http, self.headers, self.files = client, {"X-API-Key": "k-bot"}, None

        def analyze(self, files, **kw):
            self.files = files
            return super().analyze(files, **kw)

    api = Api()
    session = Silent()
    bot = Bot("42:TEST", session=session)

    async def go():
        dp = build(cfg(api_key="k-bot"), bot, api, store)
        assert await dp["agent_pull"]() is True
        assert await dp["agent_pull"]() is False

    asyncio.run(go())
    texts = [m.text for m in session.sent if isinstance(m, SendMessage)]
    assert texts[0] == tr("agent_got", "ru") and texts[1] == tr("ask_patient", "ru")
    assert all(m.chat_id == admin for m in session.sent if isinstance(m, SendMessage))


def test_a_record_from_a_revoked_agent_is_dropped(tmp_path, client):
    from aiogram import Bot
    from aiogram.methods import SendMessage
    from test_bot_group import ADMINS, FakeApi, Silent, cfg

    from ecgscan_bot.__main__ import build
    from ecgscan_bot.access import AccessStore

    admin = min(ADMINS)
    store = AccessStore(tmp_path / "a.json")
    aid = new_agent_id()
    store.add_agent(admin, aid)
    store.revoke_agents(admin)
    key, _ = issue_agent_key("k-bot", aid, 180)
    assert client.post("/miniapp/agent/upload", content=PDF, headers={"X-Agent-Key": key}).status_code == 200
    api = FakeApi()
    api.http, api.headers = client, {"X-API-Key": "k-bot"}
    session = Silent()

    async def go():
        dp = build(cfg(api_key="k-bot"), Bot("42:TEST", session=session), api, store)
        assert await dp["agent_pull"]() is True

    asyncio.run(go())
    assert not [m for m in session.sent if isinstance(m, SendMessage)] and api.calls == 0
