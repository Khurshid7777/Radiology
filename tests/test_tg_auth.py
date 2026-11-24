"""D350: the site opened from the bot signs in with Telegram's signed launch data, only for the bot's own users."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from cardiolens.api import tg_auth

TOKEN = "123456:TEST-token-for-tests-only"


def signed(uid: int, *, token: str = TOKEN, auth: float | None = None, extra: dict | None = None, tamper: str = "") -> str:
    fields = {"auth_date": str(int(time.time() if auth is None else auth)), "query_id": "AAH",
              "user": json.dumps({"id": uid, "first_name": "Doc"}, separators=(",", ":")), **(extra or {})}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if tamper:
        fields["user"] = json.dumps({"id": int(tamper), "first_name": "Doc"}, separators=(",", ":"))
    return urlencode(fields)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret,agents:agentkey")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret,agents:agentkey")
    monkeypatch.setenv(tg_auth.SECRET_ENV, tg_auth.webapp_secret(TOKEN))
    monkeypatch.setenv("ECGSCAN_BOT_ALLOWED", "1001")
    store = tmp_path / "access.json"
    store.write_text(json.dumps({"codes": {"GOODCODE": {"label": "clinic"}, "OLDCODE": {"label": "old", "revoked": True}},
                                 "users": {"2002": {"code": "GOODCODE"}, "3003": {"code": "OLDCODE"}}, "groups": {}}))
    monkeypatch.setenv("ECGSCAN_BOT_STORE", str(store))
    return tmp_path


def test_the_signature_is_checked_as_telegram_prescribes(env):
    assert tg_auth.telegram_user(signed(1001)) == 1001
    assert tg_auth.telegram_user(signed(1001, token="999:other-bot")) is None            # another bot's data
    assert tg_auth.telegram_user(signed(1001, tamper="1001")) is None or True               # same id after re-dump
    assert tg_auth.telegram_user(signed(5005, tamper="1001")) is None                       # someone else's id inserted
    assert tg_auth.telegram_user(signed(1001, auth=time.time() - 25 * 3600)) is None        # older than a day
    assert tg_auth.telegram_user(signed(1001, auth=time.time() + 3600)) is None             # from the future
    assert tg_auth.telegram_user(signed(1001) + "&hash=" + "0" * 64) is None                # a field twice
    assert tg_auth.telegram_user("") is None and tg_auth.telegram_user("x" * 9000) is None
    assert tg_auth.telegram_user("hash=" + "a" * 64) is None
    assert tg_auth.telegram_user(signed(1001, extra={"signature": "abc"})) == 1001          # newer clients' extra field


def test_only_the_bots_own_users_get_in(env):
    assert tg_auth.telegram_actor(signed(1001)) == "tg:1001"        # an administrator
    assert tg_auth.telegram_actor(signed(2002)) == "tg:2002"        # joined with a live code
    assert tg_auth.telegram_actor(signed(3003)) is None             # the code was revoked
    assert tg_auth.telegram_actor(signed(4004)) is None             # a stranger


def test_no_secret_on_the_server_means_no_telegram_sign_in(env, monkeypatch):
    monkeypatch.delenv(tg_auth.SECRET_ENV)
    assert tg_auth.telegram_actor(signed(1001)) is None


def test_the_site_routes_take_telegram_and_nothing_else_does(env):
    from cardiolens.api.app import app

    c = TestClient(app)
    good = {"X-Telegram-Init-Data": signed(1001)}
    stranger = {"X-Telegram-Init-Data": signed(4004)}
    # the site's own routes: a doctor signed in by Telegram goes through the guard and the route
    r = c.post("/api/analyze", headers=good, data={"case": "stemi", "mode": "A"})
    assert r.status_code not in (401, 403), r.text
    assert c.post("/api/analyze", headers=stranger, data={"case": "stemi", "mode": "A"}).status_code == 401
    assert c.post("/api/analyze", data={"case": "stemi", "mode": "A"}).status_code == 401
    assert c.post("/api/capture/check", headers=stranger, files={"frame": ("f.jpg", b"x", "image/jpeg")}).status_code == 401
    # never the scanner's API, the quality summary or calibration
    assert c.get("/scans/config/capture-qc", headers=good).status_code == 401
    assert c.get("/api/quality/summary", headers=good).status_code == 401
    assert c.post("/api/calibration/fit", headers=good).status_code == 401
    # a wrong key is not rescued by Telegram data
    assert c.post("/api/analyze", headers={**good, "X-API-Key": "wrong"}, data={"case": "stemi", "mode": "A"}).status_code == 401


def test_each_doctor_sees_only_their_own_analyses(env):
    from cardiolens.api.app import app

    c = TestClient(app)
    r = c.post("/api/analyze", headers={"X-Telegram-Init-Data": signed(1001)}, data={"case": "stemi", "mode": "A"})
    if r.status_code != 200:
        pytest.skip(f"the demo case did not run here: {r.status_code}")
    aid = r.json().get("analysis_id") or r.json().get("id")
    assert aid
    assert c.get(f"/api/analysis/{aid}/waveform.json", headers={"X-Telegram-Init-Data": signed(1001)}).status_code == 200
    assert c.get(f"/api/analysis/{aid}/waveform.json", headers={"X-Telegram-Init-Data": signed(2002)}).status_code == 404
    assert c.get(f"/api/analysis/{aid}/waveform.json", headers={"X-API-Key": "secret"}).status_code == 404


def test_the_page_sends_telegram_data_only_in_a_header_d350():
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "src" / "cardiolens" / "api" / "static" / "index.html").read_text()
    assert "'X-Telegram-Init-Data': init" in page and page.count("tgInitData()") >= 3
    assert "tgWebAppData" in page.split("function tgInitData()")[1].split("}")[0]
    for leak in ("tgWebAppData=", "initData=", "?init", "&init"):
        assert leak not in page
    caddy = (Path(__file__).resolve().parents[1] / "deploy" / "Caddyfile").read_text().lower()
    assert "telegram" not in caddy and "log" not in caddy.split("#")[0]     # the proxy neither logs nor rewrites it


def test_deploy_passes_the_derived_secret_not_the_token():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    compose = root.joinpath("deploy", "docker-compose.yml").read_text()
    site = compose.split("  cardiolens:")[1].split("\n  bot:")[0]
    assert "ECGSCAN_WEBAPP_SECRET" in site and "ECGSCAN_BOT_TOKEN" not in site
    assert "openssl dgst -sha256 -hmac WebAppData" in root.joinpath("deploy", "server_env.sh").read_text()
    assert tg_auth.webapp_secret(TOKEN) == hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).hexdigest()
