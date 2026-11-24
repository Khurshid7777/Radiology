"""SD-142 (the core's D349): ``/app`` opens the whole site as a Mini App; the chat's menu button does the same."""

import pytest

pytest.importorskip("aiogram")

from aiogram.methods import SendMessage, SetChatMenuButton
from aiogram.types import MenuButtonWebApp
from test_bot_group import ADMINS, GROUP, FakeApi, Silent, cfg, run, text_update

from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import PUBLIC_URL_ENV, site_url
from ecgscan_bot.texts import tr

ADMIN = min(ADMINS)


def of(sent, kind):
    return [m for m in sent if isinstance(m, kind)]


def test_app_sends_one_web_app_button_to_the_sites_root_and_sets_the_menu_button(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = run(cfg(public_url="https://site.example/"), FakeApi(), AccessStore(tmp_path / "a.json"),
               text_update(ADMIN, "private", ADMIN, "/app"))
    (msg,) = of(sent, SendMessage)
    ((button,),) = msg.reply_markup.inline_keyboard
    assert button.web_app.url == "https://site.example/" and button.url is None and button.callback_data is None
    assert button.text == tr("app_btn", "ru") and str(ADMIN) not in button.web_app.url
    (menu,) = of(sent, SetChatMenuButton)
    assert menu.chat_id == ADMIN and isinstance(menu.menu_button, MenuButtonWebApp)
    assert menu.menu_button.web_app.url == "https://site.example/" and menu.menu_button.text == tr("app_btn", "ru")


def test_start_sets_the_menu_button_for_one_let_in_only(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    c = cfg(public_url="https://site.example")
    sent = run(c, FakeApi(), AccessStore(tmp_path / "a.json"), text_update(ADMIN, "private", ADMIN, "/start"))
    assert len(of(sent, SendMessage)) == 1 and len(of(sent, SetChatMenuButton)) == 1
    stranger = run(c, FakeApi(), AccessStore(tmp_path / "b.json"), text_update(999, "private", 999, "/start"))
    assert not of(stranger, SetChatMenuButton)


def test_app_follows_the_deployments_address(monkeypatch):
    monkeypatch.setenv(PUBLIC_URL_ENV, "https://deployed.example/")
    assert site_url(cfg(public_url="https://site.example")) == "https://deployed.example/"


@pytest.mark.parametrize("public", [None, "", "http://plain.example"])
def test_app_without_an_https_address_says_so_and_sets_nothing(tmp_path, monkeypatch, public):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = run(cfg(public_url=public), FakeApi(), AccessStore(tmp_path / "a.json"), text_update(ADMIN, "private", ADMIN, "/app"))
    (msg,) = of(sent, SendMessage)
    assert msg.reply_markup is None and msg.text == tr("app_off", "ru") and not of(sent, SetChatMenuButton)


def test_app_is_refused_to_a_stranger_and_silent_in_a_group(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    c = cfg(public_url="https://site.example")
    stranger = run(c, FakeApi(), AccessStore(tmp_path / "a.json"), text_update(999, "private", 999, "/app"))
    (msg,) = of(stranger, SendMessage)
    assert msg.reply_markup is None and "/code" in msg.text and not of(stranger, SetChatMenuButton)
    assert not run(c, FakeApi(), AccessStore(tmp_path / "b.json"), text_update(GROUP, "supergroup", ADMIN, "/app"))


def test_a_failed_menu_button_does_not_break_the_answer(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    real = Silent.make_request

    async def refuse(self, bot, method, timeout=None):
        if isinstance(method, SetChatMenuButton):
            raise ConnectionError("Telegram said no")
        return await real(self, bot, method, timeout)

    monkeypatch.setattr(Silent, "make_request", refuse)
    sent = run(cfg(public_url="https://site.example"), FakeApi(), AccessStore(tmp_path / "a.json"),
               text_update(ADMIN, "private", ADMIN, "/app"))
    assert len(of(sent, SendMessage)) == 1


def test_app_texts_exist_in_both_languages():
    for key in ("app", "app_btn", "app_off"):
        assert tr(key, "ru") and tr(key, "uz") and tr(key, "ru") != tr(key, "uz")


def test_one_who_comes_in_by_a_code_gets_the_menu_button_too(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    store = AccessStore(tmp_path / "a.json")
    code = store.new_code("clinic", ADMIN)
    sent = run(cfg(public_url="https://site.example"), FakeApi(), store, text_update(555, "private", 555, f"/start {code}"))
    (menu,) = of(sent, SetChatMenuButton)
    assert menu.chat_id == 555 and menu.menu_button.web_app.url == "https://site.example/"
    bad = run(cfg(public_url="https://site.example"), FakeApi(), AccessStore(tmp_path / "b.json"),
              text_update(556, "private", 556, "/start NOPE"))
    assert not of(bad, SetChatMenuButton)
