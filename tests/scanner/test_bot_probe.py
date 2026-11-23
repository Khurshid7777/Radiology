"""SD-136: ``/probe`` opens the phone probe page as a Mini App, in a private chat, for those let in."""

import pytest

pytest.importorskip("aiogram")

from aiogram.methods import SendMessage
from test_bot_group import ADMINS, GROUP, FakeApi, cfg, run, text_update

from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import PUBLIC_URL_ENV, miniapp_url

ADMIN = min(ADMINS)


def messages(sent):
    return [m for m in sent if isinstance(m, SendMessage)]


def test_probe_sends_one_web_app_button_to_the_site_page(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = messages(run(cfg(public_url="https://site.example/"), FakeApi(), AccessStore(tmp_path / "a.json"),
                        text_update(ADMIN, "private", ADMIN, "/probe")))
    assert len(sent) == 1 and sent[0].chat_id == ADMIN
    (row,) = sent[0].reply_markup.inline_keyboard
    (button,) = row
    assert button.web_app.url == "https://site.example/miniapp/probe"
    assert button.url is None and button.callback_data is None
    assert "ничего не отправляет" in sent[0].text


def test_probe_follows_the_deployments_public_address(tmp_path, monkeypatch):
    monkeypatch.setenv(PUBLIC_URL_ENV, "https://deployed.example")
    assert miniapp_url(cfg(public_url="https://site.example"), "probe") == "https://deployed.example/miniapp/probe"


@pytest.mark.parametrize("public", [None, "", "http://plain.example"])
def test_probe_without_an_https_address_says_so_and_sends_no_button(tmp_path, monkeypatch, public):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    sent = messages(run(cfg(public_url=public), FakeApi(), AccessStore(tmp_path / "a.json"),
                        text_update(ADMIN, "private", ADMIN, "/probe")))
    assert len(sent) == 1 and sent[0].reply_markup is None and "не настроена" in sent[0].text


def test_probe_is_refused_to_a_stranger_and_silent_in_a_group(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_URL_ENV, raising=False)
    c = cfg(public_url="https://site.example")
    stranger = messages(run(c, FakeApi(), AccessStore(tmp_path / "a.json"), text_update(999, "private", 999, "/probe")))
    assert len(stranger) == 1 and stranger[0].reply_markup is None and "/code" in stranger[0].text
    # In a group the bot writes nothing at all (SD-107), and a web_app button would not work there anyway.
    assert not messages(run(c, FakeApi(), AccessStore(tmp_path / "b.json"), text_update(GROUP, "supergroup", ADMIN, "/probe")))


def test_probe_texts_exist_in_both_languages():
    from ecgscan_bot.texts import tr

    for key in ("probe", "probe_btn", "probe_off"):
        assert tr(key, "ru") and tr(key, "uz") and tr(key, "ru") != tr(key, "uz")
