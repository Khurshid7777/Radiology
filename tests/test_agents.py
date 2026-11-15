"""D279: the agents bot — only the owner commands, topics route to agents, secrets never leave, isolation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cardiovision_agents import logic, runner
from cardiovision_agents.logic import (
    ROLES,
    is_approval,
    is_owner,
    redact,
    role_for_thread,
    split_message,
)
from cardiovision_agents.store import Store

ROOT = Path(__file__).resolve().parents[1]


def test_only_the_owner_commands():
    assert is_owner(42, 42)
    assert not is_owner(43, 42)
    assert not is_owner(None, 42)
    assert not is_owner(42, None) and not is_owner(42, 0)      # no owner configured -> nobody


def test_topics_route_to_their_agent_and_nothing_else():
    topics = {"dev": 11, "sec": 12}
    assert role_for_thread(topics, 11).key == "dev"
    assert role_for_thread(topics, 12).key == "sec"
    assert role_for_thread(topics, 99) is None
    assert role_for_thread(topics, None) is None
    assert role_for_thread({"bogus": 5}, 5) is None


def test_approval_is_a_plain_yes_only():
    for t in ("да", "Да!", "yes", "да, выкладывай", " выкладывай. "):
        assert is_approval(t), t
    for t in ("да, но сначала поправь тест", "нет", "не выкладывай", "", None, "дай подумать"):
        assert not is_approval(t), t


def test_stop_and_start_words():
    assert logic.STOP.match("стоп агенты") and logic.STOP.match("Стоп агенты!")
    assert logic.START.match("старт агенты") and not logic.STOP.match("стоп")


def test_tokens_are_redacted_before_they_leave():
    tg = "1234567890:AAH" + "x" * 32
    gh = "ghp_" + "A" * 36
    pat = "github_pat_" + "B" * 40
    ant = "sk-ant-" + "c" * 40
    text = f"token {tg} and {gh} and {pat} and {ant} and secret-value-123"
    out = redact(text, ("secret-value-123",))
    for s in (tg, gh, pat, ant, "secret-value-123"):
        assert s not in out
    assert out.count("[скрыто]") == 5


def test_long_answers_are_split_for_telegram():
    parts = split_message("строка\n" * 2000, limit=4000)
    assert all(len(p) <= 4000 for p in parts) and len(parts) >= 3
    assert split_message("") == ["(пустой ответ)"]


def test_developer_branch_is_never_main():
    assert logic.branch_name("20261115-1030") == "agents/dev-20261115-1030"
    with pytest.raises(ValueError):
        logic.branch_name("main")


def test_only_the_developer_edits_and_works_on_a_branch():
    for key, role in ROLES.items():
        assert role.branch == (key == "dev")
        if key != "dev":
            assert "Edit" not in role.tools
        if key == "sec":
            assert "Bash" not in role.tools and "Write" not in role.tools
        assert (ROOT / "docs/agents" / role.doc).exists(), role.doc


def test_the_agent_runs_unprivileged_with_a_clean_environment(monkeypatch):
    monkeypatch.setenv("AGENTS_TG_TOKEN", "1234567890:AAH" + "x" * 32)
    cmd = runner.command(ROLES["dev"], "hi", None)
    assert cmd[0] == "setpriv" and "--reuid=agent" in cmd and "claude" in cmd
    assert "--resume" not in cmd and "--resume" in runner.command(ROLES["dev"], "hi", "s1")
    assert set(runner.CLEAN_ENV) <= {"HOME", "PATH", "LANG", "OMP_NUM_THREADS"}
    assert not any("TOKEN" in k or "KEY" in k for k in runner.CLEAN_ENV)


def test_prompt_names_the_instructions_and_the_branch_rules():
    p = logic.build_prompt(ROLES["dev"], "почини X")
    assert "developer.md" in p and "do not push" in p and "почини X" in p
    assert "do not push" not in logic.build_prompt(ROLES["sec"], "отчёт")


def test_store_keeps_state_private(tmp_path):
    s = Store(tmp_path / "a.json")
    s["chat_id"] = -100123
    assert Store(tmp_path / "a.json")["chat_id"] == -100123
    assert (tmp_path / "a.json").stat().st_mode & 0o077 == 0


def test_the_agents_container_is_isolated_from_the_site():
    comp = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())
    ag = comp["services"]["agents"]
    vols = " ".join(ag.get("volumes", []))
    assert "appdata" not in vols and "docker.sock" not in vols and ".env" not in vols
    assert "/security:ro" in vols
    assert ag["profiles"] == ["agents"]
    env = ag["environment"]
    assert "ECGSCAN_API_KEYS" not in env and "ECGSCAN_STORAGE_KEY" not in env and "ANTHROPIC_API_KEY" not in env
    # a missing agents token must never stop the site from starting (no ":?" for the agents' variables)
    assert ":?" not in str(env)


def test_group_messages_count_only_from_the_owner_in_the_bound_group():
    assert logic.accepts(42, -100, 42, -100)
    assert not logic.accepts(43, -100, 42, -100)        # another member: ignored, nothing starts
    assert not logic.accepts(42, -200, 42, -100)        # another group: ignored
    assert not logic.accepts(42, -100, 42, None)        # not bound yet: only /setup works


def test_the_bot_module_wires_the_owner_check():
    src = (ROOT / "src/cardiovision_agents/bot.py").read_text()
    assert "accepts(msg.from_user" in src
    assert src.count("is_owner(msg.from_user") >= 3      # /setup, /unbind, private chat


def test_security_check_writes_no_secret_values():
    sh = (ROOT / "deploy/security_check.sh").read_text()
    assert "sed -n 's/=.*//p'" in sh            # only the variable names of deploy/.env
    assert "cat " not in sh.replace("# ", "")
