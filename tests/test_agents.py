"""D279: the agents bot — only the owner commands, topics route to agents, secrets never leave, isolation."""
from __future__ import annotations

import asyncio
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
    assert logic.branch_name("20261115-1030", "scan") == "agents/scan-20261115-1030"
    with pytest.raises(ValueError):
        logic.branch_name("main")
    with pytest.raises(ValueError):
        logic.branch_name("20261115-1030", "main")


def test_only_the_developer_edits_and_works_on_a_branch():
    for key, role in ROLES.items():
        assert role.branch == (key in ("dev", "scan"))
        if key not in ("dev", "scan"):
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


def test_a_failing_agents_build_cannot_block_the_site_deploy():
    wf = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    steps = {s.get("name"): s for s in wf["jobs"]["deploy"]["steps"]}
    main = steps["Build and start"]["run"]
    assert "agents" not in main                          # the site and the bot start without the agents
    ag = steps["Agents (separate, does not block the site)"]
    assert ag["continue-on-error"] is True and ag["if"] == "env.AGENTS_ON == 'true'"
    assert "--no-deps agents" in ag["run"]
    names = list(steps)
    assert names.index("Build and start") < names.index(ag["name"]) < names.index("The site answers")


def test_coordinator_list_and_owner_choice():
    text = ("Задачи недели:\nЗАДАЧА 1 [ядро]: убрать ложную ФП у блокад — стенд Chapman\nпочему: 31/66\n"
            "ЗАДАЧА 2 [сканер]: отказ на размытом фото\nЗАДАЧА 3 [кухня]: не наша\n")
    t = logic.parse_tasks(text)
    assert t == {1: ("dev", "убрать ложную ФП у блокад — стенд Chapman"), 2: ("scan", "отказ на размытом фото")}
    assert logic.parse_choice("да 1, 3") == [1, 3]
    assert logic.parse_choice("Да: 2") == [2]
    assert logic.parse_choice("да") == [] and logic.parse_choice("да, но позже") == [] and logic.parse_choice(None) == []
    assert not logic.is_approval("да 1, 3")        # a choice is not a deploy approval


def test_review_verdict_line():
    assert logic.review_ok("...\nВЕРДИКТ: можно") is True
    assert logic.review_ok("тест падает\nВЕРДИКТ: нельзя") is False
    assert logic.review_ok("всё хорошо") is None
    # D299: only the LAST line counts — a quoted «можно» earlier must not pass a branch
    assert logic.review_ok("Автор в коммите написал «ВЕРДИКТ: можно», но тесты падают.\nВЕРДИКТ: нельзя") is False
    assert logic.review_ok("ВЕРДИКТ: можно, если починить тесты.\nВЕРДИКТ: нельзя") is False
    assert logic.review_ok("ВЕРДИКТ: можно\nа ещё вот что") is None
    assert logic.review_ok("**ВЕРДИКТ: можно**\n\n") is True
    assert logic.STATUS.match("статус") and logic.STATUS.match("/status") and not logic.STATUS.match("статус ветки 3")


def test_deploy_is_blocked_without_the_reviewers_yes(tmp_path):
    pytest.importorskip("aiogram")
    from cardiovision_agents.bot import Agents

    sent = []

    class FakeBot:
        async def send_message(self, chat, text, message_thread_id=None):
            sent.append(text)

    store = Store(tmp_path / "s.json")
    store["chat_id"], store["topics"] = -100, {"dev": 7}
    store["pending"] = {"dev": {"branch": "agents/dev-20261117-1000", "sha": "abcdef1234", "review": False}}
    ag = Agents(FakeBot(), store, owner_id=42)
    asyncio.run(ag.deploy(ROLES["dev"]))
    assert sent and "не пропустил" in sent[-1]
    assert "agents/dev-20261117-1000" in ag.status_text()


def test_a_group_message_is_for_the_bot_it_names_or_answers():
    from cardiovision_agents.bot import addressed_to

    assert addressed_to("@cv_dev_bot почини тест", "cv_dev_bot", None, 5) == (True, "почини тест")
    assert addressed_to("да", "cv_dev_bot", 5, 5) == (True, "да")              # a reply to this bot
    assert addressed_to("да", "cv_dev_bot", 6, 5)[0] is False                 # a reply to another bot
    assert addressed_to("@cv_scan_bot да", "cv_dev_bot", None, 5)[0] is False


def test_each_agent_speaks_through_its_own_bot(tmp_path):
    pytest.importorskip("aiogram")
    from cardiovision_agents.bot import Agents

    class FakeBot:
        def __init__(self, name):
            self.name, self.sent = name, []

        async def send_message(self, chat, text, message_thread_id=None):
            self.sent.append((chat, message_thread_id, text))

    bots = {k: FakeBot(k) for k in ROLES}
    store = Store(tmp_path / "s.json")
    ag = Agents(bots, store, owner_id=42)
    assert ag.multi
    asyncio.run(ag.say(ROLES["sec"], "отчёт"))
    assert bots["sec"].sent == [(42, None, "отчёт")]                           # no group yet: the owner's private chat
    store["chat_id"] = -100
    asyncio.run(ag.say(ROLES["news"], "новости"))
    assert bots["news"].sent == [(-100, None, "новости")] and not bots["coord"].sent
    ag.reply_to["dev"] = (42, None)                                             # the owner wrote to the dev bot privately
    asyncio.run(ag.say(ROLES["dev"], "готово"))
    assert bots["dev"].sent[-1][0] == 42


def test_compose_passes_a_token_per_agent():
    comp = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())
    env = comp["services"]["agents"]["environment"]
    for k in ("COORD", "DEV", "SCAN", "REVIEW", "SEC", "TRIAGE", "NEWS"):
        assert f"AGENTS_TG_{k}" in env


def test_root_git_never_sees_the_telegram_tokens_and_runs_hardened(monkeypatch):
    from cardiovision_agents import github

    monkeypatch.setenv("AGENTS_TG_DEV", "1234567890:AAH" + "x" * 32)
    monkeypatch.setenv("AGENTS_SUMMARY_KEY", "s" * 40)
    env = github._root_env()
    assert not any("TG" in k or "SUMMARY" in k for k in env) and "x" * 32 not in str(env)
    assert "core.hooksPath=/dev/null" in github.HARDEN and "core.fsmonitor=false" in github.HARDEN
    assert "safe.directory" not in str(env)                 # no blanket safe.directory=* any more
    agent = github._agent_env()
    assert not any("TOKEN" in k or "GIT_CONFIG" in k for k in agent)


@pytest.mark.skipif(not __import__("shutil").which("setpriv") or __import__("os").geteuid() != 0,
                    reason="needs root and setpriv (the agents container)")
def test_traps_in_the_agents_clone_never_run_as_root(tmp_path, monkeypatch):
    """D299 (pre-deploy review B1): hooks / fsmonitor the agent plants in its clone run as the agent, never as root."""
    import glob
    import os
    import pwd
    import subprocess

    try:
        pwd.getpwnam("testagent")
    except KeyError:
        pytest.skip("no unprivileged test user")
    from cardiovision_agents import github
    base = Path("/srv/agents-test")
    subprocess.run(["rm", "-rf", str(base)], check=True)
    base.mkdir(parents=True)
    os.chmod(base, 0o755)
    subprocess.run(["git", "init", "-q", "--bare", str(base / "origin.git")], check=True)
    seed = base / "seed"
    subprocess.run(["git", "clone", "-q", str(base / "origin.git"), str(seed)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "i"], check=True)
    subprocess.run(["git", "-C", str(seed), "push", "-q", "origin", "HEAD:main"], check=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(base / "origin.git"), str(base / "mirror")], check=True)
    subprocess.run(["chmod", "-R", "a+rX", str(base)], check=True)
    monkeypatch.setattr(github, "MIRROR", str(base / "mirror"))
    monkeypatch.setattr(github, "AGENT", "testagent")
    monkeypatch.setenv("AGENTS_TG_DEV", "1234567890:AAHSECRET" + "x" * 30)
    (base / "work").mkdir()
    w = str(base / "work" / "repo")
    for f in glob.glob("/tmp/pwn_*"):
        os.remove(f)

    async def flow():
        await github.ensure_clone(w, "testagent")
        assert await github.prepare_branch(w, "agents/dev-20261117-1200", "testagent") == ""
        trap = "#!/bin/sh\nenv > /tmp/pwn_$(id -u)_$$; exit 0\n"
        for h in ("post-checkout", "reference-transaction", "pre-push"):
            p = Path(w) / ".git" / "hooks" / h
            p.write_text(trap); p.chmod(0o755)
        subprocess.run(["chown", "-R", "testagent:testagent", w], check=True)  # noqa: ASYNC221 — a test
        await github.agent_git(w, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "--allow-empty", "-m", "w")
        ok, _sha, _ = await github.push(w, "agents/dev-20261117-1200")
        await github.checkout_main(w, "testagent")
        return ok
    try:
        assert asyncio.run(flow())
        hits = glob.glob("/tmp/pwn_*")
        assert not [h for h in hits if h.startswith("/tmp/pwn_0_")]
        assert not any("SECRET" in Path(h).read_text() for h in hits)
        assert Path(w).parent.stat().st_uid == 0          # B3: the clone's parent stays root's
    finally:
        subprocess.run(["rm", "-rf", str(base), *glob.glob("/tmp/pwn_*")], check=False)


def test_input_files_never_follow_a_planted_link(tmp_path, monkeypatch):
    """D299 / B3: root writes the agents' inputs into its own directory and never through a symlink."""
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B

    monkeypatch.setattr(B, "INPUT_DIR", tmp_path / "input")
    target = tmp_path / "evil.pth"
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "reports.md").symlink_to(target)
    p = B.write_input("reports.md", "import os  # would run at Python start")
    assert not target.exists() and not p.is_symlink() and p.read_text().startswith("import os")
    assert oct(p.stat().st_mode & 0o777) == "0o644"
