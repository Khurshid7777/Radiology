"""D279: the agents bot — only the owner commands, topics route to agents, secrets never leave, isolation."""
from __future__ import annotations

import asyncio
import os
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
        if key in ("sec", "audit"):
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
    text = ("Что происходит: всё спокойно.\n1. [ядро] убрать ложную ФП у блокад — стенд Chapman — почему: 31/66\n"
            "2. [сканер] отказ на размытом фото\n3. [кухня] не наша\n**4. [Ядро]** PR по LUDB\n"
            "ЗАДАЧА 5 [сканер]: старый формат\n")
    t = logic.parse_tasks(text)
    assert t == {1: ("dev", "убрать ложную ФП у блокад — стенд Chapman — почему: 31/66"),
                 2: ("scan", "отказ на размытом фото"), 4: ("dev", "PR по LUDB"), 5: ("scan", "старый формат")}
    menu = logic.task_menu(t)
    assert "1. [Программист] убрать ложную ФП" in menu and "2. [Сканер] отказ" in menu and "да 1, 3" in menu
    assert logic.parse_choice("да 1, 3") == [1, 3]
    assert logic.parse_choice("Да: 2") == [2]
    assert logic.parse_choice("да") == [] and logic.parse_choice("да, но позже") == [] and logic.parse_choice(None) == []
    assert not logic.is_approval("да 1, 3")        # a choice is not a deploy approval


def test_reports_morning_four_evening_all_two_minutes_apart():
    """D300 schedule as narrowed by D308: in the morning only security, errors, news and the coordinator."""
    pytest.importorskip("aiogram")
    from cardiovision_agents.bot import SCHEDULE

    morning = [k for k, h, _ in SCHEDULE if h == 9]
    evening = [k for k, h, _ in SCHEDULE if h == 21]
    assert morning == ["sec", "triage", "news", "coord"] and set(evening) == set(ROLES) and evening[-1] == "coord"
    for hour, n in ((9, 4), (21, len(ROLES))):
        assert sorted(m for _, h, m in SCHEDULE if h == hour) == [2 * i for i in range(n)]


def test_agents_work_in_parallel_in_their_own_clones(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    from cardiovision_agents import github, runner

    assert len({runner.workdir_for(r) for r in ROLES.values()}) == len(ROLES)
    running, peak = set(), []

    async def fake_run(role, prompt, session=None, cwd=None, model=None):
        running.add(role.key)
        peak.append(len(running))
        await asyncio.sleep(0.05)
        running.discard(role.key)
        return f"ответ {role.key}", None, {"ok": True, "model": role.model, "cost_usd": 0.1}

    async def nothing(*a, **k):
        return ""

    monkeypatch.setattr(runner, "run", fake_run)
    for name in ("ensure_clone", "checkout_main", "prepare_branch"):
        monkeypatch.setattr(github, name, nothing)
    monkeypatch.setattr(B, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(B, "USAGE_LOG", tmp_path / "usage.jsonl")

    class FakeBot:
        async def send_message(self, *a, **k):
            pass

    ag = B.Agents({k: FakeBot() for k in ROLES}, Store(tmp_path / "s.json"), owner_id=42)

    async def go():
        await asyncio.gather(*(ag.turn(ROLES[k], "отчёт", True) for k in ("sec", "news", "coord")))
    asyncio.run(go())
    assert max(peak) == 3                       # three agents at once
    assert set(ag.store["last_reports"]) == {"sec", "news", "coord"}


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
    for k in ("COORD", "DEV", "SCAN", "REVIEW", "SEC", "TRIAGE", "NEWS", "AUDIT"):
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


def _team(tmp_path, monkeypatch, answers=None, metas=None):
    """An Agents object whose Claude runs are faked; returns (agents, sent messages, calls)."""
    from cardiovision_agents import bot as B
    from cardiovision_agents import github, runner

    sent, calls, models = [], [], []

    async def fake_run(role, prompt, session=None, cwd=None, model=None):
        calls.append((role.key, prompt))
        models.append((role.key, model or runner.model_for(role)))
        await asyncio.sleep(0.01)
        meta = (metas or {}).get(role.key, []).pop(0) if (metas or {}).get(role.key) else {"ok": True, "model": role.model}
        return (answers or {}).get(role.key, f"ответ {role.key}"), None, meta

    async def nothing(*a, **k):
        return ""

    async def no_push(workdir, branch):
        return False, "", "нет коммитов поверх main"

    monkeypatch.setattr(runner, "run", fake_run)
    for name in ("ensure_clone", "checkout_main", "prepare_branch"):
        monkeypatch.setattr(github, name, nothing)
    monkeypatch.setattr(github, "push", no_push)
    monkeypatch.setattr(B, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(B, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(B, "LIMIT_RETRY_MIN", 0)

    class FakeBot:
        def __init__(self, key):
            self.key = key

        async def send_message(self, chat, text, message_thread_id=None):
            sent.append((self.key, text))

    ag = B.Agents({k: FakeBot(k) for k in ROLES}, Store(tmp_path / "s.json"), owner_id=42)
    ag.models_used = models
    return ag, sent, calls


def test_tasks_queue_per_agent_and_report_start_and_end(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    ag, sent, calls = _team(tmp_path, monkeypatch)

    async def go():
        a = await ag.enqueue(ROLES["sec"], "первая")
        b = await ag.enqueue(ROLES["sec"], "вторая")
        await ag.workers["sec"]
        return a, b
    a, b = asyncio.run(go())
    texts = [t for k, t in sent if k == "sec"]
    assert [c[0] for c in calls] == ["sec", "sec"] and "первая" in calls[0][1] and "вторая" in calls[1][1]
    assert any(t.startswith(f"Начал задачу №{a}") for t in texts) and any(t.startswith(f"Готово, задача №{a}") for t in texts)
    assert any(f"Задача №{b} в очереди" in t for t in texts) and any(t.startswith(f"Начал задачу №{b}") for t in texts)
    assert ag._queue("sec") == [] and ag.queue_text().startswith("Очередь пуста")
    rows = [__import__("json").loads(x) for x in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [r["role"] for r in rows] == ["sec", "sec"] and all(r["kind"] == "task" for r in rows)


def test_queue_survives_a_restart_and_a_task_can_be_cancelled(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    ag, _, _ = _team(tmp_path, monkeypatch)
    ag._set_queue("news", [{"id": 7, "text": "найди наборы", "source": "владелец", "at": "2026-11-17 10:00", "reply": None},
                           {"id": 8, "text": "ещё", "source": "владелец", "at": "2026-11-17 10:01", "reply": None}])
    ag2, _, calls = _team(tmp_path, monkeypatch)             # a new process reads the same state file
    assert [it["id"] for it in ag2._queue("news")] == [7, 8]
    assert "снята" in asyncio.run(ag2.cancel(8)) and "нет" in asyncio.run(ag2.cancel(99))

    async def go():
        ag2.kick(ROLES["news"])
        await ag2.workers["news"]
    asyncio.run(go())
    assert len(calls) == 1 and "найди наборы" in calls[0][1] and ag2._queue("news") == []


def test_a_task_stopped_by_the_limit_is_kept_and_retried(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    ag, sent, calls = _team(tmp_path, monkeypatch, metas={"dev": [{"ok": False, "limit": True}, {"ok": True}]})

    async def go():
        await ag.enqueue(ROLES["dev"], "почини")
        await ag.workers["dev"]
    asyncio.run(go())
    assert len(calls) == 2                                   # the same task ran again after the limit
    assert any("Лимит подписки исчерпан" in t and "сохранена" in t for _, t in sent)
    assert ag._queue("dev") == []


def test_news_findings_are_approved_into_the_coordinators_list(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    news = "Нашёл:\n1. [данные] Набор ЭКГ с разметкой P, 12 000 записей\n2. [ядро] Новые критерии АВ-блокады\n3. [реклама] мимо"
    ag, sent, calls = _team(tmp_path, monkeypatch, answers={"news": news, "coord": "Спокойно.\n1. [ядро] взять набор — почему: P"})

    async def go():
        await ag.turn(ROLES["news"], "отчёт", True)
        await ag.approve_news([1, 3])
        await ag.turn(ROLES["coord"], "план", True)
    asyncio.run(go())
    assert any("Находки на подтверждение" in t and "1. [данные]" in t and "[реклама]" not in t for _, t in sent)
    assert any("Подтверждено: 1" in t for _, t in sent)
    assert "approved_news.md" in calls[-1][1]                # the coordinator was pointed at the approved findings
    assert "Набор ЭКГ с разметкой P" in (tmp_path / "input" / "approved_news.md").read_text()
    assert ag.store["approved_news"] == [] and any("Задачи на выбор" in t for _, t in sent)
    assert "usage.md" in calls[-1][1] and "| Новости |" in (tmp_path / "input" / "usage.md").read_text()


def test_models_and_helpers_by_role(monkeypatch):
    from cardiovision_agents.logic import FABLE, helper_agents

    assert {k for k, r in ROLES.items() if r.model == FABLE} == {"coord", "dev", "sec", "audit"}
    cmd = runner.command(ROLES["dev"], "hi", None)
    assert cmd[cmd.index("--model") + 1] == FABLE and "--agents" in cmd
    helper = __import__("json").loads(cmd[cmd.index("--agents") + 1])["helper"]
    assert helper["model"] == "sonnet" and "Task" in cmd[cmd.index("--allowedTools") + 1]
    assert "Edit" not in helper_agents(ROLES["sec"])["helper"]["tools"]
    monkeypatch.setenv("AGENTS_MODEL_DEV", "opus")
    assert runner.model_for(ROLES["dev"]) == "opus"
    assert logic.MODEL_HINT.search("API Error: model claude-fable-5-2 is not available on your plan")
    assert logic.LIMIT_HINT.search("Claude usage limit reached. Your limit will reset at 3pm")
    assert logic.QUEUE.match("очередь") and logic.CANCEL.match("отмена 12").group(1) == "12"


# --- D301, the reviewer's findings ------------------------------------------------------------------------------
REAL_MODEL_ERROR = "There's an issue with the selected model (claude-fable-5-2). It may not exist or you may not have access to it."
REAL_LIMITS = ("You've hit your session limit · resets 3pm", "You've hit your limit", "You're out of usage credits",
               "You've hit your monthly spend limit", "Claude usage limit reached")


def _fake_once(monkeypatch, script):
    """runner._once replaced: ``script(model, session)`` -> (code, res, tail); returns the list of calls."""
    seen = []

    async def once(role, prompt, session, cwd, model=None):
        seen.append((model, session))
        return script(model, session)
    monkeypatch.setattr(runner, "_once", once)
    return seen


def test_unavailable_model_falls_back_to_opus_on_the_real_message(monkeypatch):
    seen = _fake_once(monkeypatch, lambda m, s: (1, {}, REAL_MODEL_ERROR) if m != "opus"
                      else (0, {"result": "готово", "total_cost_usd": 0.2}, ""))
    text, _, meta = asyncio.run(runner.run(ROLES["dev"], "p", None, "/tmp"))
    assert [m for m, _ in seen] == ["claude-fable-5-2", "opus"]
    assert text == "готово" and meta["ok"] and meta["fallback"] and meta["model"] == "opus"


def test_real_limit_messages_keep_the_task(monkeypatch):
    for msg in REAL_LIMITS:
        seen = _fake_once(monkeypatch, lambda m, s, msg=msg: (1, {}, msg))
        _, sid, meta = asyncio.run(runner.run(ROLES["scan"], "p", "sess", "/tmp"))
        assert meta.get("limit") and not meta["ok"], msg
        assert len(seen) == 1 and sid == "sess", msg         # no blind retry, the session is kept
    seen = _fake_once(monkeypatch, lambda m, s: (1, {}, "You've hit your Fable limit") if m != "opus" else (0, {"result": "ok"}, ""))
    _, _, meta = asyncio.run(runner.run(ROLES["sec"], "p", None, "/tmp"))
    assert meta["ok"] and meta["fallback"] and len(seen) == 2    # one model's own limit: the work goes on on Opus


def test_read_only_roles_are_hard_denied_write_and_shell():
    import json as _json

    from cardiovision_agents.logic import helper_agents
    for key in ("sec", "audit"):
        cmd = runner.command(ROLES[key], "hi", None)
        denied = cmd[cmd.index("--disallowedTools") + 1].split(",")
        assert {"Bash", "Edit", "Write"} <= set(denied)
        helper = _json.loads(cmd[cmd.index("--agents") + 1])["helper"]["tools"]
        assert not {"Bash", "Edit", "Write"} & set(helper)
        assert not {"Bash", "Edit", "Write"} & set(helper_agents(ROLES[key])["helper"]["tools"])
    dev = runner.command(ROLES["dev"], "hi", None)
    assert "Bash" not in dev[dev.index("--disallowedTools") + 1].split(",")


def test_usage_log_holds_no_task_text(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    ag, _, _ = _team(tmp_path, monkeypatch)

    async def go():
        await ag.enqueue(ROLES["triage"], "СЕКРЕТНЫЙ-ТЕКСТ-ЗАДАЧИ")
        await ag.workers["triage"]
    asyncio.run(go())
    raw = (tmp_path / "usage.jsonl").read_text()
    assert "СЕКРЕТНЫЙ" not in raw and "ответ triage" not in raw and '"role": "triage"' in raw


def test_second_branch_task_waits_for_the_owner_decision_on_the_first(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, sent, calls = _team(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "HOLD_POLL_S", 0.01)

    async def offer(role, workdir, branch):               # every dev task ends with a reviewer-approved branch
        pend = dict(ag.store.data.get("pending") or {})
        pend[role.key] = {"branch": branch, "sha": "a" * 40, "review": True}
        ag.store["pending"] = pend
    monkeypatch.setattr(ag, "_offer", offer)

    async def go():
        await ag.enqueue(ROLES["dev"], "первая")
        await ag.enqueue(ROLES["dev"], "вторая")
        await asyncio.sleep(0.3)
        first = ag.store["pending"]["dev"]["branch"]
        assert len(calls) == 1 and len(ag._queue("dev")) == 1          # the second did not start, the offer stands
        assert any("ждёт: ветка" in t and "пропустить" in t for _, t in sent)
        await ag.skip(ROLES["dev"])
        await ag.workers["dev"]
        return first
    first = asyncio.run(go())
    assert len(calls) == 2 and first and sum("ждёт: ветка" in t for _, t in sent) == 1


def test_a_bot_failure_does_not_block_or_endlessly_repeat_a_task(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, sent, calls = _team(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "FAIL_RETRY_S", 0)
    real_turn, n = ag.turn, {"n": 0}

    async def turn(role, text, scheduled=False, task_id=None):
        n["n"] += 1
        if text == "плохая":
            raise RuntimeError("telegram down")
        return await real_turn(role, text, scheduled, task_id)
    monkeypatch.setattr(ag, "turn", turn)

    async def go():
        await ag.enqueue(ROLES["sec"], "плохая")
        await ag.enqueue(ROLES["sec"], "хорошая")
        await ag.workers["sec"]
    asyncio.run(go())
    assert n["n"] == 3 and len(calls) == 1 and "хорошая" in calls[0][1]     # tried twice, dropped, the next one ran
    assert ag._queue("sec") == [] and any("Снята" in t for _, t in sent)


def test_stop_keeps_the_task_and_the_limit_retry_is_capped(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    ag, _, _ = _team(tmp_path, monkeypatch)

    async def paused_turn(role, text, scheduled=False, task_id=None):
        return {}                                           # what turn() returns after «стоп агенты»
    monkeypatch.setattr(ag, "turn", paused_turn)

    async def go():
        await ag.enqueue(ROLES["news"], "не потеряй")
        await ag.workers["news"]
    asyncio.run(go())
    assert [it["text"] for it in ag._queue("news")] == ["не потеряй"]

    from cardiovision_agents.logic import LIMIT_TRIES
    ag2, sent2, calls2 = _team(tmp_path, monkeypatch, metas={"dev": [{"ok": False, "limit": True} for _ in range(LIMIT_TRIES + 5)]})

    async def go2():
        await ag2.enqueue(ROLES["dev"], "под лимитом")
        await ag2.workers["dev"]
    asyncio.run(go2())
    assert len(calls2) == LIMIT_TRIES and len(ag2._queue("dev")) == 1
    assert any("больше не повторяю" in t for _, t in sent2)
    assert any("продолжу в " in t and ":" in t.split("продолжу в ")[1][:5] for _, t in sent2)


def test_backup_script_makes_a_decryptable_archive_and_cleans_up(tmp_path):
    import shutil
    import subprocess
    if not shutil.which("openssl"):
        pytest.skip("no openssl")
    src, out, key = tmp_path / "src", tmp_path / "out", tmp_path / "key"
    src.mkdir()
    (src / "a.txt").write_text("данные")
    out.mkdir()
    (out / "old.tmp").write_text("x")
    env = {**os.environ, "BACKUP_SRC": str(src), "BACKUP_DIR": str(out), "BACKUP_KEY": str(key)}
    r = subprocess.run(["bash", str(ROOT / "deploy/backup.sh")], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    files = sorted(p.name for p in out.iterdir() if p.name != ".lock")
    assert len(files) == 1 and files[0].endswith(".tar.gz.enc")  # the stale old.tmp is gone
    assert (out / files[0]).stat().st_mode & 0o077 == 0 and key.stat().st_mode & 0o077 == 0
    dec = subprocess.run(f"openssl enc -d -aes-256-cbc -pbkdf2 -pass file:{key} -in {out / files[0]} | tar tz",
                         shell=True, capture_output=True, text=True, check=False)
    assert "a.txt" in dec.stdout and key.read_text().strip() not in r.stdout + r.stderr


# --- D301, the second review ------------------------------------------------------------------------------------
def test_skip_during_ci_cancels_the_deploy(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    from cardiovision_agents import github
    ag, sent, _ = _team(tmp_path, monkeypatch)
    deployed = []

    async def ci(sha):
        await ag.skip(ROLES["dev"])                         # the owner changes their mind while CI runs
        return "green"

    async def ff(sha):
        deployed.append(sha)
        return ""
    monkeypatch.setattr(github, "ci_state", ci)
    monkeypatch.setattr(github, "fast_forward_and_deploy", ff)
    monkeypatch.setattr(B.asyncio, "sleep", asyncio.sleep)
    ag.store["pending"] = {"dev": {"branch": "agents/dev-20261117-1000", "sha": "a" * 40, "review": True}}
    asyncio.run(ag.deploy(ROLES["dev"]))
    assert deployed == [] and any("не продолжается" in t for _, t in sent)
    ag.store["pending"] = {"dev": {"branch": "b", "sha": "b" * 40, "review": None}}
    asyncio.run(ag.deploy(ROLES["dev"]))
    assert deployed == [] and any("не состоялась" in t for _, t in sent) and "проверки не было" in ag.status_text()


def test_telegram_outage_does_not_empty_the_queue(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, _, calls = _team(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "FAIL_RETRY_S", 0.2)
    real_say, down = ag.say, {"on": True}

    async def say(role, text):
        if down["on"]:
            raise OSError("telegram down")
        await real_say(role, text)
    monkeypatch.setattr(ag, "say", say)
    for i, t in enumerate(("а", "б", "в"), 1):
        ag._set_queue("sec", [*ag._queue("sec"), {"id": i, "text": t, "source": "x", "at": "", "reply": None}])

    async def go():
        ag.kick(ROLES["sec"])
        await asyncio.sleep(0.1)
        assert len(ag._queue("sec")) == 3 and calls == []   # nothing ran, nothing was dropped
        down["on"] = False
        await ag.workers["sec"]
    asyncio.run(go())
    assert len(calls) == 3 and ag._queue("sec") == []


def test_model_hint_ignores_ordinary_errors_and_news_approval_echoes_text(tmp_path, monkeypatch):
    for ok in ("the file tests/x.py may not exist", "invalid value for field in model Foo", "Session not found"):
        assert not logic.MODEL_HINT.search(ok), ok
    assert logic.MODEL_HINT.search(REAL_MODEL_ERROR)
    pytest.importorskip("aiogram")
    ag, sent, _ = _team(tmp_path, monkeypatch)
    ag.store["news_items"] = {"2": ["данные", "Набор X, 5 000 записей"]}
    asyncio.run(ag.approve_news([2]))
    assert any("Набор X, 5 000 записей" in t for _, t in sent)
    ag.store["paused"] = True
    ag.store["proposals"] = {"1": ["dev", "задача"]}
    asyncio.run(ag.dispatch([1]))
    assert ag.store["proposals"] == {"1": ["dev", "задача"]} and ag._queue("dev") == []


# --- D301, the third review -------------------------------------------------------------------------------------
def test_limit_then_bot_failure_still_retries_and_a_lost_answer_still_offers_the_branch(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    from cardiovision_agents import github
    ag, sent, calls = _team(tmp_path, monkeypatch, metas={"sec": [{"ok": False, "limit": True}, {"ok": True}]})
    monkeypatch.setattr(B, "FAIL_RETRY_S", 0)
    n = {"clone": 0}

    async def clone(*a, **k):
        n["clone"] += 1
        if n["clone"] == 2:                                 # the retry after the limit fails before the agent runs
            raise OSError("mirror busy")
    monkeypatch.setattr(github, "ensure_clone", clone)

    async def go():
        await ag.enqueue(ROLES["sec"], "важная")
        await ag.workers["sec"]
    asyncio.run(go())
    assert len(calls) == 2 and ag._queue("sec") == []       # limit run + the real run: the task was not dropped
    assert any("Повторю один раз" in t for _, t in sent) and not any("отработал" in t for _, t in sent)

    ag2, _, _ = _team(tmp_path, monkeypatch)
    offered, real_say = [], ag2.say

    async def say(role, text):
        if text.startswith("Готово"):
            raise OSError("flood control")
        await real_say(role, text)

    async def offer(role, workdir, branch):
        offered.append(branch)
    monkeypatch.setattr(ag2, "say", say)
    monkeypatch.setattr(ag2, "_offer", offer)

    async def go2():
        await ag2.enqueue(ROLES["dev"], "правка")
        await ag2.workers["dev"]
    asyncio.run(go2())
    assert len(offered) == 1 and ag2._queue("dev") == []


def test_an_undelivered_answer_is_never_silent(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    monkeypatch.setattr(B, "SEND_RETRY_S", 0)
    for fails, expect in ((1, "Готово, задача"), (2, "ответ не доставлен")):
        ag, sent, calls = _team(tmp_path, monkeypatch)
        real_say, left = ag.say, {"n": fails}

        async def say(role, text, real_say=real_say, left=left):
            if text.startswith("Готово") and left["n"] > 0:
                left["n"] -= 1
                raise OSError("timeout")
            await real_say(role, text)
        monkeypatch.setattr(ag, "say", say)

        async def go(ag=ag):
            await ag.enqueue(ROLES["sec"], "проверь")
            await ag.workers["sec"]
        asyncio.run(go())
        assert len(calls) == 1 and ag._queue("sec") == []
        assert any(expect in t for _, t in sent), (fails, sent)


# --- D308: economy mode ----------------------------------------------------------------------------------------
def test_unchanged_agent_reports_without_a_model_run_and_reports_use_the_cheap_model(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, sent, calls = _team(tmp_path, monkeypatch)

    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))
    assert ag.models_used == [("dev", ROLES["dev"].helper)]          # a routine report: the helper's model
    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))               # nothing happened in between
    assert len(calls) == 1 and "модель не запускалась" in sent[-1][1]

    async def task():
        await ag.enqueue(ROLES["dev"], "почини тест")
        await ag.workers["dev"]
    asyncio.run(task())                                              # the agent worked: the next report runs
    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))
    assert len(calls) == 3

    ag.store["pending"] = {"dev": {"branch": "agents/dev-x", "sha": "a" * 40, "review": True}}
    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))               # pending changed -> runs
    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))               # same again -> bot answers, with a reminder
    assert len(calls) == 4 and "ждёт вашего «да»" in sent[-1][1]

    for _ in range(2):                                # coordinator, news, security, errors always run
        for k in ("coord", "news", "sec", "triage"):
            asyncio.run(ag.turn(ROLES[k], B.REPORT, True))
    assert len(calls) == 12


def test_a_planned_report_neither_resumes_nor_replaces_the_working_session(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    from cardiovision_agents import runner
    seen = []

    async def run(role, prompt, session=None, cwd=None, model=None):
        seen.append(session)
        return "ok", "report-session", {"ok": True}
    ag, _, _ = _team(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "run", run)
    ag.store["sessions"] = {"dev": "work-session"}
    asyncio.run(ag.turn(ROLES["dev"], B.REPORT, True))
    assert seen == [None] and ag.store["sessions"]["dev"] == "work-session"


def test_tasks_review_morning_plan_and_news_keep_the_lead_model(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, _, _ = _team(tmp_path, monkeypatch)

    async def go():
        await ag.enqueue(ROLES["sec"], "проверь порты")
        await ag.workers["sec"]
        await ag.turn(ROLES["coord"], B.DAILY_PLAN, True)
        await ag.turn(ROLES["news"], B.REPORT, True)
        await ag._review(ROLES["dev"], "agents/dev-20261118-1000")
    asyncio.run(go())
    assert ag.models_used == [("sec", ROLES["sec"].model), ("coord", ROLES["coord"].model),
                              ("news", ROLES["news"].model), ("review", ROLES["review"].model)]


def test_a_limit_hit_pauses_planned_reports_and_queued_tasks_wait(tmp_path, monkeypatch):
    pytest.importorskip("aiogram")
    from cardiovision_agents import bot as B
    ag, sent, calls = _team(tmp_path, monkeypatch, metas={"dev": [{"ok": False, "limit": True}]})
    monkeypatch.setattr(B, "LIMIT_RETRY_MIN", 30)

    async def go():
        await ag.enqueue(ROLES["dev"], "почини")
        await asyncio.sleep(0.2)                        # the task hits the limit and is kept for later
        ag.workers["dev"].cancel()
        await ag.turn(ROLES["sec"], B.REPORT, True)     # planned report during the pause: no model run
        await ag.enqueue(ROLES["scan"], "проверь фото")  # a task of another agent waits for the shared pause
        await asyncio.sleep(0.2)
        ag.workers["scan"].cancel()
    asyncio.run(go())
    assert [k for k, _ in calls] == ["dev"]
    assert ag.limited() and any("пауза до" in t for _, t in sent)
    assert [t["text"] for t in ag._queue("dev")] == ["почини"] and [t["text"] for t in ag._queue("scan")] == ["проверь фото"]
    assert any(k == "scan" and "ждёт общей паузы" in t for k, t in sent)
