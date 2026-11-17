"""The agents bots (D279, D298): one supergroup; one bot per agent (or one bot with a topic per agent); commands only
from the owner.

    AGENTS_TG_COORD / _DEV / _SCAN / _REVIEW / _SEC / _TRIAGE / _NEWS   one bot per agent (D298); a missing one falls
                                               back to AGENTS_TG_TOKEN (the single bot of D279)
    AGENTS_OWNER_ID                            required
    AGENTS_GH_TOKEN                            the developer's pushes, the merge and the deploy after «да»
    AGENTS_SUMMARY_KEY, AGENTS_SITE_URL        the triage agent's input (the bot fetches it; the agent never sees the key)
    AGENTS_STATE (/state/agents.json), AGENTS_WORKDIR (/work/repo), AGENTS_SECURITY_REPORT (/security/report.txt)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from . import github, runner
from .logic import (
    ROLES,
    START,
    STATUS,
    STOP,
    Role,
    accepts,
    branch_name,
    build_prompt,
    is_approval,
    is_owner,
    parse_choice,
    parse_tasks,
    redact,
    review_ok,
    role_for_thread,
    split_message,
)
from .store import Store

log = logging.getLogger("agents")
TZ = ZoneInfo("Asia/Tashkent")
# (role, weekday or None for daily, hour, minute) — Tashkent time, off the round hour.
SCHEDULE = (("sec", None, 8, 17), ("triage", 0, 9, 23), ("news", 2, 10, 13), ("coord", 4, 16, 7))
WEEKLY_PLAN = ("Плановый запуск: составь задачи недели. Прочитай последние сообщения агентов (их отчёты лежат в "
               "/state/input/reports.md), docs/PROJECT_STATE.md и DECISIONS.md. Выбери не больше трёх задач, которые сильнее всего "
               "улучшат продукт по цифрам. Каждую — отдельной строкой ровно так: «ЗАДАЧА <номер> [ядро|сканер]: <что "
               "сделать и как проверить>», под ней одна строка «почему: <цифры>». Владелец ответит «да 1, 3».")
SERVER_NOTE = ("Here, on the server, the parts of your instructions about git branches, pushing, send_message and "
               "scheduled tasks do not apply: your answer is posted to your Telegram topic and that is your report.")


INPUT_DIR = Path(os.environ.get("AGENTS_INPUT_DIR", "/state/input"))   # root's, readable by the agent (D299 / B3)


def write_input(name: str, text: str) -> Path:
    """Write an input file for the agents into root's own directory, never following a link (D299 / B3)."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(INPUT_DIR, 0o755)
    path = INPUT_DIR / name
    if path.is_symlink():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _secrets() -> tuple[str, ...]:
    return tuple(os.environ.get(k, "") for k in ("AGENTS_TG_TOKEN", "AGENTS_GH_TOKEN", "AGENTS_SUMMARY_KEY"))


class Agents:
    def __init__(self, bot, store: Store, owner_id: int):
        # ``bot``: one bot for every agent (D279) or {role key: bot} (D298, a bot per agent).
        self.bots = dict(bot) if isinstance(bot, dict) else {k: bot for k in ROLES}
        self.bot = self.bots.get("coord") or next(iter(self.bots.values()))
        self.store, self.owner_id = store, owner_id
        self.lock = asyncio.Lock()          # one Claude run at a time: the site shares the server
        self.reply_to: dict[str, tuple[int, int | None]] = {}   # role -> where the owner last wrote to it

    @property
    def multi(self) -> bool:
        return len({id(b) for b in self.bots.values()}) > 1

    def _dest(self, role: Role | None) -> tuple[int | None, int | None]:
        if role is not None and role.key in self.reply_to:
            return self.reply_to[role.key]
        chat = self.store["chat_id"]
        if chat:
            thread = None if self.multi or role is None else self.store["topics"].get(role.key)
            return chat, thread
        return self.owner_id, None          # no group bound yet: straight to the owner's private chat

    async def say(self, role: Role | None, text: str) -> None:
        chat, thread = self._dest(role)
        if not chat:
            return
        bot = self.bots.get(role.key) if role else self.bot
        for part in split_message(redact(text, _secrets())):
            await (bot or self.bot).send_message(chat, part, message_thread_id=thread)

    # -- setup -------------------------------------------------------------------------------------------------
    async def setup(self, chat_id: int) -> str:
        if self.store["chat_id"] and self.store["chat_id"] != chat_id:
            return "Бот уже привязан к другой группе. Сначала /unbind там."
        self.store["chat_id"] = chat_id
        if self.multi:                      # D298: a bot per agent — they talk in the group itself, no topics
            for role in ROLES.values():
                await self.say(role, f"Я — «{role.title}». В группе обращайтесь ко мне через @ или ответом на моё "
                                     "сообщение; в личке пишите прямо. /new — начать разговор заново.")
            return "Готово: все агенты в группе. Команды принимаю только от владельца."
        topics = dict(self.store["topics"])
        for key, role in ROLES.items():
            if key not in topics:
                t = await self.bot.create_forum_topic(chat_id, role.title)
                topics[key] = t.message_thread_id
        self.store["topics"] = topics
        for role in ROLES.values():
            await self.say(role, f"Тема агента «{role.title}». Пишите задачу сюда — отвечать буду здесь же. "
                                 "/new — начать разговор заново.")
        return "Готово: темы созданы. Команды принимаю только от владельца."

    # -- one turn ----------------------------------------------------------------------------------------------
    async def turn(self, role: Role, text: str, scheduled: bool = False) -> None:
        if scheduled:
            self.reply_to.pop(role.key, None)   # planned reports go to the group (or the owner's private chat)
        if self.store["paused"] and not scheduled:
            await self.say(role, "Агенты остановлены («стоп агенты»). Включить: «старт агенты».")
            return
        if self.store["paused"]:
            return
        if self.lock.locked():
            await self.say(role, "В очереди: сейчас работает другой агент.")
        async with self.lock:
            workdir = runner.WORKDIR
            await github.ensure_clone(workdir, runner.AGENT_USER)
            extra = "" if role.branch else SERVER_NOTE
            branch = None
            if role.branch:
                branch = branch_name(dt.datetime.now(TZ).strftime("%Y%m%d-%H%M"), role.key)
                err = await github.prepare_branch(workdir, branch, runner.AGENT_USER)
                if err:
                    await self.say(role, f"Не удалось подготовить ветку: {err}")
                    return
            else:
                await github.checkout_main(workdir, runner.AGENT_USER)
            extra += await self._inputs(role)
            await self.say(role, "Принял, работаю…")
            sessions = dict(self.store["sessions"])
            answer, sid = await runner.run(role, build_prompt(role, text, extra), sessions.get(role.key))
            sessions[role.key] = sid
            self.store["sessions"] = sessions
            await self.say(role, answer)
            self._remember(role, answer)
            if role.key == "coord":
                tasks = parse_tasks(answer)
                if tasks:
                    self.store["proposals"] = {str(k): v for k, v in tasks.items()}
                    await self.say(role, "Ответьте здесь «да» и номера задач (например «да 1, 3») — раздам их агентам.")
            if branch:
                await self._offer(role, workdir, branch)

    def _remember(self, role: Role, answer: str) -> None:
        """The last report of every agent, for the coordinator's weekly list (kept short)."""
        last = dict(self.store.data.get("last_reports", {}))
        last[role.key] = {"at": dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), "text": redact(answer, _secrets())[-3000:]}
        self.store["last_reports"] = last

    async def _inputs(self, role: Role) -> str:
        if role.key == "coord":
            last = self.store.data.get("last_reports", {})
            body = "\n\n".join(f"## {ROLES[k].title} ({v['at']})\n{v['text']}" for k, v in last.items() if k in ROLES)
            path = write_input("reports.md", body or "(отчётов агентов пока нет)")
            return f"\nThe agents' latest reports are in {path}."
        if role.key == "sec":
            path = os.environ.get("AGENTS_SECURITY_REPORT", "/security/report.txt")
            return f"\nThe host's daily read-only check is in {path} — read it; it is your only view of the host."
        if role.key == "triage":
            key, url = os.environ.get("AGENTS_SUMMARY_KEY", ""), os.environ.get("AGENTS_SITE_URL", "http://cardiolens:8000")
            if not key:
                return "\nThe summary key is not set on the server: say so and stop."
            async with httpx.AsyncClient(timeout=30) as c:
                for days in (7, 30):
                    r = await c.get(f"{url}/api/quality/summary", params={"days": days}, headers={"X-API-Key": key})
                    write_input(f"summary_{days}.json", r.text if r.status_code == 200 else f'{{"error": {r.status_code}}}')
            return (f"\nThe server summary was fetched for you: {INPUT_DIR}/summary_7.json and summary_30.json "
                    "(do not request it yourself).")
        return ""

    async def _offer(self, role: Role, workdir: str, branch: str) -> None:
        ok, sha, out = await github.push(workdir, branch)
        if not ok:
            await self.say(role, f"Ветка не отправлена ({out}). Выкладывать нечего.")
            return
        verdict = await self._review(role, branch)
        pend = dict(self.store.data.get("pending") or {})
        pend[role.key] = {"branch": branch, "sha": sha, "review": verdict}
        self.store["pending"] = pend
        if verdict:
            await self.say(role, f"Ветка `{branch}` ({sha[:7]}) отправлена, проверяющий: можно. CI проверяет. Чтобы "
                                 "выложить на сайт — ответьте здесь «да». Выложу только при зелёном CI.")
        else:
            await self.say(role, f"Ветка `{branch}` ({sha[:7]}) отправлена, но проверяющий не пропустил — см. тему "
                                 "«Проверяющий». Напишите здесь, что исправить; выкладка этой ветки заблокирована.")

    async def _review(self, author: Role, branch: str) -> bool:
        """The reviewer reads the branch against main with fresh eyes (runs inside the author's turn and lock)."""
        rv = ROLES["review"]
        prompt = build_prompt(rv, (f"Проверь ветку `{branch}` агента «{author.title}» перед выкладкой: `git diff origin/main...{branch}` "
                                   f"и `git log origin/main..{branch}`. Повтори ключевые проверки сам (тесты затронутых частей). "
                                   "Последней строкой напиши ровно «ВЕРДИКТ: можно» или «ВЕРДИКТ: нельзя»."), SERVER_NOTE)
        answer, _ = await runner.run(rv, prompt, None)
        await self.say(rv, f"Ветка `{branch}` ({author.title}):\n{answer}")
        self._remember(rv, answer)
        return review_ok(answer) is True

    # -- deploy after «да» -------------------------------------------------------------------------------------
    async def deploy(self, role: Role) -> None:
        p = (self.store.data.get("pending") or {}).get(role.key)
        if not p:
            await self.say(role, "Нечего выкладывать: нет готовой ветки.")
            return
        if not p.get("review"):
            await self.say(role, "Проверяющий эту ветку не пропустил — не выкладываю.")
            return
        for _ in range(90):                       # CI takes ~10-25 min
            state = await github.ci_state(p["sha"])
            if state != "pending":
                break
            await asyncio.sleep(30)
        if state != "green":
            await self.say(role, f"CI по `{p['branch']}`: {state}. Не выкладываю.")
            return
        err = await github.fast_forward_and_deploy(p["sha"])
        if err:
            await self.say(role, err)
            return
        pend = dict(self.store.data.get("pending") or {})
        pend.pop(role.key, None)
        self.store["pending"] = pend
        await self.say(role, f"CI зелёный, main перемотан на {p['sha'][:7]}, выкладка запущена.")
        await asyncio.sleep(60)
        for _ in range(60):
            status, concl, url = await github.last_deploy()
            if status == "completed":
                await self.say(role, f"Выкладка: {concl}. {url}")
                return
            await asyncio.sleep(30)
        await self.say(role, "Выкладка идёт дольше 30 мин — проверьте Actions.")

    async def safe(self, role: Role, coro) -> None:
        """A background job whose failure is told to the owner instead of vanishing."""
        try:
            await coro
        except Exception as ex:
            log.exception("agent job failed")
            await self.say(role, f"Сбой: {type(ex).__name__}. Подробности — в журнале контейнера агентов.")

    # -- status and the coordinator's dispatch ------------------------------------------------------------------
    def status_text(self) -> str:
        lines = ["Агенты: " + ("остановлены" if self.store["paused"] else "работают") +
                 (" (сейчас кто-то занят)" if self.lock.locked() else "")]
        for k, p in (self.store.data.get("pending") or {}).items():
            if k in ROLES and p:
                lines.append(f"• {ROLES[k].title}: ветка `{p['branch']}` ({p['sha'][:7]}), проверяющий: "
                             + ("можно" if p.get("review") else "не пропустил") + " — ждёт «да» в своей теме")
        props = self.store.data.get("proposals") or {}
        if props:
            lines.append("Задачи недели (ответ «да 1, 3» в теме Координатор):")
            lines += [f"  {n}. [{ROLES[r].title}] {t}" for n, (r, t) in sorted(props.items(), key=lambda kv: int(kv[0]))]
        for k, v in (self.store.data.get("last_reports") or {}).items():
            if k in ROLES:
                lines.append(f"• последний отчёт «{ROLES[k].title}»: {v['at']}")
        sched = {"sec": "ежедневно 08:17", "triage": "пн 09:23", "news": "ср 10:13", "coord": "пт 16:07"}
        lines.append("Плановые запуски (Ташкент): " + ", ".join(f"{ROLES[k].title} — {w}" for k, w in sched.items()))
        return "\n".join(lines)

    async def dispatch(self, numbers: list[int]) -> None:
        props = self.store.data.get("proposals") or {}
        coord = ROLES["coord"]
        chosen = [(n, props[str(n)]) for n in numbers if str(n) in props]
        if not chosen:
            await self.say(coord, "Таких номеров в списке задач нет.")
            return
        await self.say(coord, "Раздаю: " + "; ".join(f"{n} → {ROLES[r].title}" for n, (r, _) in chosen))
        self.store["proposals"] = {k: v for k, v in props.items() if int(k) not in {n for n, _ in chosen}}
        for n, (r, task) in chosen:
            await self.say(ROLES[r], f"Задача {n} от координатора (владелец: «да»):\n{task}")
            asyncio.create_task(self.safe(ROLES[r], self.turn(ROLES[r], task)))

    # -- schedule ----------------------------------------------------------------------------------------------
    async def scheduler(self) -> None:
        done: set[str] = set()
        while True:
            now = dt.datetime.now(TZ)
            for key, wd, h, m in SCHEDULE:
                tag = f"{key}:{now:%Y-%m-%d}"
                if (wd is None or now.weekday() == wd) and (now.hour, now.minute) >= (h, m) and tag not in done:
                    done.add(tag)
                    if now.hour == h:            # missed by more than an hour (restart) -> skip, not catch up
                        text = WEEKLY_PLAN if key == "coord" else "Плановый запуск: сделай свой отчёт."
                        asyncio.create_task(self.safe(ROLES[key], self.turn(ROLES[key], text, True)))
            await asyncio.sleep(30)


def addressed_to(text: str, username: str | None, reply_from_id: int | None, bot_id: int | None) -> tuple[bool, str]:
    """D298: in a group with a bot per agent, a message is for this bot when it names it (@username) or answers it.
    Returns (for this bot?, text without the mention)."""
    tag = f"@{username}" if username else None
    if tag and tag.lower() in (text or "").lower():
        import re
        return True, re.sub(re.escape(tag), "", text, flags=re.IGNORECASE).strip()
    if bot_id is not None and reply_from_id == bot_id:
        return True, (text or "").strip()
    return False, text or ""


def build_dispatcher(agents: Agents, idents: dict[int, tuple[str, str | None]] | None = None):
    """``idents``: bot id -> (role key, username) for a bot per agent (D298); None for the single bot (D279)."""
    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import Command
    from aiogram.types import Message

    dp = Dispatcher()
    owner = agents.owner_id
    idents = idents or {}

    def own_role(bot: Bot) -> Role | None:
        key = idents.get(bot.id, (None, None))[0] if idents else None
        return ROLES.get(key) if key else None

    def primary(bot: Bot) -> bool:
        """Group-wide commands are answered by one bot only (the coordinator's, or the single bot)."""
        return not idents or idents.get(bot.id, (None,))[0] == "coord"

    async def handle(role: Role, text: str, msg: Message) -> None:
        if role.key == "coord" and parse_choice(text):
            asyncio.create_task(agents.safe(ROLES["coord"], agents.dispatch(parse_choice(text))))
            return
        if text.strip() == "/new":
            s = dict(agents.store["sessions"])
            s.pop(role.key, None)
            agents.store["sessions"] = s
            await msg.reply("Начинаем разговор заново.")
            return
        if role.branch and is_approval(text):
            asyncio.create_task(agents.safe(role, agents.deploy(role)))
            return
        asyncio.create_task(agents.safe(role, agents.turn(role, text)))

    async def common(text: str, msg: Message) -> bool:
        if STOP.match(text):
            agents.store["paused"] = True
            await msg.reply("Агенты остановлены, в том числе плановые запуски. Включить: «старт агенты».")
            return True
        if START.match(text):
            agents.store["paused"] = False
            await msg.reply("Агенты включены.")
            return True
        if STATUS.match(text):
            await msg.reply(agents.status_text())
            return True
        return False

    @dp.message(Command("setup"), F.chat.type == "supergroup")
    async def _setup(msg: Message, bot: Bot):
        if is_owner(msg.from_user and msg.from_user.id, owner) and primary(bot):
            await msg.reply(await agents.setup(msg.chat.id))

    @dp.message(Command("unbind"), F.chat.type == "supergroup")
    async def _unbind(msg: Message, bot: Bot):
        if (is_owner(msg.from_user and msg.from_user.id, owner) and agents.store["chat_id"] == msg.chat.id
                and primary(bot)):
            agents.store["chat_id"], agents.store["topics"] = None, {}
            await msg.reply("Отвязано.")

    @dp.message(F.chat.type == "private")
    async def _private(msg: Message, bot: Bot):
        if not is_owner(msg.from_user and msg.from_user.id, owner):
            return                                 # strangers get no answer at all
        text = msg.text or msg.caption or ""
        role = own_role(bot) or ROLES["coord"]
        if not text or await common(text, msg):
            return
        agents.reply_to[role.key] = (msg.chat.id, None)     # D298: answer in this private chat
        await handle(role, text, msg)

    @dp.message(F.chat.type == "supergroup")
    async def _group(msg: Message, bot: Bot):
        if not accepts(msg.from_user and msg.from_user.id, msg.chat.id, owner, agents.store["chat_id"]):
            return                                 # D279: only the owner commands; other chats are ignored
        text = msg.text or msg.caption or ""
        if idents:                                 # D298: a bot per agent
            username = idents.get(bot.id, (None, None))[1]
            reply_from = msg.reply_to_message.from_user.id if (msg.reply_to_message and msg.reply_to_message.from_user) else None
            mine, text = addressed_to(text, username, reply_from, bot.id)
            if not mine:
                others = any(addressed_to(msg.text or msg.caption or "", u, reply_from, i)[0]
                             for i, (_, u) in idents.items() if i != bot.id)
                if primary(bot) and not others:
                    await common(text, msg)        # «статус» / «стоп агенты» to the whole group: the coordinator answers
                return
            if await common(text, msg):
                return
            role = own_role(bot)
            if role is None or not text:
                return
            agents.reply_to[role.key] = (msg.chat.id, msg.message_thread_id)
            await handle(role, text, msg)
            return
        if await common(text, msg):
            return
        role = role_for_thread(agents.store["topics"], msg.message_thread_id)
        if role is None or not text:
            return
        agents.reply_to.pop(role.key, None)
        await handle(role, text, msg)

    return dp


TOKEN_ENV = {"coord": "AGENTS_TG_COORD", "dev": "AGENTS_TG_DEV", "scan": "AGENTS_TG_SCAN", "review": "AGENTS_TG_REVIEW",
             "sec": "AGENTS_TG_SEC", "triage": "AGENTS_TG_TRIAGE", "news": "AGENTS_TG_NEWS"}


async def main() -> None:
    from aiogram import Bot

    logging.basicConfig(level=logging.INFO)
    owner = int(os.environ["AGENTS_OWNER_ID"])
    single = os.environ.get("AGENTS_TG_TOKEN", "")
    tokens = {k: os.environ.get(env, "") or single for k, env in TOKEN_ENV.items()}
    fallback = tokens.get("coord") or next((t for t in tokens.values() if t), "")
    if not fallback:
        raise SystemExit("no Telegram token at all (AGENTS_TG_COORD ... or AGENTS_TG_TOKEN)")
    missing = [k for k, t in tokens.items() if not t]
    if missing:      # a missing agent bot speaks through the coordinator's (no restart loop)
        log.warning("no own bot for %s: they use the coordinator's", ", ".join(missing))
        tokens = {k: t or fallback for k, t in tokens.items()}
    by_token: dict[str, Bot] = {}
    bots = {k: by_token.setdefault(t, Bot(t)) for k, t in tokens.items()}
    store = Store(os.environ.get("AGENTS_STATE", "/state/agents.json"))
    agents = Agents(bots, store, owner)
    idents: dict[int, tuple[str, str | None]] = {}
    if agents.multi:
        for k, b in bots.items():
            me = await b.get_me()
            idents[me.id] = (k, me.username)
    asyncio.create_task(agents.scheduler())
    await build_dispatcher(agents, idents or None).start_polling(*by_token.values())
