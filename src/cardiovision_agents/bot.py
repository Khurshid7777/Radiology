"""The agents bot (D279): one supergroup, one topic per agent, commands only from the owner.

    AGENTS_TG_TOKEN, AGENTS_OWNER_ID          required
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
    STOP,
    Role,
    accepts,
    branch_name,
    build_prompt,
    is_approval,
    is_owner,
    redact,
    role_for_thread,
    split_message,
)
from .store import Store

log = logging.getLogger("agents")
TZ = ZoneInfo("Asia/Tashkent")
# (role, weekday or None for daily, hour, minute) — Tashkent time, off the round hour.
SCHEDULE = (("sec", None, 8, 17), ("triage", 0, 9, 23), ("news", 2, 10, 13))
SERVER_NOTE = ("Here, on the server, the parts of your instructions about git branches, pushing, send_message and "
               "scheduled tasks do not apply: your answer is posted to your Telegram topic and that is your report.")


def _secrets() -> tuple[str, ...]:
    return tuple(os.environ.get(k, "") for k in ("AGENTS_TG_TOKEN", "AGENTS_GH_TOKEN", "AGENTS_SUMMARY_KEY"))


class Agents:
    def __init__(self, bot, store: Store, owner_id: int):
        self.bot, self.store, self.owner_id = bot, store, owner_id
        self.lock = asyncio.Lock()          # one Claude run at a time: the site shares the server

    async def say(self, role: Role | None, text: str) -> None:
        chat = self.store["chat_id"]
        if not chat:
            return
        thread = self.store["topics"].get(role.key) if role else None
        for part in split_message(redact(text, _secrets())):
            await self.bot.send_message(chat, part, message_thread_id=thread)

    # -- setup -------------------------------------------------------------------------------------------------
    async def setup(self, chat_id: int) -> str:
        if self.store["chat_id"] and self.store["chat_id"] != chat_id:
            return "Бот уже привязан к другой группе. Сначала /unbind там."
        self.store["chat_id"] = chat_id
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
                branch = branch_name(dt.datetime.now(TZ).strftime("%Y%m%d-%H%M"))
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
            if branch:
                await self._offer(role, workdir, branch)

    async def _inputs(self, role: Role) -> str:
        if role.key == "sec":
            path = os.environ.get("AGENTS_SECURITY_REPORT", "/security/report.txt")
            return f"\nThe host's daily read-only check is in {path} — read it; it is your only view of the host."
        if role.key == "triage":
            key, url = os.environ.get("AGENTS_SUMMARY_KEY", ""), os.environ.get("AGENTS_SITE_URL", "http://cardiolens:8000")
            if not key:
                return "\nThe summary key is not set on the server: say so and stop."
            os.makedirs("/work/input", exist_ok=True)
            async with httpx.AsyncClient(timeout=30) as c:
                for days in (7, 30):
                    r = await c.get(f"{url}/api/quality/summary", params={"days": days}, headers={"X-API-Key": key})
                    Path(f"/work/input/summary_{days}.json").write_text(
                        r.text if r.status_code == 200 else f'{{"error": {r.status_code}}}')
            return ("\nThe server summary was fetched for you: /work/input/summary_7.json and summary_30.json "
                    "(do not request it yourself).")
        return ""

    async def _offer(self, role: Role, workdir: str, branch: str) -> None:
        ok, sha, out = await github.push(workdir, branch)
        if not ok:
            await self.say(role, f"Ветка не отправлена ({out}). Выкладывать нечего.")
            return
        self.store["pending"] = {"branch": branch, "sha": sha}
        await self.say(role, f"Ветка `{branch}` ({sha[:7]}) отправлена, CI проверяет. Чтобы выложить на сайт — "
                             "ответьте здесь «да». Выложу только при зелёном CI.")

    # -- deploy after «да» -------------------------------------------------------------------------------------
    async def deploy(self, role: Role) -> None:
        p = self.store["pending"]
        if not p:
            await self.say(role, "Нечего выкладывать: нет готовой ветки.")
            return
        for _ in range(90):                       # CI takes ~10-25 min
            state = await github.ci_state(p["sha"])
            if state != "pending":
                break
            await asyncio.sleep(30)
        if state != "green":
            await self.say(role, f"CI по `{p['branch']}`: {state}. Не выкладываю.")
            return
        err = await github.merge_and_deploy(p["branch"])
        if err:
            await self.say(role, err)
            return
        self.store["pending"] = None
        await self.say(role, "CI зелёный, влито в main, выкладка запущена.")
        await asyncio.sleep(60)
        for _ in range(60):
            status, concl, url = await github.last_deploy()
            if status == "completed":
                await self.say(role, f"Выкладка: {concl}. {url}")
                return
            await asyncio.sleep(30)
        await self.say(role, "Выкладка идёт дольше 30 мин — проверьте Actions.")

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
                        asyncio.create_task(self.turn(ROLES[key], "Плановый запуск: сделай свой отчёт.", True))
            await asyncio.sleep(30)


def build_dispatcher(agents: Agents):
    from aiogram import Dispatcher, F
    from aiogram.filters import Command
    from aiogram.types import Message

    dp = Dispatcher()
    owner = agents.owner_id

    @dp.message(Command("setup"), F.chat.type == "supergroup")
    async def _setup(msg: Message):
        if is_owner(msg.from_user and msg.from_user.id, owner):
            await msg.reply(await agents.setup(msg.chat.id))

    @dp.message(Command("unbind"), F.chat.type == "supergroup")
    async def _unbind(msg: Message):
        if is_owner(msg.from_user and msg.from_user.id, owner) and agents.store["chat_id"] == msg.chat.id:
            agents.store["chat_id"], agents.store["topics"] = None, {}
            await msg.reply("Отвязано.")

    @dp.message(F.chat.type == "private")
    async def _private(msg: Message):
        if is_owner(msg.from_user and msg.from_user.id, owner):
            await msg.answer("Добавьте меня администратором в супергруппу с темами и отправьте там /setup.")

    @dp.message(F.chat.type == "supergroup")
    async def _group(msg: Message):
        if not accepts(msg.from_user and msg.from_user.id, msg.chat.id, owner, agents.store["chat_id"]):
            return                                 # D279: only the owner commands; other chats are ignored
        text = msg.text or msg.caption or ""
        if STOP.match(text):
            agents.store["paused"] = True
            await msg.reply("Агенты остановлены, в том числе плановые запуски. Включить: «старт агенты».")
            return
        if START.match(text):
            agents.store["paused"] = False
            await msg.reply("Агенты включены.")
            return
        role = role_for_thread(agents.store["topics"], msg.message_thread_id)
        if role is None or not text:
            return
        if text.strip() == "/new":
            s = dict(agents.store["sessions"])
            s.pop(role.key, None)
            agents.store["sessions"] = s
            await msg.reply("Начинаем разговор заново.")
            return
        if role.branch and is_approval(text):
            asyncio.create_task(agents.deploy(role))
            return
        asyncio.create_task(agents.turn(role, text))

    return dp


async def main() -> None:
    from aiogram import Bot

    logging.basicConfig(level=logging.INFO)
    token, owner = os.environ["AGENTS_TG_TOKEN"], int(os.environ["AGENTS_OWNER_ID"])
    bot = Bot(token)
    agents = Agents(bot, Store(os.environ.get("AGENTS_STATE", "/state/agents.json")), owner)
    asyncio.create_task(agents.scheduler())
    await build_dispatcher(agents).start_polling(bot)
