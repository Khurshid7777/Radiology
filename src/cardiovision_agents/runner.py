"""Runs one agent turn: Claude Code headless as the unprivileged user `agent`, with a clean environment (D279).

The bot's tokens live only in the bot's own environment (root); the agent's process gets HOME, PATH and LANG and
nothing else, so a command the agent runs cannot read them (/proc/<pid>/environ of root is closed to it).
"""
from __future__ import annotations

import asyncio
import json
import os

from .logic import Role

AGENT_USER = os.environ.get("AGENTS_RUN_AS", "agent")
WORKDIR = os.environ.get("AGENTS_WORKDIR", "/work/repo")     # the old shared clone (D279); now the base for per-role clones
WORK_BASE = os.environ.get("AGENTS_WORK_BASE", "/work")


def workdir_for(role: Role) -> str:
    """D300: every agent has its own clone, so several can work at the same time."""
    return f"{WORK_BASE}/{role.key}/repo"
TIMEOUT_S = int(os.environ.get("AGENTS_TIMEOUT_S", "2700"))
CLEAN_ENV = {"HOME": f"/home/{AGENT_USER}", "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
             "OMP_NUM_THREADS": "2"}


def command(role: Role, prompt: str, session: str | None) -> list[str]:
    cmd = ["setpriv", f"--reuid={AGENT_USER}", f"--regid={AGENT_USER}", "--init-groups", "--",
           "nice", "-n", "10", "claude", "-p", prompt, "--output-format", "json",
           "--allowedTools", ",".join(role.tools)]
    if session:
        cmd += ["--resume", session]
    return cmd


async def _once(role: Role, prompt: str, session: str | None, cwd: str) -> tuple[int, dict, str]:
    proc = await asyncio.create_subprocess_exec(*command(role, prompt, session), cwd=cwd, env=CLEAN_ENV,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        return -9, {}, "timeout"
    try:
        res = json.loads(out.decode() or "{}")
    except json.JSONDecodeError:
        res = {}
    return proc.returncode, res if isinstance(res, dict) else {}, (err.decode(errors="replace") or out.decode(errors="replace"))[-600:]


async def run(role: Role, prompt: str, session: str | None = None, cwd: str | None = None) -> tuple[str, str | None]:
    """(answer text, Claude session id). Errors come back as text for the topic, never as an exception."""
    cwd = cwd or workdir_for(role)
    code, res, tail = await _once(role, prompt, session, cwd)
    if (code != 0 or not res.get("result")) and session and tail != "timeout":
        # a session kept from another working directory (or expired) cannot be resumed: start a new conversation
        session = None
        code, res, tail = await _once(role, prompt, None, cwd)
    if tail == "timeout":
        return f"Агент не уложился в {TIMEOUT_S // 60} мин и остановлен.", session
    text = res.get("result")
    if code != 0 or not text:
        if "login" in tail.lower() or "auth" in tail.lower():
            return ("Claude Code на сервере не вошёл в подписку. Владельцу: один раз выполнить на сервере "
                    "`docker compose -f deploy/docker-compose.yml exec -u agent agents claude` и войти (/login)."), session
        return f"Агент завершился с ошибкой (код {code}): {tail}", session
    return text, res.get("session_id") or session
