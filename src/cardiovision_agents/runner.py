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
WORKDIR = os.environ.get("AGENTS_WORKDIR", "/work/repo")
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


async def run(role: Role, prompt: str, session: str | None = None) -> tuple[str, str | None]:
    """(answer text, Claude session id). Errors come back as text for the topic, never as an exception."""
    proc = await asyncio.create_subprocess_exec(*command(role, prompt, session), cwd=WORKDIR, env=CLEAN_ENV,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        return f"Агент не уложился в {TIMEOUT_S // 60} мин и остановлен.", session
    try:
        res = json.loads(out.decode() or "{}")
    except json.JSONDecodeError:
        res = {}
    text = res.get("result")
    if proc.returncode != 0 or not text:
        tail = (err.decode(errors="replace") or out.decode(errors="replace"))[-600:]
        if "login" in tail.lower() or "auth" in tail.lower():
            return ("Claude Code на сервере не вошёл в подписку. Владельцу: один раз выполнить на сервере "
                    "`docker compose -f deploy/docker-compose.yml exec -u agent agents claude` и войти (/login)."), session
        return f"Агент завершился с ошибкой (код {proc.returncode}): {tail}", session
    return text, res.get("session_id") or session
