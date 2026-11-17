"""Runs one agent turn: Claude Code headless as the unprivileged user `agent`, with a clean environment (D279).

The bot's tokens live only in the bot's own environment (root); the agent's process gets HOME, PATH and LANG and
nothing else, so a command the agent runs cannot read them (/proc/<pid>/environ of root is closed to it).
"""
from __future__ import annotations

import asyncio
import json
import os

from .logic import (
    FALLBACK_MODEL,
    HELPER_TOOLS,
    LIMIT_HINT,
    MODEL_HINT,
    MODEL_LIMIT_HINT,
    Role,
    denied_tools,
    helper_agents,
)

AGENT_USER = os.environ.get("AGENTS_RUN_AS", "agent")
WORKDIR = os.environ.get("AGENTS_WORKDIR", "/work/repo")     # the old shared clone (D279); now the base for per-role clones
WORK_BASE = os.environ.get("AGENTS_WORK_BASE", "/work")


def workdir_for(role: Role) -> str:
    """D300: every agent has its own clone, so several can work at the same time."""
    return f"{WORK_BASE}/{role.key}/repo"
TIMEOUT_S = int(os.environ.get("AGENTS_TIMEOUT_S", "2700"))
CLEAN_ENV = {"HOME": f"/home/{AGENT_USER}", "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
             "OMP_NUM_THREADS": "2"}


def model_for(role: Role) -> str:
    """The role's model; AGENTS_MODEL_<KEY> overrides it (e.g. AGENTS_MODEL_NEWS=haiku)."""
    return os.environ.get(f"AGENTS_MODEL_{role.key.upper()}", "") or role.model


def command(role: Role, prompt: str, session: str | None, model: str | None = None) -> list[str]:
    cmd = ["setpriv", f"--reuid={AGENT_USER}", f"--regid={AGENT_USER}", "--init-groups", "--",
           "nice", "-n", "10", "claude", "-p", prompt, "--output-format", "json",
           "--model", model or model_for(role),
           "--agents", json.dumps(helper_agents(role), ensure_ascii=False),
           "--allowedTools", ",".join(role.tools + HELPER_TOOLS)]
    if denied_tools(role):
        cmd += ["--disallowedTools", ",".join(denied_tools(role))]
    if session:
        cmd += ["--resume", session]
    return cmd


async def _once(role: Role, prompt: str, session: str | None, cwd: str, model: str | None = None) -> tuple[int, dict, str]:
    proc = await asyncio.create_subprocess_exec(*command(role, prompt, session, model), cwd=cwd, env=CLEAN_ENV,
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
    res = res if isinstance(res, dict) else {}
    tail = (err.decode(errors="replace") or out.decode(errors="replace"))[-600:]
    if res.get("is_error") and res.get("result"):
        tail = str(res["result"])[-600:] + " " + tail
        res = {k: v for k, v in res.items() if k != "result"}
    return proc.returncode, res, tail


def _meta(res: dict, model: str, **extra) -> dict:
    u = res.get("usage") or {}
    return {"model": model, "cost_usd": res.get("total_cost_usd"), "ms": res.get("duration_ms"),
            "turns": res.get("num_turns"), "in": u.get("input_tokens"), "out": u.get("output_tokens"), **extra}


async def run(role: Role, prompt: str, session: str | None = None, cwd: str | None = None) -> tuple[str, str | None, dict]:
    """(answer text, Claude session id, meta). Errors come back as text, never as an exception. ``meta``: model,
    cost_usd, ms, turns, tokens, ``ok``, ``limit`` (the subscription limit was hit), ``fallback`` (model replaced)."""
    cwd = cwd or workdir_for(role)
    model = model_for(role)
    extra: dict = {}
    code, res, tail = await _once(role, prompt, session, cwd, model)
    bad = code != 0 or not res.get("result")
    if bad and model != FALLBACK_MODEL and (MODEL_HINT.search(tail) or MODEL_LIMIT_HINT.search(tail)):
        # the subscription does not serve this model, or this model's own limit is spent: the work goes on on Opus
        model, extra["fallback"] = FALLBACK_MODEL, True
        code, res, tail = await _once(role, prompt, session, cwd, model)
        bad = code != 0 or not res.get("result")
    if bad and session and tail != "timeout" and not LIMIT_HINT.search(tail):
        # a session kept from another working directory (or expired) cannot be resumed: start a new conversation
        session = None
        code, res, tail = await _once(role, prompt, None, cwd, model)
        bad = code != 0 or not res.get("result")
    if tail == "timeout":
        return f"Агент не уложился в {TIMEOUT_S // 60} мин и остановлен.", session, _meta(res, model, ok=False, **extra)
    if bad:
        if LIMIT_HINT.search(tail):
            return "Лимит подписки исчерпан.", session, _meta(res, model, ok=False, limit=True, **extra)
        if "login" in tail.lower() or "auth" in tail.lower():
            return ("Claude Code на сервере не вошёл в подписку. Владельцу: один раз выполнить на сервере "
                    "`docker compose -f deploy/docker-compose.yml exec -u agent agents claude` и войти (/login)."), \
                session, _meta(res, model, ok=False, **extra)
        return f"Агент завершился с ошибкой (код {code}): {tail}", session, _meta(res, model, ok=False, **extra)
    return res["result"], res.get("session_id") or session, _meta(res, model, ok=True, **extra)
