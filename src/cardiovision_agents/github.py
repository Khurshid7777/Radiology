"""Git and GitHub for the working agents (D279, D299): their branch, the push, CI, main and the deploy.

Trust boundary (D299, pre-deploy review): the bot runs as root and holds the tokens; the agents run as the unprivileged
user ``agent`` and own their clone (/work/repo). Root never runs git inside the agent's clone — its hooks, config or
fsmonitor would run with root's environment. Instead:

* root keeps its own mirror (``MIRROR``, root-owned, world-readable, never writable by the agent) and talks to GitHub
  only from there, with the token in GIT_CONFIG_* variables and a minimal environment;
* every git command inside the agent's clone runs AS the agent, with a clean environment (no tokens);
* to take the agent's commits, root fetches into the mirror with ``--upload-pack`` running as the agent — the agent's
  side of the transfer runs under the agent's uid, root only receives objects;
* the deploy fast-forwards ``main`` to the exact commit that passed CI and the reviewer (no new merge commit).
"""
from __future__ import annotations

import asyncio
import base64
import os
import re

import httpx

REPO = os.environ.get("AGENTS_GH_REPO", "Khurshid7777/Radiology")
API = "https://api.github.com"
MIRROR = os.environ.get("AGENTS_MIRROR", "/state/mirror")
AGENT = os.environ.get("AGENTS_RUN_AS", "agent")
HARDEN = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "protocol.file.allow=always")


def _token() -> str:
    return os.environ.get("AGENTS_GH_TOKEN", "")


def _root_env() -> dict:
    """Root's git: PATH, HOME and the GitHub token as an extra header — no Telegram tokens, no summary key."""
    basic = base64.b64encode(f"x-access-token:{_token()}".encode()).decode()
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}


def _agent_env() -> dict:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": f"/home/{AGENT}", "LANG": "C.UTF-8",
            "GIT_TERMINAL_PROMPT": "0"}


def as_agent() -> list[str]:
    return ["setpriv", f"--reuid={AGENT}", f"--regid={AGENT}", "--init-groups", "--"]


async def _run(cmd: list[str], env: dict, cwd: str | None = None) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*cmd, env=env, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    out, _ = await p.communicate()
    return p.returncode, out.decode(errors="replace")


async def mirror_git(*args: str) -> tuple[int, str]:
    """git as root in root's own mirror."""
    return await _run(["git", *HARDEN, "-C", MIRROR, *args], _root_env())


async def agent_git(workdir: str, *args: str) -> tuple[int, str]:
    """git as the agent in the agent's clone (no tokens in its environment)."""
    return await _run([*as_agent(), "git", "-C", workdir, *args], _agent_env())


_locks: dict[int, asyncio.Lock] = {}


def _mirror_lock() -> asyncio.Lock:
    """One lock per event loop for everything that touches root's mirror (D300: agents work in parallel)."""
    return _locks.setdefault(id(asyncio.get_running_loop()), asyncio.Lock())


async def ensure_mirror() -> None:
    async with _mirror_lock():
        await _mirror_sync()


async def _mirror_sync() -> None:
    if not os.path.isdir(os.path.join(MIRROR, "objects")):
        os.makedirs(os.path.dirname(MIRROR) or "/", exist_ok=True)
        await _run(["git", *HARDEN, "clone", "--bare", f"https://github.com/{REPO}.git", MIRROR], _root_env())
    await mirror_git("fetch", "origin", "+refs/heads/main:refs/heads/main")
    os.chmod(MIRROR, 0o755)       # readable by the agent (to clone / fetch main), never writable


async def ensure_clone(workdir: str, user: str = AGENT) -> None:
    async with _mirror_lock():
        await _ensure_clone(workdir, user)


async def _ensure_clone(workdir: str, user: str) -> None:
    await _mirror_sync()
    # The mirror is root's and only read by the agent: trusting that one path in the agent's own global config is safe
    # (git honours safe.directory only from protected config, not from -c for a local clone).
    await _run([*as_agent(), "git", "config", "--global", "--replace-all", "safe.directory", MIRROR], _agent_env())
    if not os.path.isdir(os.path.join(workdir, ".git")):
        # D299 / B3: only the clone itself is the agent's — its parent stays root's, so the agent cannot plant
        # symlinks next to it where root writes. Root makes an empty directory, hands over that one, the agent clones
        # into it.
        os.makedirs(workdir, exist_ok=True)
        if not os.listdir(workdir):
            await _run(["chown", f"{user}:{user}", workdir], {"PATH": "/usr/bin:/bin"})
        await _run([*as_agent(), "git", "clone", MIRROR, workdir], _agent_env())


async def prepare_branch(workdir: str, branch: str, user: str = AGENT) -> str:
    await ensure_mirror()
    await agent_git(workdir, "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main")
    await agent_git(workdir, "reset", "--hard")
    code, out = await agent_git(workdir, "checkout", "-B", branch, "origin/main")
    return "" if code == 0 else out[-300:]


async def checkout_main(workdir: str, user: str = AGENT, clean: bool = False) -> None:
    await ensure_mirror()
    await agent_git(workdir, "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main")
    if clean:       # D342: a member's task leaves nothing behind for the owner's next branch
        await agent_git(workdir, "reset", "--hard")
        await agent_git(workdir, "clean", "-fdq")
    await agent_git(workdir, "checkout", "-B", "main", "origin/main")


async def push(workdir: str, branch: str) -> tuple[bool, str, str]:
    """(pushed, head sha, output). The agent's branch is fetched into root's mirror with the agent's side of the
    transfer running as the agent, then pushed to GitHub from the mirror."""
    # The environment is emptied BEFORE setpriv, so not even setpriv sees root's git environment (the token).
    upload = " ".join(["env", "-i", "PATH=/usr/bin:/bin", *as_agent(), "git-upload-pack"])
    async with _mirror_lock():
        code, out = await mirror_git("fetch", f"--upload-pack={upload}", workdir, f"+refs/heads/{branch}:refs/heads/{branch}")
        if code != 0:
            return False, "", out[-300:]
        _, ahead = await mirror_git("rev-list", "--count", f"main..{branch}")
        if ahead.strip() in ("", "0"):
            return False, "", "нет коммитов поверх main"
        code, out = await mirror_git("push", "origin", f"refs/heads/{branch}:refs/heads/{branch}")
        _, sha = await mirror_git("rev-parse", branch)
    return code == 0, sha.strip(), out[-300:]


async def branch_files(workdir: str, rev: str = "HEAD") -> list[str]:
    """D347: the files the agent's branch changes against main (names only), read as the agent in its clone."""
    code, out = await agent_git(workdir, "diff", "--name-only", f"origin/main...{rev}")
    return [x.strip() for x in out.splitlines() if x.strip()] if code == 0 else []


async def pin(workdir: str, sha: str) -> bool:
    """D347: the clone shows exactly the pushed commit — no uncommitted edits, no leftovers of the reviewer's run —
    before the security agent reads it."""
    if not re.fullmatch(r"[0-9a-f]{7,64}", sha or ""):
        return False
    code, _ = await agent_git(workdir, "reset", "--hard", "--quiet", sha)
    code2, _ = await agent_git(workdir, "clean", "-fdq")
    code3, flags = await agent_git(workdir, "ls-files", "-v")
    return code == 0 and code2 == 0 and code3 == 0 and index_plain(flags)


def index_plain(ls_files_v: str) -> bool:
    """No file is hidden from the checkout (skip-worktree / assume-unchanged): `git ls-files -v` marks every file «H»."""
    return all(line.startswith("H ") for line in ls_files_v.splitlines() if line.strip())


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API, timeout=30, headers={
        "Authorization": f"Bearer {_token()}", "Accept": "application/vnd.github+json"})


async def ci_state(sha: str) -> str:
    """green / red / pending for the CI check runs of one commit."""
    async with _client() as c:
        r = await c.get(f"/repos/{REPO}/commits/{sha}/check-runs", params={"per_page": 50})
        r.raise_for_status()
        runs = r.json().get("check_runs", [])
    if not runs or any(x["status"] != "completed" for x in runs):
        return "pending"
    return "green" if all(x["conclusion"] in ("success", "skipped", "neutral") for x in runs) else "red"


async def fast_forward_and_deploy(sha: str) -> str:
    """Move main to exactly ``sha`` (fast-forward only — the commit CI and the reviewer passed), then start Deploy."""
    async with _client() as c:
        r = await c.patch(f"/repos/{REPO}/git/refs/heads/main", json={"sha": sha, "force": False})
        if r.status_code != 200:
            return (f"main не перемотан на {sha[:7]} ({r.status_code}): main ушёл вперёд — нужна новая ветка от "
                    "свежего main")
        r = await c.post(f"/repos/{REPO}/actions/workflows/deploy.yml/dispatches",
                         json={"ref": "main", "inputs": {"ref": "main"}})
        if r.status_code != 204:
            return f"выкладка не запущена: {r.status_code} {r.text[:200]}"
    return ""


async def last_deploy() -> tuple[str, str, str]:
    """(status, conclusion, url) of the newest Deploy run."""
    async with _client() as c:
        r = await c.get(f"/repos/{REPO}/actions/workflows/deploy.yml/runs", params={"per_page": 1})
        r.raise_for_status()
        run = (r.json().get("workflow_runs") or [{}])[0]
    return run.get("status", ""), run.get("conclusion") or "", run.get("html_url", "")
