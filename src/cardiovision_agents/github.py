"""Git and GitHub for the developer agent (D279): its branch, the push, CI, the merge into main, the deploy.

The token stays in this (root) process: git gets it through GIT_CONFIG_* environment variables, which the agent's
user cannot read, never on a command line.
"""
from __future__ import annotations

import asyncio
import base64
import os

import httpx

REPO = os.environ.get("AGENTS_GH_REPO", "Khurshid7777/Radiology")
API = "https://api.github.com"


def _token() -> str:
    return os.environ.get("AGENTS_GH_TOKEN", "")


def _git_env() -> dict:
    basic = base64.b64encode(f"x-access-token:{_token()}".encode()).decode()
    return {**os.environ, "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}", "GIT_CONFIG_KEY_1": "safe.directory",
            "GIT_CONFIG_VALUE_1": "*", "GIT_TERMINAL_PROMPT": "0"}


async def git(workdir: str, *args: str) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec("git", "-C", workdir, *args, env=_git_env(),
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await p.communicate()
    return p.returncode, out.decode(errors="replace")


async def ensure_clone(workdir: str, user: str) -> None:
    if not os.path.isdir(os.path.join(workdir, ".git")):
        os.makedirs(os.path.dirname(workdir), exist_ok=True)
        p = await asyncio.create_subprocess_exec("git", "clone", f"https://github.com/{REPO}.git", workdir,
                                                 env=_git_env())
        await p.wait()
    await _chown(workdir, user)


async def _chown(path: str, user: str) -> None:
    p = await asyncio.create_subprocess_exec("chown", "-R", f"{user}:{user}", path)
    await p.wait()


async def prepare_branch(workdir: str, branch: str, user: str) -> str:
    await git(workdir, "fetch", "origin", "main")
    await git(workdir, "reset", "--hard")
    code, out = await git(workdir, "checkout", "-B", branch, "origin/main")
    await _chown(workdir, user)
    return "" if code == 0 else out[-300:]


async def checkout_main(workdir: str, user: str) -> None:
    await git(workdir, "fetch", "origin", "main")
    await git(workdir, "checkout", "-B", "main", "origin/main")
    await _chown(workdir, user)


async def push(workdir: str, branch: str) -> tuple[bool, str, str]:
    """(pushed, head sha, git output). Nothing to push (no commit on top of main) counts as not pushed."""
    _, ahead = await git(workdir, "rev-list", "--count", f"origin/main..{branch}")
    if ahead.strip() in ("", "0"):
        return False, "", "нет коммитов поверх main"
    code, out = await git(workdir, "push", "-u", "origin", f"{branch}:{branch}")
    _, sha = await git(workdir, "rev-parse", branch)
    return code == 0, sha.strip(), out[-300:]


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


async def merge_and_deploy(branch: str) -> str:
    async with _client() as c:
        r = await c.post(f"/repos/{REPO}/merges", json={"base": "main", "head": branch,
                                                         "commit_message": f"Merge {branch} (agents, owner: да)"})
        if r.status_code not in (201, 204):
            return f"слияние в main не удалось: {r.status_code} {r.text[:200]}"
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
