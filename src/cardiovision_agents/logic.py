"""Pure rules of the agents bot (D279): who may command, which agent a topic belongs to, approvals, redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Tools each agent may use in Claude Code (headless). The developer edits and runs tests in its clone; the others
# read. Nobody gets a tool that reaches the site's data: the container has none of it mounted (deploy compose).
DEV_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob")
READ_TOOLS = ("Read", "Grep", "Glob")


@dataclass(frozen=True)
class Role:
    key: str
    title: str          # topic name in the supergroup
    doc: str            # instructions, docs/agents/<doc>
    tools: tuple[str, ...]
    branch: bool        # works on its own git branch, the bot pushes it and offers a deploy


ROLES: dict[str, Role] = {r.key: r for r in (
    Role("coord", "Координатор", "coordinator.md", READ_TOOLS + ("Bash",), False),
    Role("dev", "Программист", "developer.md", DEV_TOOLS, True),
    Role("scan", "Сканер", "scanner.md", DEV_TOOLS, True),
    Role("review", "Проверяющий", "reviewer.md", READ_TOOLS + ("Bash",), False),
    Role("sec", "Безопасность", "security.md", READ_TOOLS, False),
    Role("news", "Новости", "watch.md", READ_TOOLS + ("WebSearch", "WebFetch", "Write"), False),
    Role("triage", "Ошибки", "triage.md", READ_TOOLS + ("Bash", "Write"), False),
)}

APPROVE = re.compile(r"^\s*(да|yes|ha|выкладывай|да,?\s*выкладывай)\s*[.!]*\s*$", re.IGNORECASE)
STOP = re.compile(r"^\s*стоп\s+агенты\s*[.!]*\s*$", re.IGNORECASE)
START = re.compile(r"^\s*старт\s+агенты\s*[.!]*\s*$", re.IGNORECASE)
TG_LIMIT = 4000


def is_owner(user_id: int | None, owner_id: int | None) -> bool:
    """Only the owner commands (D279: «только я»); no owner configured means nobody."""
    return bool(owner_id) and user_id == owner_id


def accepts(user_id: int | None, chat_id: int | None, owner_id: int | None, bound_chat: int | None) -> bool:
    """A group message is acted on only when it is the owner's and in the bound supergroup."""
    return is_owner(user_id, owner_id) and bound_chat is not None and chat_id == bound_chat


def role_for_thread(topics: dict[str, int], thread_id: int | None) -> Role | None:
    for key, tid in topics.items():
        if tid == thread_id and key in ROLES:
            return ROLES[key]
    return None


def is_approval(text: str | None) -> bool:
    return bool(text and APPROVE.match(text))


_TOKEN_PATTERNS = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),          # Telegram bot token
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),           # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),            # Anthropic keys
)


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove known secret values and anything shaped like a token before a text leaves the server."""
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[скрыто]")
    for p in _TOKEN_PATTERNS:
        text = p.sub("[скрыто]", text)
    return text


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Telegram takes 4096 characters per message; split on line breaks where possible."""
    text = text.strip() or "(пустой ответ)"
    out: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    out.append(text)
    return out


def branch_name(stamp: str, kind: str = "dev") -> str:
    """agents/<dev|scan>-<YYYYMMDD-HHMM>: a working agent's own branch, never main."""
    if not re.fullmatch(r"\d{8}-\d{4}", stamp) or kind not in ("dev", "scan"):
        raise ValueError("stamp must be YYYYMMDD-HHMM, kind dev or scan")
    return f"agents/{kind}-{stamp}"


# The coordinator's weekly list: one line per task, "ЗАДАЧА <n> [ядро|сканер]: <text>".
TASK_LINE = re.compile(r"^\s*ЗАДАЧА\s+(\d+)\s*\[\s*(ядро|сканер)\s*\]\s*[:—-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
TARGET = {"ядро": "dev", "сканер": "scan"}
CHOICE = re.compile(r"^\s*да\s*[,:]?\s*((?:\d+\s*[, ]\s*)*\d+)\s*[.!]*\s*$", re.IGNORECASE)
STATUS = re.compile(r"^\s*(статус|/status)\s*[?.!]*\s*$", re.IGNORECASE)
VERDICT = re.compile(r"ВЕРДИКТ\s*:\s*(можно|нельзя)", re.IGNORECASE)


def parse_tasks(text: str) -> dict[int, tuple[str, str]]:
    """{n: (role key, task text)} from the coordinator's weekly list."""
    return {int(n): (TARGET[t.lower()], body) for n, t, body in TASK_LINE.findall(text or "")}


def parse_choice(text: str | None) -> list[int]:
    """«да 1, 3» -> [1, 3]; a plain «да» or anything else -> []."""
    m = CHOICE.match(text or "")
    return sorted({int(x) for x in re.findall(r"\d+", m.group(1))}) if m else []


VERDICT_LINE = re.compile(r"\s*\**\s*ВЕРДИКТ\s*:\s*(можно|нельзя)\s*[.!]?\s*\**\s*", re.IGNORECASE)


def review_ok(text: str) -> bool | None:
    """The reviewer's verdict — ONLY its last non-empty line counts (D299: a quoted «ВЕРДИКТ: можно» earlier in the
    answer, e.g. from the author's commit message, must not pass a branch). True «можно», False «нельзя», None when
    the last line is not a verdict (treated as not passed)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    m = VERDICT_LINE.fullmatch(lines[-1]) if lines else None
    return None if m is None else m.group(1).lower() == "можно"


def build_prompt(role: Role, text: str, extra: str = "") -> str:
    head = (f"You are the «{role.title}» agent of CardioVision, running on the project's server and talking to the "
            f"owner in a Telegram topic. First read docs/agents/README.md and docs/agents/{role.doc}; their rules bind "
            "you. Answer the owner in Russian, short and plain. Never print keys or tokens; there is no patient data "
            "here and you must not look for any.")
    if role.branch:
        head += (" You are on your own git branch in this clone: commit your work there (git commit), do not switch "
                 "branches, do not push (the bot pushes). Run the full test suite before you finish. End with: what "
                 "changed, the test result, and whether it is ready to deploy.")
    return f"{head}\n\n{extra}\n\nСообщение владельца:\n{text}".replace("\n\n\n", "\n\n")
