"""Pure rules of the agents bot (D279): who may command, which agent a topic belongs to, approvals, redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Tools each agent may use in Claude Code (headless). The developer edits and runs tests in its clone; the others
# read. Nobody gets a tool that reaches the site's data: the container has none of it mounted (deploy compose).
DEV_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob")
READ_TOOLS = ("Read", "Grep", "Glob")


FABLE, OPUS, SONNET, HAIKU = "claude-fable-5-2", "opus", "sonnet", "haiku"
FALLBACK_MODEL = OPUS        # when the subscription does not serve a role's model


@dataclass(frozen=True)
class Role:
    key: str
    title: str          # topic name in the supergroup
    doc: str            # instructions, docs/agents/<doc>
    tools: tuple[str, ...]
    branch: bool        # works on its own git branch, the bot pushes it and offers a deploy
    model: str = OPUS   # D301: the department lead's model
    helper: str = SONNET  # D301: the model of its helper subagents (bulk legwork, checked by the lead)


ROLES: dict[str, Role] = {r.key: r for r in (
    Role("coord", "Координатор", "coordinator.md", READ_TOOLS + ("Bash",), False, FABLE, SONNET),
    Role("dev", "Программист", "developer.md", DEV_TOOLS, True, FABLE, SONNET),
    Role("scan", "Сканер", "scanner.md", DEV_TOOLS, True, OPUS, SONNET),
    Role("review", "Проверяющий", "reviewer.md", READ_TOOLS + ("Bash",), False, OPUS, SONNET),
    Role("sec", "Безопасность", "security.md", READ_TOOLS, False, FABLE, SONNET),
    Role("news", "Новости", "watch.md", READ_TOOLS + ("WebSearch", "WebFetch", "Write"), False, SONNET, HAIKU),
    Role("triage", "Ошибки", "triage.md", READ_TOOLS + ("Bash", "Write"), False, OPUS, SONNET),
    Role("audit", "Контролёр", "audit.md", READ_TOOLS, False, FABLE, HAIKU),
)}
HELPER_TOOLS = ("Task", "Agent")     # the tool a lead delegates with (name differs between Claude Code versions)


def helper_agents(role: Role) -> dict:
    """D301: the `helper` subagent of a department lead — a cheaper model for bulk legwork, same read tools."""
    tools = [t for t in role.tools if t not in ("Edit", "Write")] if not role.branch else list(role.tools)
    return {"helper": {
        "description": "Bulk legwork for the lead: searching and reading code, logs, pages; trying variants; drafts.",
        "prompt": ("You are a helper of a CardioVision department lead. Do exactly the piece of work you are given, "
                   "report facts with file paths / links and numbers, say what you could not check. Never print keys; "
                   "no patient data. The lead verifies your result."),
        "model": role.helper, "tools": tools}}


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


# The coordinator's task list: one line per task — "1. [ядро] <text>" (the older "ЗАДАЧА 1 [ядро]: <text>" is read too).
TASK_LINE = re.compile(r"^\s*(?:[-*•]\s*)?\**\s*(?:ЗАДАЧА\s+)?(\d+)\s*[.):]?\s*\**\s*\[\s*(ядро|сканер)\s*\]\s*\**\s*[:—–-]?\s*(.+?)\s*$",
                       re.IGNORECASE | re.MULTILINE)
TARGET = {"ядро": "dev", "сканер": "scan"}
CHOICE = re.compile(r"^\s*да\s*[,:]?\s*((?:\d+\s*(?:[,; ]|\bи\b|\band\b)\s*)*\d+)\s*[.!]*\s*$", re.IGNORECASE)   # D340: «да 1 и 3»
STATUS = re.compile(r"^\s*(статус|/status)\s*[?.!]*\s*$", re.IGNORECASE)
VERDICT = re.compile(r"ВЕРДИКТ\s*:\s*(можно|нельзя)", re.IGNORECASE)


def parse_tasks(text: str) -> dict[int, tuple[str, str]]:
    """{n: (role key, task text)} from the coordinator's weekly list."""
    return {int(n): (TARGET[t.lower()], body) for n, t, body in TASK_LINE.findall(text or "")}


NEWS_LINE = re.compile(r"^\s*(?:[-*•]\s*)?\**\s*(\d+)\s*[.):]\s*\**\s*\[\s*(данные|ядро|сканер|безопасность|платформа)\s*\]\s*\**\s*[:—–-]?\s*(.+?)\s*$",
                       re.IGNORECASE | re.MULTILINE)
QUEUE = re.compile(r"^\s*(очередь|/queue)\s*[?.!]*\s*$", re.IGNORECASE)
CANCEL = re.compile(r"^\s*отмена\s+(\d+)\s*[.!]*\s*$", re.IGNORECASE)
# Claude Code's own wording differs between versions («You've hit your session limit · resets …», «usage limit reached»).
LIMIT_HINT = re.compile(r"usage limit|limit reached|rate limit|limit will reset|out of (?:extra )?usage|5-hour limit|weekly limit|"
                        r"hit your .{0,40}limit|out of usage credits|spend limit", re.IGNORECASE)
MODEL_LIMIT_HINT = re.compile(r"hit your (fable|mythos|opus|sonnet|haiku)\b.{0,20}limit", re.IGNORECASE)   # one model's own limit
MODEL_HINT = re.compile(r"issue with the selected model|model.{0,60}(may not exist|not have access|not (?:found|available|supported)|"
                        r"does not exist)|(invalid|unknown|unsupported) model|model_not_found|not_found_error.{0,40}model",
                        re.IGNORECASE | re.DOTALL)
SKIP = re.compile(r"^\s*(пропустить|не выкладывать)\s*[.!]*\s*$", re.IGNORECASE)
LIMIT_TRIES = 12                     # D301: after that many limit hits a task waits for «старт агенты»
HARD_DENY = ("Bash", "Edit", "Write", "NotebookEdit")


def denied_tools(role: Role) -> tuple[str, ...]:
    """Tools a role must not get by any route (its helper or a built-in subagent included): a hard deny list."""
    return tuple(t for t in HARD_DENY if t not in role.tools)


def parse_news(text: str) -> dict[int, tuple[str, str]]:
    """{n: (tag, item)} from the news agent's numbered list «1. [данные] …»."""
    return {int(n): (tag.lower(), body) for n, tag, body in NEWS_LINE.findall(text or "")}


def news_menu(items: dict[int, tuple[str, str]]) -> str:
    lines = [f"{n}. [{tag}] {t}" for n, (tag, t) in sorted(items.items())]
    return ("Находки на подтверждение:\n" + "\n".join(lines)
            + "\n\nОтветьте на это сообщение (reply): «да» и номера — например «да 1, 3». Подтверждённое передам координатору в список задач.")


def task_menu(tasks: dict[int, tuple[str, str]]) -> str:
    """The clean numbered list the bot shows the owner (D300)."""
    lines = [f"{n}. [{ROLES[r].title}] {t}" for n, (r, t) in sorted(tasks.items())]
    return "Задачи на выбор:\n" + "\n".join(lines) + "\n\nНапишите «да» и номера — например «да 1, 3» (можно просто в группу)."


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


def build_prompt(role: Role, text: str, extra: str = "", works: bool | None = None) -> str:
    """``works``: the agent is on its own branch for a task (default: every turn of a branch role)."""
    works = role.branch if works is None else works
    head = (f"You are the «{role.title}» agent of CardioVision, running on the project's server and talking to the "
            f"owner in Telegram. First read docs/agents/README.md and docs/agents/{role.doc}; their rules bind "
            "you. Answer the owner in Russian, short and plain. Never print keys or tokens; there is no patient data "
            "here and you must not look for any.")
    head += (" You lead your department: hand bulky legwork (searching and reading many files, logs or pages, trying "
             "variants) to your `helper` subagent — it runs on a cheaper model, several at once if useful — and check "
             "its results yourself before you answer; decisions and the final answer are yours.")
    if role.key != "review":      # the reviewer's last line must stay «ВЕРДИКТ: …» (D299)
        head += (" When you finish a task, end with a short report for the owner: what you did, the result (with "
                 "numbers), and what is next or what you need from the owner.")
    if works:
        head += (" You are on your own git branch in this clone: commit your work there (git commit), do not switch "
                 "branches, do not push (the bot pushes). Run the full test suite before you finish. End with: what "
                 "changed, the test result, and whether it is ready to deploy.")
    return f"{head}\n\n{extra}\n\nСообщение владельца:\n{text}".replace("\n\n\n", "\n\n")
