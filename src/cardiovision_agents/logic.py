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


MEMBER_ADD = re.compile(r"^\s*добавить\s+участника\s+(\d{3,15})\s*[.!]*\s*$", re.IGNORECASE)    # D342
MEMBER_DEL = re.compile(r"^\s*убрать\s+участника\s+(\d{3,15})\s*[.!]*\s*$", re.IGNORECASE)
LESSONS = re.compile(r"^\s*уроки\s*[?.!]*\s*$", re.IGNORECASE)                             # D347
LESSON_DEL = re.compile(r"^\s*убрать\s+урок\s+(\d{1,4})\s*[.!]*\s*$", re.IGNORECASE)
MEMBERS = re.compile(r"^\s*участники\s*[?.!]*\s*$", re.IGNORECASE)


def is_member(user_id: int | None, members) -> bool:
    """D342: people the owner added to the group — they may talk to the agents, not command them."""
    return bool(user_id) and user_id in set(members or [])


def owner_only(text: str | None) -> bool:
    """D342: what only the owner may say — stop / start / cancel, choosing tasks or news, «да» to a branch, «пропустить»,
    adding or removing members. Anything else from a member is a question or a task for an agent."""
    t = re.sub(r"@\w+", " ", text or "")       # review D342: a mention anywhere (end, twice, new line) is stripped
    return bool(STOP.match(t) or START.match(t) or re.match(r"^\s*отмена\s+\d+", t, re.IGNORECASE)
                or CHOICE.match(t) or APPROVE.match(t) or SKIP.match(t)
                or MEMBER_ADD.match(t) or MEMBER_DEL.match(t) or LESSONS.match(t) or LESSON_DEL.match(t)
                or any(t.strip().startswith(c) for c in ("/setup", "/unbind", "/new")))


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
# D346: the coordinator hands the owner's task out itself, one line per part: «→ [ядро] что сделать и как проверить».
ROUTE_TARGET = {**TARGET, "проверка": "review", "безопасность": "sec", "ошибки": "triage", "новости": "news"}
MAX_ROUTES = 4
# D346 (+review): only the bot's own task headers count — a task number quoted somewhere in a text names nothing.
TASK_IN_REPLY = re.compile(r"^\s*(?:Готово, задача|Начал задачу|Раздал задачу|Задача)\s+№\s*(\d+)")
TASK_IN_TEXT = re.compile(r"^\s*(?:по\s+)?задач[аеу]\s*№?\s*(\d+)(?:\s*:|\s+[—–-](?=\s))\s*(\S.*)$", re.IGNORECASE | re.DOTALL)
ROUTE_LINE_MAX = 2000
ARROWS = ("→", "->", "=>")


def parse_routes(text: str) -> list[tuple[str, str]]:
    """[(role key, task)] from the coordinator's «→ [ядро] …» lines, at most MAX_ROUTES (D346)."""
    out = []
    for line in (text or "").splitlines():       # plain string work, linear: a regex here once took a minute on one line
        if len(line) > ROUTE_LINE_MAX:
            continue
        rest = line.strip()
        while rest and rest[0] in "-*• \t" and not rest.startswith(ARROWS):     # list bullets, but not the «->» arrow itself
            rest = rest[1:]
        arrow = next((a for a in ARROWS if rest.startswith(a)), None)
        if arrow is None:
            continue
        rest = rest[len(arrow):].lstrip("* \t")
        if not rest.startswith("[") or "]" not in rest:
            continue
        tag, _, body = rest[1:].partition("]")
        body = body.strip().lstrip("*").lstrip().lstrip(":—–-").strip()
        if tag.strip().lower() in ROUTE_TARGET and body:
            out.append((ROUTE_TARGET[tag.strip().lower()], body))
    return out[:MAX_ROUTES]


def task_ref(text: str | None, reply_text: str | None = None) -> tuple[int | None, str]:
    """D346: which task the owner's message is about — «задача 12: …» in the text itself, else the header of the
    bot's task message it replies to («Готово, задача №N», «Начал задачу №N», «Задача №N …», «Раздал задачу №N»).
    Returns (task id or None, the message without the reference). The caller passes ``reply_text`` only for a
    message one of our bots wrote."""
    m = TASK_IN_TEXT.match((text or "")[:4000])
    if m:
        return int(m.group(1)), (text or "")[m.start(2):].strip()
    m = TASK_IN_REPLY.match(reply_text or "")
    return (int(m.group(1)), (text or "").strip()) if m else (None, (text or "").strip())
CHOICE = re.compile(r"^\s*да\s*[,:]?\s*((?:\d+(?:\s*[,;]\s*|\s+(?:(?:и|and)\s+)?))*\d+)\s*[.!]*\s*$", re.IGNORECASE)   # D340: «да 1 и 3»
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


NEWS_MENU_HEAD = "Находки на подтверждение:"


def news_menu(items: dict[int, tuple[str, str]]) -> str:
    lines = [f"{n}. [{tag}] {t}" for n, (tag, t) in sorted(items.items())]
    return (NEWS_MENU_HEAD + "\n" + "\n".join(lines)
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


def build_prompt(role: Role, text: str, extra: str = "", works: bool | None = None, sender: str = "владельца") -> str:
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
    return f"{head}\n\n{extra}\n\nСообщение {sender}:\n{text}".replace("\n\n\n", "\n\n")


# -- D347: lessons, examples and the bot's own check -------------------------------------------------------------
LESSON_MENU_HEAD = "Уроки на подтверждение:"
MAX_LESSONS, LESSON_LEN = 15, 300
MENU_LESSONS = 3                                  # the controller proposes at most three at a time
SECURITY_FIX = re.compile(r"(?:срочно[\s.!:,]{1,4})?безопасность\s*(?:\([^)]{0,40}\))?\s*[:—–-]", re.IGNORECASE)


def is_security_fix(text: str | None) -> bool:
    """The task is a fix of a security finding: it starts with «Безопасность:» (bold, brackets, an arrow and a tag
    before it, «Срочно.», a dash instead of the colon are all the same word to the bot)."""
    head = (text or "")[:160]
    for ch in "*_`→>":
        head = head.replace(ch, " ")
    head = re.sub(r"^\s*(?:[-=]\s*)?(?:\[[^\]]{0,20}\]\s*)*", "", head)        # «[ядро]» / «[безопасность]» tags in front
    head = head.replace("[", " ").replace("]", " ").lstrip(" -=")
    return bool(SECURITY_FIX.match(head)) or bool(re.match(r"^\s*[*_ ]*\[\s*безопасность\s*\]", (text or "")[:60], re.IGNORECASE))


def _role_by_title(tag: str) -> str | None:
    tag = tag.strip().lower()
    return next((k for k, r in ROLES.items() if tag in (k, r.title.lower())), None)


def parse_lessons(text: str) -> dict[int, tuple[str, str]]:
    """{n: (role key, lesson)} from the controller's «1. [Программист] что делать иначе — задача №K» lines. Plain
    string work (no regex over the model's text); an unknown agent in the brackets is not a lesson."""
    out: dict[int, tuple[str, str]] = {}
    for line in (text or "").splitlines():
        rest = line.strip().lstrip("-*• \t")
        digits = rest[:len(rest[:4]) - len(rest[:4].lstrip("0123456789"))]
        if not digits or len(line) > 2000:
            continue
        rest = rest[len(digits):].lstrip(".):* \t")
        if not rest.startswith("[") or "]" not in rest:
            continue
        tag, _, body = rest[1:].partition("]")
        key, body = _role_by_title(tag), body.strip().lstrip("*").lstrip().lstrip(":—–-").strip()
        if key and body:
            out[int(digits)] = (key, body[:LESSON_LEN])
    return dict(sorted(out.items())[:MENU_LESSONS])


def lesson_menu(items: dict[int, tuple[str, str]]) -> str:
    lines = [f"{n}. [{ROLES[r].title}] {t}" for n, (r, t) in sorted(items.items())]
    return (LESSON_MENU_HEAD + "\n" + "\n".join(lines)
            + "\n\nОтветьте на это сообщение (reply): «да» и номера — например «да 1, 3». Урок начнёт действовать "
              "со следующей задачи агента. Показать действующие: «уроки»; снять: «убрать урок N».")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zа-яё0-9_]{4,}", (text or "").lower()[:2000])}


def similar_tasks(text: str, history: list[dict], role: str, limit: int = 3, floor: float = 0.2) -> list[dict]:
    """Up to ``limit`` finished tasks of the same role most like ``text`` (shared words / all words), not below
    ``floor`` — an unlike task is no example."""
    mine = _words(text)
    scored = []
    for h in history:
        if h.get("role") != role or not h.get("done"):
            continue
        other = _words(h.get("text", ""))
        score = len(mine & other) / len(mine | other) if mine and other else 0.0
        if score >= floor:
            scored.append((score, h.get("id", 0), h))
    return [h for _, _, h in sorted(scored, key=lambda x: (-x[0], -x[1]))[:limit]]


def outcome_text(h: dict) -> str:
    bits = []
    if h.get("review") is True:
        bits.append("проверяющий: можно")
    elif h.get("review") is False:
        bits.append("проверяющий: нельзя")
    if h.get("bot_check"):
        bits.append("бот вернул: " + str(h["bot_check"]))
    if h.get("redo"):
        bits.append(f"владелец просил переделать ({h['redo']})")
    if h.get("failed"):
        bits.append("сбой")
    return "; ".join(bits) or "без замечаний"


def bot_check(files: list[str]) -> str | None:
    """D347: the first «a mistake became a check» — code under src/ changed and no test did."""
    src = [f for f in files if f.startswith("src/") and f.endswith(".py")]
    tests = [f for f in files if f.startswith("tests/") and f.endswith(".py")]
    return "ветка меняет код в src/, но не меняет ни одного теста в tests/" if src and not tests else None
