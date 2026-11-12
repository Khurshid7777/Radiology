"""``python -m ecgscan_bot``: the aiogram adapter (aiogram 3, MIT licence; SD-040).

Needs ``pip install aiogram`` and the environment described in
``configs/scanner/bot.yaml``. Each message is handled in a worker thread (the API
calls block); the photo bytes live only in memory for the upload. In ``analyze``
mode (D161) the reply is the CardioVision conclusion, the traces read drawn over the
photo (D163) and the PDF; several files sent together (a Telegram album) are one record;
in ``scan`` mode the scanner's verdict (SD-041).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os

from ecgscan.i18n import t
from ecgscan_bot.access import AccessStore
from ecgscan_bot.logic import (
    GROUP_CHAT_TYPES,
    AnalyzeApi,
    BotConfig,
    HourlyLimit,
    PaperScale,
    ScanApi,
    allowed,
    analyse,
    apply_choice,
    first_question,
    group_head,
    handle_file,
    intake,
    is_admin,
    lang_of,
    message_link,
    parse_caption,
    parse_scale,
    parse_symptoms,
    patient_answer,
    render,
    scale_buttons,
    symptom_buttons,
    toggle_symptom,
    watched_group,
)
from ecgscan_bot.texts import tr

TOKEN_ENV = "ECGSCAN_BOT_TOKEN"
ALBUM_WAIT_S = 2.0          # Telegram delivers an album's files one by one; wait this long after the last
RECALC_KEEP_S = 15 * 60     # a record's files are kept in memory this long for the speed / gain buttons
RECALC_MAX = 20             # ... and no more records than this (the oldest go first)
log = logging.getLogger("ecgscan_bot")


async def main() -> None:
    from aiogram import Bot

    cfg = BotConfig.load()
    token = os.environ.get(TOKEN_ENV)
    if not token or not cfg.api_key or not cfg.allowed:
        raise SystemExit(f"set {TOKEN_ENV}, ECGSCAN_BOT_API_KEY and ECGSCAN_BOT_ALLOWED")
    bot = Bot(token)
    api = AnalyzeApi(cfg) if cfg.mode == "analyze" else ScanApi(cfg)
    log.info("bot mode %s, api %s, %d allowed users", cfg.mode, cfg.api_url, len(cfg.allowed))
    await build(cfg, bot, api, AccessStore()).start_polling(bot)


def build(cfg: BotConfig, bot, api, store: AccessStore):
    """The dispatcher with every handler; the bot, the API client and the access store are given
    (the tests pass a bot that talks to no one)."""
    from aiogram import Dispatcher, F
    from aiogram.enums import ChatAction
    from aiogram.filters import Command, CommandObject, CommandStart
    from aiogram.types import (
        BufferedInputFile,
        CallbackQuery,
        ChatMemberUpdated,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message,
    )

    dp = Dispatcher()
    pending: dict[int, dict] = {}   # D209: chat -> the files waiting for the answers to the questions
    albums: dict[str, dict] = {}     # media_group_id -> {"files", "captions", "message", "lang", "task"}
    group_limit = HourlyLimit(cfg.group_max_per_hour)
    # The last records' files, in memory only, for re-running with another speed / gain.
    recent: dict[str, dict] = {}     # token -> {"files", "caption", "scale", "lang", "chat", "at"}

    def keep(rec: dict) -> str:
        now = asyncio.get_running_loop().time()
        for k in [k for k, v in recent.items() if now - v["at"] > RECALC_KEEP_S]:
            del recent[k]
        while len(recent) >= RECALC_MAX:
            del recent[min(recent, key=lambda k: recent[k]["at"])]
        token = os.urandom(6).hex()
        recent[token] = {**rec, "at": now}
        return token

    def actions(token: str, rec: dict, lang: str) -> InlineKeyboardMarkup:
        """Under a reply: speed / gain (D180), then symptoms, a link for a colleague, the language (D208)."""
        rows = [[InlineKeyboardButton(text=label, callback_data=f"re:{token}:{choice}") for label, choice in row]
                for row in scale_buttons(rec["scale"], lang)]
        rows.append([InlineKeyboardButton(text=tr("btn_symptoms", lang), callback_data=f"sy:{token}"),
                     InlineKeyboardButton(text=tr("btn_share", lang), callback_data=f"sh:{token}")])
        rows.append([InlineKeyboardButton(text=tr("btn_lang", lang), callback_data=f"lg:{token}")])
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def symptom_keyboard(token: str, rec: dict, lang: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"yt:{token}:{choice}") for label, choice in row]
            for row in symptom_buttons(rec["symptoms"], rec["onset"], lang)])

    def user_lang(user) -> str:
        return store.lang(user.id) or lang_of(user.language_code)

    def guard(message: Message) -> str | None:
        user = message.from_user
        ok = user is not None and allowed(cfg, user.id, message.chat.type, store)
        return user_lang(user) if ok else None

    def not_allowed(message: Message) -> str:
        user = message.from_user
        lang = lang_of(user.language_code) if user else "ru"
        if message.chat.type != "private":
            return tr("group", lang)
        return tr("need_code", lang, user_id=user.id if user else "?")

    async def keep_busy(chat_id: int) -> None:
        """"Sending a document…" in the chat while the analysis runs (Telegram shows it ~5 s)."""
        with contextlib.suppress(Exception):
            while True:
                await bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
                await asyncio.sleep(4)

    async def send_result(chat_id: int, text: str, aid: str | None, body: dict | None,
                          files: list[tuple[str, bytes]], caption: str | None, lang: str, scale: PaperScale,
                          symptoms: list[str] | None, onset: str | None, note: str | None,
                          head: str | None = None, overlay: bytes | None = None, pdf: bytes | None = None) -> None:
        """The reply to one chat: the conclusion, what was read drawn over the page (D163 — the doctor
        checks that the coloured lines lie on the curves before trusting it), the PDF, the buttons.
        ``head`` names the group a photo came from (SD-107)."""
        await bot.send_message(chat_id, f"{head}\n\n{text}" if head else text)
        if not aid:
            return
        if overlay is None:
            overlay = await asyncio.to_thread(api.overlay, aid)
        if overlay:
            await bot.send_photo(chat_id, BufferedInputFile(overlay, "overlay.jpg"),
                                 caption=t("app.overlay_hint", lang))
        if pdf is None:
            pdf = await asyncio.to_thread(api.pdf, aid)
        if pdf:
            await bot.send_document(chat_id, BufferedInputFile(pdf, f"cardiovision_{aid}.pdf"))
        sym, ons = (symptoms, onset) if symptoms is not None else parse_symptoms(caption)
        rec = {"files": files, "caption": caption, "scale": scale, "lang": lang, "chat": chat_id,
               "aid": aid, "body": body, "symptoms": list(sym), "onset": ons, "note": note,
               "patient_given": parse_caption(caption).given}
        token = keep(rec)
        await bot.send_message(chat_id, t("bot.recalc", lang) + "\n" + tr("actions", lang),
                               reply_markup=actions(token, rec, lang))

    async def answer_analysis(message: Message, files: list[tuple[str, bytes]], caption: str | None,
                              lang: str, scale: PaperScale | None = None, symptoms: list[str] | None = None,
                              onset: str | None = None, note: str | None = None) -> None:
        busy = asyncio.create_task(keep_busy(message.chat.id))
        scale = scale if scale is not None else parse_scale(caption)
        try:
            text, aid, body = await asyncio.to_thread(analyse, cfg, api, files, lang, caption, scale,
                                                      symptoms, onset, note)
            log.info("analysis %s answered (%d file(s))", aid, len(files))
            await send_result(message.chat.id, text, aid, body, files, caption, lang, scale, symptoms, onset, note)
        finally:
            busy.cancel()

    # --- SD-107: ECG photos posted in a group -> the result to administrators in private ---
    async def to_admins(send) -> None:
        """``send(user_id, lang)`` for every recipient; one who never started the bot is skipped."""
        for uid in sorted(cfg.group_recipients):
            try:
                await send(uid, store.lang(uid) or "ru")
            except Exception:
                log.warning("group result not delivered to one recipient (did they start the bot?)")

    async def group_analyse(chat, message_id: int | None, files: list[tuple[str, bytes]],
                            caption: str | None) -> None:
        if not group_limit.allow(chat.id, asyncio.get_running_loop().time()):
            log.warning("group %s: over %d analyses an hour, a photo skipped", chat.id, cfg.group_max_per_hour)
            return
        scale = parse_scale(caption)
        text, aid, body = await asyncio.to_thread(analyse, cfg, api, files, "ru", caption, scale)
        log.info("group %s: analysis %s (%d file(s))", chat.id, aid, len(files))
        if aid is None and not cfg.group_notify_refusals:
            return
        overlay = await asyncio.to_thread(api.overlay, aid) if aid else None
        pdf = await asyncio.to_thread(api.pdf, aid) if aid else None
        url = message_link(chat.id, message_id, getattr(chat, "username", None))
        given = parse_caption(caption).given

        async def send(uid: int, lang: str) -> None:
            mine = render(body, lang, scale, given) if body else text
            await send_result(uid, mine, aid, body, files, caption, lang, scale, None, None, None,
                              head=group_head(chat.title, url, lang), overlay=overlay, pdf=pdf)

        await to_admins(send)

    async def flush_group_album(key: str) -> None:
        await asyncio.sleep(ALBUM_WAIT_S)
        album = albums.pop(key, None)
        if album:
            caption = " ".join(c for c in album["captions"] if c) or None
            await group_analyse(album["message"].chat, album["message"].message_id, album["files"], caption)

    def group_card(chat_id: int, title: str | None, lang: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text=tr("btn_group_on", lang), callback_data=f"gr:{chat_id}:on"),
            InlineKeyboardButton(text=tr("btn_group_off", lang), callback_data=f"gr:{chat_id}:off")]])

    async def offer_group(chat) -> None:
        """Tell the administrators, in private, that the bot is in a group, with the switch."""
        store.set_group(chat.id, chat.title, store.group_on(chat.id))

        async def send(uid: int, lang: str) -> None:
            await bot.send_message(uid, tr("group_added", lang, title=(chat.title or "—")[:80]),
                                   reply_markup=group_card(chat.id, chat.title, lang))

        await to_admins(send)

    async def flush_album(group_id: str) -> None:
        await asyncio.sleep(ALBUM_WAIT_S)
        album = albums.pop(group_id, None)
        if album:
            caption = " ".join(c for c in album["captions"] if c) or None
            await begin(album["message"], album["files"], caption, album["lang"])

    async def begin(message: Message, files: list[tuple[str, bytes]], caption: str | None, lang: str) -> None:
        """D209: before the analysis, ask what the caption did not say — sex and age, then the
        doctor's diagnosis or description (both optional: "skip" / "no")."""
        now = asyncio.get_running_loop().time()
        stage = first_question(caption)
        pending[message.chat.id] = {"files": files, "caption": caption, "lang": lang, "stage": stage,
                                    "at": now, "message": message}
        await ask(message, stage, lang)

    async def ask(message: Message, stage: str, lang: str, again: bool = False) -> None:
        if stage == "patient":
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=tr("btn_skip", lang),
                                                                             callback_data="q:skip")]])
            await message.answer(tr("ask_patient_again" if again else "ask_patient", lang), reply_markup=kb)
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=tr("btn_no", lang),
                                                                             callback_data="q:no")]])
            await message.answer(tr("ask_note", lang), reply_markup=kb)

    async def run_pending(chat_id: int, note: str | None) -> None:
        p = pending.pop(chat_id, None)
        if p is None:
            return
        await p["message"].answer(tr("analysing", p["lang"]))
        await answer_analysis(p["message"], p["files"], p["caption"], p["lang"], note=note)

    def live_pending(chat_id: int) -> dict | None:
        p = pending.get(chat_id)
        if p and asyncio.get_running_loop().time() - p["at"] > RECALC_KEEP_S:
            pending.pop(chat_id, None)
            return None
        return p

    async def join(message: Message, code: str | None) -> None:
        user = message.from_user
        lang = lang_of(user.language_code) if user else "ru"
        if message.chat.type != "private" or user is None:
            await message.answer(tr("group", lang))
            return
        if code and store.join(user.id, code):
            log.info("user joined with an invite code")
            await message.answer(tr("code_ok", lang) + "\n\n" + t("bot.welcome", lang))
        else:
            await message.answer(tr("code_bad", lang))

    in_group = F.chat.type.in_(set(GROUP_CHAT_TYPES))

    @dp.my_chat_member()
    async def membership(event: ChatMemberUpdated) -> None:
        if event.chat.type not in GROUP_CHAT_TYPES:
            return
        status = event.new_chat_member.status
        if status in ("member", "administrator") and event.old_chat_member.status in ("left", "kicked"):
            log.info("added to group %s", event.chat.id)
            await offer_group(event.chat)
        elif status in ("left", "kicked"):
            log.info("removed from group %s", event.chat.id)
            store.set_group(event.chat.id, event.chat.title, False)

    @dp.message(in_group, Command("watch"))
    async def watch(message: Message) -> None:
        # An administrator asks in the group; the answer (the switch) goes to private chats only.
        if message.from_user and is_admin(cfg, message.from_user.id):
            await offer_group(message.chat)

    @dp.message(in_group, F.photo | F.document)
    async def group_upload(message: Message) -> None:
        chat = message.chat
        if cfg.mode != "analyze" or not watched_group(cfg, chat.id, chat.type, store):
            return
        try:
            doc = message.document
            if intake(cfg, kind="photo" if message.photo else "document", mime=doc.mime_type if doc else None,
                      size=doc.file_size if doc else None, lang="ru"):
                return                      # not an image, too large: nothing is said in a group
            buf = io.BytesIO()
            if message.photo:
                await bot.download(message.photo[-1], destination=buf)
                name = "photo.jpg"
            else:
                await bot.download(doc, destination=buf)
                name = doc.file_name or "ecg"
            if message.media_group_id:
                key = f"g:{chat.id}:{message.media_group_id}"
                album = albums.setdefault(key, {"files": [], "captions": [], "message": message, "task": None})
                album["files"].append((name, buf.getvalue()))
                album["captions"].append(message.caption)
                if album["task"]:
                    album["task"].cancel()
                album["task"] = asyncio.create_task(flush_group_album(key))
                return
            await group_analyse(chat, message.message_id, [(name, buf.getvalue())], message.caption)
        except Exception:
            log.exception("group upload failed")

    @dp.message(in_group)
    async def group_silence(message: Message) -> None:
        """Everything else in a group: the bot never writes there (D136, SD-107)."""

    @dp.callback_query(F.data.startswith("gr:"))
    async def group_switch(query: CallbackQuery) -> None:
        await query.answer()
        user, msg = query.from_user, query.message
        if msg is None or msg.chat.type != "private" or not is_admin(cfg, user.id):
            return
        _, chat_id, choice = (query.data or "").split(":", 2)
        known = {g["chat_id"]: g for g in store.groups()}
        if not chat_id.lstrip("-").isdigit() or int(chat_id) not in known:
            return
        on = choice == "on"
        store.set_group(int(chat_id), None, on, by=user.id)
        log.info("group %s reading %s", chat_id, "on" if on else "off")
        await msg.answer(tr("group_on" if on else "group_off", user_lang(user),
                            title=known[int(chat_id)].get("title") or chat_id))

    @dp.message(Command("groups"))
    async def groups(message: Message) -> None:
        user = message.from_user
        lang = user_lang(user) if user else "ru"
        if not (user and is_admin(cfg, user.id) and message.chat.type == "private"):
            await message.answer(tr("admin_only", lang))
            return
        rows = store.groups()
        if not rows:
            await message.answer(tr("groups_none", lang))
            return
        kb = [[InlineKeyboardButton(text=f"{'✓' if g.get('on') else '✗'} {g.get('title') or g['chat_id']}",
                                    callback_data=f"gr:{g['chat_id']}:{'off' if g.get('on') else 'on'}")]
              for g in rows]
        await message.answer(tr("groups_list", lang), reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))

    @dp.message(Command("admins"))
    async def admins(message: Message) -> None:
        user = message.from_user
        lang = user_lang(user) if user else "ru"
        if not (user and is_admin(cfg, user.id) and message.chat.type == "private"):
            await message.answer(tr("admin_only", lang))
            return
        await message.answer(tr("admins", lang, ids=", ".join(str(i) for i in sorted(cfg.allowed)),
                                group_ids=", ".join(str(i) for i in sorted(cfg.group_recipients))))

    @dp.message(CommandStart())
    async def start(message: Message, command: CommandObject) -> None:
        if command.args and not guard(message):          # t.me/<bot>?start=<code> (D208)
            await join(message, command.args)
            return
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else not_allowed(message))

    @dp.message(Command("code"))
    async def code(message: Message, command: CommandObject) -> None:
        await join(message, command.args)

    @dp.message(Command("invite"))
    async def invite(message: Message, command: CommandObject) -> None:
        user = message.from_user
        lang = lang_of(user.language_code) if user else "ru"
        if not (user and is_admin(cfg, user.id) and message.chat.type == "private"):
            await message.answer(tr("admin_only", lang))
            return
        label = (command.args or "").strip() or "—"
        new_code = store.new_code(label, user.id)
        me = await bot.get_me()
        await message.answer(tr("invite", lang, code=new_code, label=label,
                                link=f"https://t.me/{me.username}?start={new_code}"))

    @dp.message(Command("codes"))
    async def codes(message: Message) -> None:
        user = message.from_user
        lang = lang_of(user.language_code) if user else "ru"
        if not (user and is_admin(cfg, user.id) and message.chat.type == "private"):
            await message.answer(tr("admin_only", lang))
            return
        rows = store.codes()
        if not rows:
            await message.answer(tr("codes_none", lang))
            return
        await message.answer("\n".join(
            f"{'✗' if c['revoked'] else '✓'} {c['code']} — {c['label']} — {c['users']}" for c in rows))

    @dp.message(Command("revoke"))
    async def revoke(message: Message, command: CommandObject) -> None:
        user = message.from_user
        lang = lang_of(user.language_code) if user else "ru"
        if not (user and is_admin(cfg, user.id) and message.chat.type == "private"):
            await message.answer(tr("admin_only", lang))
            return
        c = (command.args or "").strip().upper()
        await message.answer(tr("revoked", lang, code=c) if store.revoke(c) else tr("revoke_unknown", lang))

    @dp.message(F.photo | F.document)
    async def upload(message: Message) -> None:
        lang = guard(message)
        if lang is None:
            await message.answer(not_allowed(message))
            return
        try:
            doc = message.document
            refusal = intake(cfg, kind="photo" if message.photo else "document",
                             mime=doc.mime_type if doc else None, size=doc.file_size if doc else None,
                             lang=lang)
            if refusal:
                await message.answer(refusal)
                return
            buf = io.BytesIO()
            if message.photo:     # D209: the largest size Telegram kept, read in the coarse mode (D170)
                await bot.download(message.photo[-1], destination=buf)
                name = "photo.jpg"
                # D228: say at once that this is a compressed copy and how to send the original
                # (once per album: only with its first photo).
                if cfg.mode == "analyze" and (not message.media_group_id or message.media_group_id not in albums):
                    await message.answer(tr("compressed", lang))
            else:
                await bot.download(doc, destination=buf)
                name = doc.file_name or "ecg"
            if cfg.mode != "analyze":
                await message.answer(t("bot.received", lang))
                text, scan_id = await asyncio.to_thread(handle_file, cfg, api, buf.getvalue(), name, lang)
                log.info("scan %s answered", scan_id)
                await message.answer(text)
                if cfg.send_overlay and scan_id:
                    png = await asyncio.to_thread(api.overlay, scan_id)
                    if png:
                        await message.answer_photo(BufferedInputFile(png, "overlay.jpg"))
                return
            group = message.media_group_id
            if group:
                # The files of one album arrive as separate messages: gather them, answer once.
                album = albums.setdefault(group, {"files": [], "captions": [], "message": message,
                                                   "lang": lang, "task": None})
                album["files"].append((name, buf.getvalue()))
                album["captions"].append(message.caption)
                if album["task"]:
                    album["task"].cancel()
                if len(album["files"]) == 1:
                    await message.answer(t("bot.received", lang))
                album["task"] = asyncio.create_task(flush_album(group))
                return
            await message.answer(t("bot.received", lang))
            await begin(message, [(name, buf.getvalue())], message.caption, lang)
        except Exception:  # Telegram download, a bug: the doctor still gets an answer
            log.exception("upload failed")
            with contextlib.suppress(Exception):
                await message.answer(t("bot.server_error", lang))

    def own(query: CallbackQuery) -> tuple[dict | None, str, str]:
        """The record a button belongs to, if this user in this chat may use it; (rec, token, rest)."""
        user = query.from_user
        parts = (query.data or "").split(":", 2)
        token, rest = (parts + ["", ""])[1], (parts + ["", "", ""])[2]
        if not allowed(cfg, user.id, query.message.chat.type if query.message else None, store):
            return None, token, rest
        rec = recent.get(token)
        if rec is None or query.message is None or rec["chat"] != query.message.chat.id:
            return None, token, rest
        return rec, token, rest

    async def expired(query: CallbackQuery) -> None:
        await query.answer()
        if query.message:
            await query.message.answer(tr("expired", user_lang(query.from_user)))

    @dp.callback_query(F.data.startswith("re:"))
    async def recalc(query: CallbackQuery) -> None:
        rec, token, choice = own(query)
        if rec is None:
            await expired(query)
            return
        scale = apply_choice(rec["scale"], choice)
        await query.answer()
        if scale == rec["scale"]:
            return
        with contextlib.suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        del recent[token]
        await answer_analysis(query.message, rec["files"], rec["caption"], rec["lang"], scale,
                              rec["symptoms"], rec["onset"], rec.get("note"))

    @dp.callback_query(F.data.startswith("sy:"))
    async def symptoms_open(query: CallbackQuery) -> None:
        rec, token, _ = own(query)
        if rec is None:
            await expired(query)
            return
        await query.answer()
        await query.message.answer(tr("symptoms_pick", rec["lang"]),
                                   reply_markup=symptom_keyboard(token, rec, rec["lang"]))

    @dp.callback_query(F.data.startswith("yt:"))
    async def symptoms_toggle(query: CallbackQuery) -> None:
        rec, token, choice = own(query)
        if rec is None:
            await expired(query)
            return
        await query.answer()
        if choice == "go":
            with contextlib.suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)
            del recent[token]
            await answer_analysis(query.message, rec["files"], rec["caption"], rec["lang"], rec["scale"],
                                  rec["symptoms"], rec["onset"], rec.get("note"))
            return
        rec["symptoms"], rec["onset"] = toggle_symptom(rec["symptoms"], rec["onset"], choice)
        with contextlib.suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=symptom_keyboard(token, rec, rec["lang"]))

    @dp.callback_query(F.data.startswith("sh:"))
    async def share(query: CallbackQuery) -> None:
        rec, _, _ = own(query)
        if rec is None or not rec.get("aid"):
            await expired(query)
            return
        await query.answer()
        url = await asyncio.to_thread(api.share, rec["aid"])
        await query.message.answer(tr("share", rec["lang"], url=url) if url else tr("share_fail", rec["lang"]))

    @dp.callback_query(F.data.startswith("lg:"))
    async def language(query: CallbackQuery) -> None:
        rec, token, _ = own(query)
        if rec is None or rec.get("body") is None:
            await expired(query)
            return
        await query.answer()
        rec["lang"] = "ru" if rec["lang"] == "uz" else "uz"
        store.set_lang(query.from_user.id, rec["lang"])
        await query.message.answer(render(rec["body"], rec["lang"], rec["scale"], rec["patient_given"]),
                                   reply_markup=actions(token, rec, rec["lang"]))

    @dp.callback_query(F.data.startswith("q:"))
    async def question_button(query: CallbackQuery) -> None:
        await query.answer()
        if query.message is None:
            return
        chat = query.message.chat.id
        p = live_pending(chat)
        if p is None or not allowed(cfg, query.from_user.id, query.message.chat.type, store):
            await query.message.answer(tr("pending_expired", user_lang(query.from_user)))
            return
        with contextlib.suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        if query.data == "q:skip" and p["stage"] == "patient":
            p["stage"] = "note"
            await ask(p["message"], "note", p["lang"])
        elif query.data == "q:no" and p["stage"] == "note":
            await run_pending(chat, None)

    @dp.message(F.text)
    async def text_answer(message: Message) -> None:
        lang = guard(message)
        if lang is None:
            await message.answer(not_allowed(message))
            return
        p = live_pending(message.chat.id)
        if p is None:
            await message.answer(t("bot.welcome", lang))
            return
        if p["stage"] == "patient":
            if patient_answer(message.text) is None:
                await ask(message, "patient", p["lang"], again=True)
                return
            # The answer joins the caption: parse_caption reads age and sex from it.
            p["caption"] = f"{p['caption'] or ''} {message.text}".strip()
            p["stage"] = "note"
            await ask(message, "note", p["lang"])
        else:
            await run_pending(message.chat.id, message.text)

    @dp.message()
    async def other(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else not_allowed(message))

    return dp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
