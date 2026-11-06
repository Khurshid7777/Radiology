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
    AnalyzeApi,
    BotConfig,
    PaperScale,
    ScanApi,
    allowed,
    analyse,
    apply_choice,
    handle_file,
    intake,
    is_admin,
    lang_of,
    parse_scale,
    render,
    scale_buttons,
    symptom_buttons,
    toggle_symptom,
)
from ecgscan_bot.texts import tr

TOKEN_ENV = "ECGSCAN_BOT_TOKEN"
ALBUM_WAIT_S = 2.0          # Telegram delivers an album's files one by one; wait this long after the last
RECALC_KEEP_S = 15 * 60     # a record's files are kept in memory this long for the speed / gain buttons
RECALC_MAX = 20             # ... and no more records than this (the oldest go first)
log = logging.getLogger("ecgscan_bot")


async def main() -> None:
    from aiogram import Bot, Dispatcher, F
    from aiogram.enums import ChatAction
    from aiogram.filters import Command, CommandObject, CommandStart
    from aiogram.types import (
        BufferedInputFile,
        CallbackQuery,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message,
    )

    cfg = BotConfig.load()
    token = os.environ.get(TOKEN_ENV)
    if not token or not cfg.api_key or not cfg.allowed:
        raise SystemExit(f"set {TOKEN_ENV}, ECGSCAN_BOT_API_KEY and ECGSCAN_BOT_ALLOWED")
    bot, dp = Bot(token), Dispatcher()
    store = AccessStore()      # D208: invite codes and who joined with them
    api = AnalyzeApi(cfg) if cfg.mode == "analyze" else ScanApi(cfg)
    log.info("bot mode %s, api %s, %d allowed users", cfg.mode, cfg.api_url, len(cfg.allowed))
    albums: dict[str, dict] = {}     # media_group_id -> {"files", "captions", "message", "lang", "task"}
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

    async def answer_analysis(message: Message, files: list[tuple[str, bytes]], caption: str | None,
                              lang: str, scale: PaperScale | None = None, symptoms: list[str] | None = None,
                              onset: str | None = None) -> None:
        busy = asyncio.create_task(keep_busy(message.chat.id))
        scale = scale if scale is not None else parse_scale(caption)
        try:
            text, aid, body = await asyncio.to_thread(analyse, cfg, api, files, lang, caption, scale,
                                                      symptoms, onset)
            log.info("analysis %s answered (%d file(s))", aid, len(files))
            await message.answer(text)
            if aid:
                # What was read, drawn over the page (D163): the doctor checks that the
                # coloured lines lie on the curves before trusting the conclusion.
                overlay = await asyncio.to_thread(api.overlay, aid)
                if overlay:
                    await message.answer_photo(BufferedInputFile(overlay, "overlay.jpg"),
                                               caption=t("app.overlay_hint", lang))
                pdf = await asyncio.to_thread(api.pdf, aid)
                if pdf:
                    await message.answer_document(BufferedInputFile(pdf, f"cardiovision_{aid}.pdf"))
                from ecgscan_bot.logic import parse_caption, parse_symptoms

                sym, ons = (symptoms, onset) if symptoms is not None else parse_symptoms(caption)
                rec = {"files": files, "caption": caption, "scale": scale, "lang": lang, "chat": message.chat.id,
                       "aid": aid, "body": body, "symptoms": list(sym), "onset": ons,
                       "patient_given": parse_caption(caption).given}
                token = keep(rec)
                await message.answer(t("bot.recalc", lang) + "\n" + tr("actions", lang),
                                     reply_markup=actions(token, rec, lang))
        finally:
            busy.cancel()

    async def flush_album(group_id: str) -> None:
        await asyncio.sleep(ALBUM_WAIT_S)
        album = albums.pop(group_id, None)
        if album:
            caption = " ".join(c for c in album["captions"] if c) or None
            await answer_analysis(album["message"], album["files"], caption, album["lang"])

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
            await answer_analysis(message, [(name, buf.getvalue())], message.caption, lang)
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
                              rec["symptoms"], rec["onset"])

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
                                  rec["symptoms"], rec["onset"])
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

    @dp.message()
    async def other(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else not_allowed(message))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
