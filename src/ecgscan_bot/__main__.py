"""``python -m ecgscan_bot``: the aiogram adapter (aiogram 3, MIT licence; SD-040).

Needs ``pip install aiogram`` and the environment described in
``configs/scanner/bot.yaml``. Each message is handled in a worker thread (the API
calls block); the photo bytes live only in memory for the upload. In ``analyze``
mode (D161) the reply is the CardioVision conclusion and its PDF; several files
sent together (a Telegram album) are one record; in ``scan`` mode the scanner's
verdict (SD-041).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os

from ecgscan.i18n import t
from ecgscan_bot.logic import (
    AnalyzeApi,
    BotConfig,
    ScanApi,
    allowed,
    handle_analysis,
    handle_file,
    intake,
    lang_of,
)

TOKEN_ENV = "ECGSCAN_BOT_TOKEN"
ALBUM_WAIT_S = 2.0          # Telegram delivers an album's files one by one; wait this long after the last
log = logging.getLogger("ecgscan_bot")


async def main() -> None:
    from aiogram import Bot, Dispatcher, F
    from aiogram.enums import ChatAction
    from aiogram.filters import CommandStart
    from aiogram.types import BufferedInputFile, Message

    cfg = BotConfig.load()
    token = os.environ.get(TOKEN_ENV)
    if not token or not cfg.api_key or not cfg.allowed:
        raise SystemExit(f"set {TOKEN_ENV}, ECGSCAN_BOT_API_KEY and ECGSCAN_BOT_ALLOWED")
    bot, dp = Bot(token), Dispatcher()
    api = AnalyzeApi(cfg) if cfg.mode == "analyze" else ScanApi(cfg)
    log.info("bot mode %s, api %s, %d allowed users", cfg.mode, cfg.api_url, len(cfg.allowed))
    albums: dict[str, dict] = {}     # media_group_id -> {"files", "captions", "message", "lang", "task"}

    def guard(message: Message) -> str | None:
        user = message.from_user
        ok = user is not None and allowed(cfg, user.id, message.chat.type)
        return lang_of(user.language_code) if ok else None

    def not_allowed(message: Message) -> str:
        # The id is what the administrator needs to open access (ECGSCAN_BOT_ALLOWED).
        return t("bot.not_allowed", "ru", user_id=message.from_user.id if message.from_user else "?")

    async def keep_busy(chat_id: int) -> None:
        """"Sending a document…" in the chat while the analysis runs (Telegram shows it ~5 s)."""
        with contextlib.suppress(Exception):
            while True:
                await bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
                await asyncio.sleep(4)

    async def answer_analysis(message: Message, files: list[tuple[str, bytes]], caption: str | None,
                              lang: str) -> None:
        busy = asyncio.create_task(keep_busy(message.chat.id))
        try:
            text, aid = await asyncio.to_thread(handle_analysis, cfg, api, files, lang, caption)
            log.info("analysis %s answered (%d file(s))", aid, len(files))
            await message.answer(text)
            if aid:
                pdf = await asyncio.to_thread(api.pdf, aid)
                if pdf:
                    await message.answer_document(BufferedInputFile(pdf, f"cardiovision_{aid}.pdf"))
        finally:
            busy.cancel()

    async def flush_album(group_id: str) -> None:
        await asyncio.sleep(ALBUM_WAIT_S)
        album = albums.pop(group_id, None)
        if album:
            caption = " ".join(c for c in album["captions"] if c) or None
            await answer_analysis(album["message"], album["files"], caption, album["lang"])

    @dp.message(CommandStart())
    async def start(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else not_allowed(message))

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

    @dp.message()
    async def other(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else not_allowed(message))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
