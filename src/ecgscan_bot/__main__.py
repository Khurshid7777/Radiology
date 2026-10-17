"""``python -m ecgscan_bot``: the aiogram adapter (aiogram 3, MIT licence; SD-040).

Needs ``pip install aiogram`` and the environment described in
``configs/scanner/bot.yaml``. Each message is handled in a worker thread (the API
calls block); the photo bytes live only in memory for the upload.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os

from ecgscan.i18n import t
from ecgscan_bot.logic import BotConfig, ScanApi, allowed, handle_file, intake, lang_of

TOKEN_ENV = "ECGSCAN_BOT_TOKEN"
log = logging.getLogger("ecgscan_bot")


async def main() -> None:
    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import CommandStart
    from aiogram.types import BufferedInputFile, Message

    cfg = BotConfig.load()
    token = os.environ.get(TOKEN_ENV)
    if not token or not cfg.api_key or not cfg.allowed:
        raise SystemExit(f"set {TOKEN_ENV}, ECGSCAN_BOT_API_KEY and ECGSCAN_BOT_ALLOWED")
    bot, dp, api = Bot(token), Dispatcher(), ScanApi(cfg)

    def guard(message: Message) -> str | None:
        user = message.from_user
        return lang_of(user.language_code) if user and allowed(cfg, user.id) else None

    @dp.message(CommandStart())
    async def start(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else t("bot.not_allowed", "ru"))

    @dp.message(F.photo | F.document)
    async def upload(message: Message) -> None:
        lang = guard(message)
        if lang is None:
            await message.answer(t("bot.not_allowed", "ru"))
            return
        doc = message.document
        refusal = intake(cfg, kind="photo" if message.photo else "document",
                         mime=doc.mime_type if doc else None, size=doc.file_size if doc else None, lang=lang)
        if refusal:
            await message.answer(refusal)
            return
        await message.answer(t("bot.received", lang))
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        text, scan_id = await asyncio.to_thread(handle_file, cfg, api, buf.getvalue(), doc.file_name or "ecg", lang)
        log.info("scan %s answered", scan_id)
        await message.answer(text)
        if cfg.send_overlay and scan_id:
            png = await asyncio.to_thread(api.overlay, scan_id)
            if png:
                await message.answer_photo(BufferedInputFile(png, "overlay.jpg"))

    @dp.message()
    async def other(message: Message) -> None:
        lang = guard(message)
        await message.answer(t("bot.welcome", lang) if lang else t("bot.not_allowed", "ru"))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
