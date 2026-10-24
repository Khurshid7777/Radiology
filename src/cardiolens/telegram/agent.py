"""The Telegram collector: a Telethon client on the user's own account.

Reads photos in the configured groups (never writes there), runs ``analyze``,
posts a card per photo to the review chat and turns the reviewer's replies into
labels. Telethon is imported lazily so the rest of the package has no hard
dependency on it (``pip install -e ".[telegram]"``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

from cardiolens.config import load_app_config
from cardiolens.feedback import FeedbackStore
from cardiolens.layer6_interpret import default_provider
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.telegram.analyze import (
    VERDICT_HELP,
    analyse_photos,
    apply_verdict,
    feedback_from_verdict,
    format_card,
    is_ecg_photo,
    parse_verdict,
    render_review_png,
    save_photo,
    save_signal,
    summarize,
)
from cardiolens.telegram.config import TelegramConfig
from cardiolens.telegram.store import DatasetStore, Record
from cardiolens.validation.harness import SCORED_LABELS

log = logging.getLogger("cardiolens.telegram")


IMAGE_DOC_MAX_BYTES = 20 * 1024 * 1024   # a phone photo sent as a file is a few MB


def is_image_message(msg) -> bool:
    """A photo, or an image sent as a file (D129): Telegram shrinks photos to ~1280 px,
    and the bot asks for small, unreliable photos to be sent again as files — those
    arrive as documents with an image MIME type, not as ``photo``."""
    if getattr(msg, "photo", None):
        return True
    doc = getattr(msg, "document", None)
    mime = (getattr(doc, "mime_type", "") or "") if doc is not None else ""
    size = getattr(doc, "size", 0) or 0
    return mime.startswith("image/") and size <= IMAGE_DOC_MAX_BYTES


class Collector:
    """Everything the agent does with one message; the Telethon client is injected."""

    def __init__(self, client, cfg: TelegramConfig, store: DatasetStore, review_chat):
        self.client = client
        self.cfg = cfg
        self.store = store
        self.review_chat = review_chat
        self.feedback = FeedbackStore.at(cfg.feedback_path)
        self.provider = default_provider(llm=load_app_config().llm) if cfg.narrative else StubProvider()

    async def handle_photo(self, msg, chat) -> Record | None:
        return await self.handle_album([msg], chat)

    async def handle_album(self, msgs, chat) -> Record | None:
        """One card for the photos of one message or album (D123): the pages of a recorder's
        screen (TeleECG: limb page + chest page) are read as one record; the first ECG
        photo's record carries the analysis, the others are marked merged into it."""
        chat_id = int(getattr(chat, "id", msgs[0].chat_id))
        recs = []
        for msg in msgs:
            if self.store.has(chat_id, msg.id):
                continue
            rec_id = DatasetStore.record_id(chat_id, msg.id)
            data = await msg.download_media(bytes)
            if not data:
                continue
            try:
                photo = save_photo(data, self.store.photos / f"{rec_id}.jpg")
            except ValueError:        # a file the decoder cannot read (HEIC, a broken upload)
                log.info("%s: not a readable image — skipped", rec_id)
                continue
            sender = str(msg.sender_id or "")
            rec = Record(
                id=rec_id, chat_id=chat_id, msg_id=msg.id,
                chat_title=getattr(chat, "title", "") or "",
                date=msg.date.strftime("%Y-%m-%dT%H:%M:%S") if msg.date else "",
                sender_hash=hashlib.sha256(sender.encode()).hexdigest()[:12] if sender else "",
                caption=(msg.message or "")[:1000], photo_path=str(photo),
            )
            ok, pitch = await asyncio.to_thread(is_ecg_photo, photo, self.cfg.min_grid_pitch_px)
            rec.grid_pitch_px = pitch
            if not ok:
                rec.status = "not_ecg"
                self.store.upsert(rec)
                log.info("%s: no ECG grid (pitch %s) — skipped", rec_id, pitch)
            recs.append(rec)
        ecg = [r for r in recs if r.status != "not_ecg"]
        if not ecg:
            return recs[0] if recs else None

        rec = ecg[0]
        photos = [r.photo_path for r in ecg]
        result, error = await asyncio.to_thread(analyse_photos, photos, mode=self.cfg.mode, provider=self.provider)
        files = list(photos)
        if result is None:
            rec.status = "digitise_failed"
            rec.error = error
        else:
            rec.status = "analysed"
            rec.predicted = sorted(result.findings.codes())
            rec.critical = [a.code for a in result.safety.alerts]
            rec.confidences = {f.code: round(f.confidence, 3) for f in result.findings.findings}
            rec.measurements = summarize(result)
            rec.analysis_mode = result.report.mode
            rec.provider = result.report.provider
            rec.signal_path = str(save_signal(result.signal, self.store.signals / f"{rec.id}.npz"))
            rec.review_png = str(await asyncio.to_thread(
                render_review_png, result, self.store.reviews / f"{rec.id}.png"))
            files.append(rec.review_png)
        caption = format_card(rec, result, error)
        sent = await self._send(files, caption)
        rec.review_msg_ids = [m.id for m in sent]
        self.store.upsert(rec)
        for other in ecg[1:]:
            other.status, other.error = "merged", f"merged into {rec.id}"
            self.store.upsert(other)
        log.info("%s: %s (%d photo(s)) -> review message(s) %s", rec.id, rec.status, len(ecg), rec.review_msg_ids)
        return rec

    async def _send(self, files: list[str], caption: str):
        try:
            from telethon.errors import FloodWaitError
        except ImportError:                    # fake client in tests, no telethon installed
            FloodWaitError = ()
        for _attempt in range(3):
            try:
                sent = await self.client.send_file(self.review_chat, files, caption=caption[:1024])
                return sent if isinstance(sent, list) else [sent]
            except FloodWaitError as e:        # Telegram asks to slow down — obey
                log.warning("flood wait %s s", e.seconds)
                await asyncio.sleep(e.seconds + 1)
        raise RuntimeError("could not send the review card")

    async def handle_review(self, msg) -> str | None:
        text = (msg.message or "").strip()
        if text.startswith("/stats"):
            s = self.store.stats()
            reply = ("Собрано {total}: ЭКГ проанализировано {analysed}, не оцифровано {digitise_failed}, "
                     "не ЭКГ {not_ecg}. Ждут проверки {awaiting_review}. Вердикты: верно {confirmed}, "
                     "неверно {rejected}, частично {partial}, комментарии {comment}. "
                     "С метками для обучения: {verified}.").format(**s)
            await msg.reply(reply)
            return reply
        if text.startswith("/help"):
            reply = VERDICT_HELP + "\n\nКоды находок: " + ", ".join(SCORED_LABELS)
            await msg.reply(reply)
            return reply
        if not msg.reply_to_msg_id:
            return None
        rec = self.store.by_review_msg(msg.reply_to_msg_id)
        if rec is None:
            return None
        parse = parse_verdict(text, rec.predicted)
        apply_verdict(rec, parse)
        self.store.upsert(rec)
        fb = feedback_from_verdict(rec, parse, reviewer="telegram-review")
        if fb is not None:
            self.feedback.append(fb)
        if parse.kind == "comment":
            reply = f"Комментарий к {rec.id} сохранён (без метки)."
        else:
            reply = (f"Записано для {rec.id}: {parse.kind}; метка = {', '.join(parse.gold) or '—'}"
                     + (f"; добавлено {', '.join(parse.added)}" if parse.added else "")
                     + (f"; убрано {', '.join(parse.removed)}" if parse.removed else ""))
        await msg.reply(reply)
        return reply


async def _login(cfg: TelegramConfig):
    """Telethon client on the user's account; the first run asks for the phone and the login code."""
    from telethon import TelegramClient

    if not cfg.api_id or not cfg.api_hash:
        raise SystemExit("TELEGRAM_API_ID / TELEGRAM_API_HASH are not set (https://my.telegram.org)")
    Path(cfg.session).parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(cfg.session, cfg.api_id, cfg.api_hash)
    await client.start()
    me = await client.get_me()
    log.info("logged in as %s", me.username or me.id)
    return client


def dialog_lines(dialogs) -> list[str]:
    """One line per group/channel: the id to paste into configs/telegram.yaml, the kind, the title."""
    lines = []
    for d in dialogs:
        if not (getattr(d, "is_group", False) or getattr(d, "is_channel", False)):
            continue
        kind = "group" if getattr(d, "is_group", False) else "channel"
        username = getattr(getattr(d, "entity", None), "username", None)
        handle = f"@{username}" if username else ""
        lines.append(f"{d.id:>16}  {kind:7s}  {d.title}  {handle}".rstrip())
    return lines


async def list_chats(cfg: TelegramConfig) -> list[str]:
    """``python -m cardiolens.telegram chats``: every group and channel the account is in."""
    client = await _login(cfg)
    dialogs = [d async for d in client.iter_dialogs()]
    await client.disconnect()
    return dialog_lines(dialogs)


async def run_agent(cfg: TelegramConfig, *, backfill_only: bool = False) -> None:
    from telethon import events

    if not cfg.groups:
        raise SystemExit("configs/telegram.yaml: `groups` is empty — run `python -m cardiolens.telegram chats` "
                         "to list your groups and paste the ids or @usernames there")

    store = DatasetStore(cfg.store_dir)
    store.root.mkdir(parents=True, exist_ok=True)
    client = await _login(cfg)
    groups = [await client.get_entity(g) for g in cfg.groups]
    review = "me" if cfg.review_chat == "me" else await client.get_entity(cfg.review_chat)
    collector = Collector(client, cfg, store, review)

    if cfg.backfill_limit or backfill_only:
        limit = cfg.backfill_limit or 100
        for g in groups:
            n = 0
            async for m in client.iter_messages(g, limit=limit):
                if is_image_message(m):
                    await collector.handle_photo(m, g)
                    n += 1
            log.info("backfill %s: %d photos seen", getattr(g, "title", g), n)
    if backfill_only:
        await client.disconnect()
        return

    @client.on(events.Album(chats=groups))
    async def on_group_album(event):
        msgs = [m for m in event.messages if is_image_message(m)]
        if msgs:
            try:
                await collector.handle_album(msgs, await event.get_chat())
            except Exception:
                log.exception("album %s/%s failed", event.chat_id, msgs[0].id)

    @client.on(events.NewMessage(chats=groups))
    async def on_group_message(event):
        if is_image_message(event.message) and not getattr(event.message, "grouped_id", None):   # albums: above
            try:
                await collector.handle_photo(event.message, await event.get_chat())
            except Exception:
                log.exception("photo %s/%s failed", event.chat_id, event.message.id)

    @client.on(events.NewMessage(chats=review))
    async def on_review_message(event):
        try:
            await collector.handle_review(event.message)
        except Exception:
            log.exception("review message %s failed", event.message.id)

    log.info("listening to %d group(s); review chat: %s", len(groups), cfg.review_chat)
    await client.send_message(review, "CardioVision-агент запущен · " + time.strftime("%Y-%m-%d %H:%M") + "\n" + VERDICT_HELP)
    await client.run_until_disconnected()
