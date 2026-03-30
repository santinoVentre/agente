"""Proactive Telegram notifications sent by agents during task execution."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from utils.logging import setup_logging

if TYPE_CHECKING:
    from telegram.ext import Application

log = setup_logging("notifications")

# Set by bot.py at startup
_app: Application | None = None
_chat_id: int | None = None

MAX_TG_TEXT = 3500
RETRY_ATTEMPTS = 3


def set_app(app: "Application", chat_id: int):
    global _app, _chat_id
    _app = app
    _chat_id = chat_id


def _chunk_text(text: str, max_len: int = MAX_TG_TEXT) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    return chunks


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


async def _retry(coro_factory, label: str):
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            log.warning(f"{label} failed (attempt {attempt}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(0.5 * attempt)
    if last_error:
        raise last_error


async def _send_text_message(text: str, parse_mode: str = "HTML"):
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, dropping message")
        return

    for chunk in _chunk_text(text):
        try:
            await _retry(
                lambda c=chunk: _app.bot.send_message(
                    chat_id=_chat_id,
                    text=c,
                    parse_mode=parse_mode,
                ),
                "send_message",
            )
        except Exception as e:
            if parse_mode == "HTML":
                plain_chunk = _strip_html(chunk)
                try:
                    await _retry(
                        lambda c=plain_chunk: _app.bot.send_message(
                            chat_id=_chat_id,
                            text=c,
                            parse_mode=None,
                        ),
                        "send_message_plain_fallback",
                    )
                except Exception as e2:
                    log.error(f"Failed to send notification chunk: {e2}")
            else:
                log.error(f"Failed to send notification chunk: {e}")


async def notify(text: str, parse_mode: str = "HTML"):
    """Send a proactive notification to the owner."""
    await _send_text_message(text, parse_mode=parse_mode)


async def notify_progress(task_desc: str, progress: int):
    bar_len = 10
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    await notify(f"🔄 <b>{task_desc}</b>\n{bar} {progress}%")


async def notify_done(task_desc: str, details: str = ""):
    msg = f"✅ <b>{task_desc}</b> completato"
    if details:
        msg += f"\n{details}"
    await notify(msg)


async def notify_error(task_desc: str, error: str):
    await notify(f"❌ <b>{task_desc}</b> fallito\n<code>{error[:500]}</code>")


async def send_file(file_path: str, caption: str = ""):
    """Send a file/document to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, cannot send file")
        return
    try:
        await _retry(
            lambda: _app.bot.send_document(
                chat_id=_chat_id,
                document=file_path,
                caption=caption or None,
                parse_mode="HTML",
            ),
            "send_document",
        )
    except Exception as e:
        log.error(f"Failed to send file: {e}")


async def send_photo(file_path: str, caption: str = ""):
    """Send a photo to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, cannot send photo")
        return
    try:
        await _retry(
            lambda: _app.bot.send_photo(
                chat_id=_chat_id,
                photo=file_path,
                caption=caption or None,
                parse_mode="HTML",
            ),
            "send_photo",
        )
    except Exception as e:
        log.error(f"Failed to send photo: {e}")


async def send_video(file_path: str, caption: str = ""):
    """Send a video to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, cannot send video")
        return
    try:
        await _retry(
            lambda: _app.bot.send_video(
                chat_id=_chat_id,
                video=file_path,
                caption=caption or None,
                parse_mode="HTML",
            ),
            "send_video",
        )
    except Exception as e:
        log.error(f"Failed to send video: {e}")


async def update_message(message_id: int, text: str):
    """Edit an existing message."""
    if _app is None or _chat_id is None:
        return
    try:
        await _retry(
            lambda: _app.bot.edit_message_text(
                chat_id=_chat_id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
            ),
            "edit_message_text",
        )
    except Exception as e:
        log.error(f"Failed to update message: {e}")


async def notify_approval_needed(action_desc: str, task_id: int | None = None):
    """Ask user for approval via inline keyboard. Returns message for callback handling."""
    if _app is None or _chat_id is None:
        return
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    kbd = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approva", callback_data=f"approve:{task_id}"),
            InlineKeyboardButton("❌ Rifiuta", callback_data=f"reject:{task_id}"),
        ]
    ])
    try:
        await _retry(
            lambda: _app.bot.send_message(
                chat_id=_chat_id,
                text=f"⚠️ <b>Approvazione richiesta</b>\n{action_desc}",
                reply_markup=kbd,
                parse_mode="HTML",
            ),
            "send_approval_message",
        )
    except Exception as e:
        log.error(f"Failed to send approval request: {e}")
