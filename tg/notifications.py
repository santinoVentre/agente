"""Proactive Telegram notifications sent by agents during task execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.logging import setup_logging

if TYPE_CHECKING:
    from telegram.ext import Application

log = setup_logging("notifications")

# Set by bot.py at startup
_app: Application | None = None
_chat_id: int | None = None


def set_app(app: "Application", chat_id: int):
    global _app, _chat_id
    _app = app
    _chat_id = chat_id


async def notify(text: str, parse_mode: str = "HTML"):
    """Send a proactive notification to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, dropping message")
        return
    try:
        await _app.bot.send_message(chat_id=_chat_id, text=text, parse_mode=parse_mode)
    except Exception as e:
        log.error(f"Failed to send notification: {e}")


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
        await _app.bot.send_document(
            chat_id=_chat_id,
            document=file_path,
            caption=caption or None,
            parse_mode="HTML",
        )
    except Exception as e:
        log.error(f"Failed to send file: {e}")


async def send_photo(file_path: str, caption: str = ""):
    """Send a photo to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, cannot send photo")
        return
    try:
        await _app.bot.send_photo(
            chat_id=_chat_id,
            photo=file_path,
            caption=caption or None,
            parse_mode="HTML",
        )
    except Exception as e:
        log.error(f"Failed to send photo: {e}")


async def send_video(file_path: str, caption: str = ""):
    """Send a video to the owner."""
    if _app is None or _chat_id is None:
        log.warning("Notification system not initialized, cannot send video")
        return
    try:
        await _app.bot.send_video(
            chat_id=_chat_id,
            video=file_path,
            caption=caption or None,
            parse_mode="HTML",
        )
    except Exception as e:
        log.error(f"Failed to send video: {e}")


async def update_message(message_id: int, text: str):
    """Edit an existing message."""
    if _app is None or _chat_id is None:
        return
    try:
        await _app.bot.edit_message_text(
            chat_id=_chat_id,
            message_id=message_id,
            text=text,
            parse_mode="HTML",
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
    await _app.bot.send_message(
        chat_id=_chat_id,
        text=f"⚠️ <b>Approvazione richiesta</b>\n{action_desc}",
        reply_markup=kbd,
        parse_mode="HTML",
    )
