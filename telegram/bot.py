"""Telegram bot setup and lifecycle."""

from __future__ import annotations

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import config
from telegram.handlers import (
    cmd_cancel,
    cmd_costs,
    cmd_start,
    cmd_status,
    cmd_tasks,
    cmd_tools,
    handle_callback,
    handle_document,
    handle_message,
    handle_photo,
)
from telegram.notifications import set_app
from utils.logging import setup_logging

log = setup_logging("tg_bot")


def build_app() -> Application:
    """Build the Telegram Application (does NOT start polling yet)."""
    app = (
        Application.builder()
        .token(config.telegram_bot_token)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("costs", cmd_costs))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("cancel", cmd_cancel))

    # Callback queries (approval buttons)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Content handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Init notification system
    set_app(app, config.allowed_telegram_user_id)

    log.info("Telegram bot built successfully")
    return app
