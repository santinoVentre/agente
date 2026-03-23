"""Telegram message and command handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import ContextTypes

import re
from config import config
from utils.logging import setup_logging
from utils.cost_tracker import cost_tracker
from pathlib import Path

if TYPE_CHECKING:
    from core.orchestrator import Orchestrator

log = setup_logging("tg_handlers")

# Set by bot.py at startup
_orchestrator: "Orchestrator | None" = None

# Pending approval futures: task_id → asyncio.Future[bool]
_pending_approvals: dict[int, asyncio.Future[bool]] = {}


def set_orchestrator(orch: "Orchestrator"):
    global _orchestrator
    _orchestrator = orch

def md_bold_to_html(text: str) -> str:
    # Replace **text** or *text* with <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<b>\1</b>", text)
    return text


def _is_authorized(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id == config.allowed_telegram_user_id


# ── Commands ─────────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 <b>Agent Infrastructure Online</b>\n"
        "Mandami qualsiasi richiesta e ci penso io.\n\n"
        "/status — stato agenti e task\n"
        "/tasks — lista task attivi\n"
        "/costs — riepilogo costi API\n"
        "/tools — lista tool disponibili\n"
        "/cancel &lt;id&gt; — annulla un task",
        parse_mode="HTML",
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Orchestrator not ready")
        return
    status = await _orchestrator.get_status()
    await update.message.reply_text(status, parse_mode="HTML")


async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Orchestrator not ready")
        return
    tasks_text = await _orchestrator.get_active_tasks_text()
    await update.message.reply_text(tasks_text or "Nessun task attivo.", parse_mode="HTML")


async def cmd_costs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await update.message.reply_text(cost_tracker.format_summary())


async def cmd_tools(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Orchestrator not ready")
        return
    tools_text = await _orchestrator.get_tools_text()
    await update.message.reply_text(tools_text, parse_mode="HTML")


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /cancel <task_id>")
        return
    try:
        task_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("ID task non valido.")
        return
    if _orchestrator:
        result = await _orchestrator.cancel_task(task_id)
        await update.message.reply_text(result)


# ── Messages ─────────────────────────────────────────────────────────────


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Main handler: forwards user message to the orchestrator."""
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Sistema in avvio, riprova tra poco...")
        return

    text = update.message.text or update.message.caption or ""
    if not text.strip():
        await update.message.reply_text("Inviami un messaggio di testo.")
        return

    # Show typing indicator
    await update.message.chat.send_action("typing")

    # Handle via orchestrator (may be long-running → sends progress via notifications)
    response = await _orchestrator.handle_user_message(
        user_id=update.effective_user.id,
        message=text,
        chat_id=update.effective_chat.id,
    )
    if response:
        # Convert markdown bold to HTML bold
        response = md_bold_to_html(response)
        # Split long messages (Telegram limit: 4096 chars)
        for i in range(0, len(response), 4000):
            chunk = response[i : i + 4000]
            await update.message.reply_text(chunk, parse_mode="HTML")

# ── Logs command ─────────────────────────────────────────────────
async def cmd_logs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    logs_dir = config.logs_dir
    log_files = sorted(Path(logs_dir).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        await update.message.reply_text("Nessun file di log trovato.")
        return
    msg = "<b>Log files disponibili:</b>\n"
    for i, f in enumerate(log_files[:10], 1):
        msg += f"{i}. <code>{f.name}</code> ({f.stat().st_size//1024} KB)\n"
    msg += "\nRispondi con /log <nomefile> per riceverlo."
    await update.message.reply_text(msg, parse_mode="HTML")

async def cmd_log(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /log <nomefile.log>")
        return
    logs_dir = config.logs_dir
    fname = ctx.args[0]
    fpath = Path(logs_dir) / fname
    if not fpath.exists() or not fpath.is_file():
        await update.message.reply_text("File di log non trovato.")
        return
    await update.message.reply_document(document=str(fpath), filename=fpath.name)


# ── Callback queries (approve/reject) ───────────────────────────────────


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not _is_authorized(update):
        await query.answer("Non autorizzato.")
        return

    data = query.data  # "approve:123" or "reject:123"
    parts = data.split(":")
    if len(parts) != 2:
        await query.answer("Formato non valido.")
        return

    action, task_id_str = parts
    try:
        task_id = int(task_id_str)
    except ValueError:
        await query.answer("ID non valido.")
        return

    approved = action == "approve"
    future = _pending_approvals.pop(task_id, None)
    if future and not future.done():
        future.set_result(approved)

    emoji = "✅" if approved else "❌"
    await query.answer(f"{emoji} {'Approvato' if approved else 'Rifiutato'}")
    await query.edit_message_text(
        f"{emoji} Azione {'approvata' if approved else 'rifiutata'} (task #{task_id})"
    )


async def request_approval(task_id: int, timeout: float = 300.0) -> bool:
    """Block until user approves/rejects via Telegram. Returns True if approved."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    _pending_approvals[task_id] = future
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        _pending_approvals.pop(task_id, None)
        return False


# ── File/photo handlers ─────────────────────────────────────────────────


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos — pass to orchestrator with caption as instructions."""
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Sistema in avvio...")
        return

    photo = update.message.photo[-1]  # highest resolution
    file = await photo.get_file()
    file_path = config.media_dir / f"tg_{photo.file_unique_id}.jpg"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    await file.download_to_drive(str(file_path))

    caption = update.message.caption or "Immagine ricevuta, cosa devo fare?"
    response = await _orchestrator.handle_user_message(
        user_id=update.effective_user.id,
        message=caption,
        chat_id=update.effective_chat.id,
        attachments=[str(file_path)],
    )
    if response:
        await update.message.reply_text(response, parse_mode="HTML")


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle incoming documents/files."""
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Sistema in avvio...")
        return

    doc = update.message.document
    file = await doc.get_file()
    file_path = config.media_dir / f"tg_{doc.file_unique_id}_{doc.file_name}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    await file.download_to_drive(str(file_path))

    caption = update.message.caption or f"File ricevuto: {doc.file_name}. Cosa devo fare?"
    response = await _orchestrator.handle_user_message(
        user_id=update.effective_user.id,
        message=caption,
        chat_id=update.effective_chat.id,
        attachments=[str(file_path)],
    )
    if response:
        await update.message.reply_text(response, parse_mode="HTML")
