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
    # Italics: _text_ → <i>text</i> (but not __dunder__)
    text = re.sub(r"(?<![_])_([^_]+?)_(?![_])", r"<i>\1</i>", text)
    # Inline code: `text` → <code>text</code>
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    # Code blocks: ```...``` → <pre>...</pre>
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    return text


def _is_authorized(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id == config.allowed_telegram_user_id


async def _reply_html_safe(message_obj, text: str):
    """Send long responses safely to Telegram with fallback to plain text."""
    if not text:
        return

    max_len = 3500
    chunks = []
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

    for chunk in chunks:
        try:
            await message_obj.reply_text(chunk, parse_mode="HTML")
        except Exception:
            await message_obj.reply_text(chunk)


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
        "/projects — lista progetti registrati\n"
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


async def cmd_projects(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List recent projects from the persistent project registry."""
    if not _is_authorized(update):
        return

    from core.project_registry import project_registry

    rows = await project_registry.list_projects(limit=20)
    if not rows:
        await update.message.reply_text("Nessun progetto registrato.")
        return

    lines = ["📦 <b>Progetti registrati:</b>"]
    for r in rows:
        lines.append(
            f"• <b>{r.name}</b> [{r.status}]\n"
            f"  repo: <code>{r.github_repo or '-'}</code>\n"
            f"  domain: <code>{r.domain or '-'}</code>\n"
            f"  deploy: <code>{r.deploy_url or '-'}</code>"
        )
    await _reply_html_safe(update.message, "\n".join(lines))


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


async def cmd_force_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Force-cancel a stuck task by directly updating DB status."""
    if not _is_authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /force_cancel <task_id>")
        return
    try:
        task_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("ID task non valido.")
        return

    from core.task_manager import task_manager
    from db.models import TaskStatus

    # Cancel asyncio task if exists
    atask = task_manager._running_tasks.pop(task_id, None)
    if atask and not atask.done():
        atask.cancel()

    # Clear any pending approval
    future = _pending_approvals.pop(task_id, None)
    if future and not future.done():
        future.set_result(False)

    # Force DB status update
    await task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
    await update.message.reply_text(
        f"🔧 Task #{task_id} forzatamente cancellato.\n"
        f"Asyncio task: {'annullato' if atask else 'non trovato'}\n"
        f"Status DB: CANCELLED"
    )


async def cmd_cancel_all(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Cancel ALL active/stuck tasks at once."""
    if not _is_authorized(update):
        return
    from core.task_manager import task_manager
    # Also reject all pending approvals
    for task_id, future in list(_pending_approvals.items()):
        if not future.done():
            future.set_result(False)
    _pending_approvals.clear()
    count = await task_manager.cancel_all_active()
    await update.message.reply_text(
        f"🗑️ <b>{count} task cancellati.</b>"
        if count else "✅ Nessun task attivo da cancellare.",
        parse_mode="HTML",
    )


async def cmd_budget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show/set cost budgets."""
    if not _is_authorized(update):
        return

    if ctx.args and len(ctx.args) >= 2:
        budget_type = ctx.args[0].lower()
        try:
            value = float(ctx.args[1])
        except ValueError:
            await update.message.reply_text("Valore non valido. Uso: /budget [task|daily] <valore>")
            return

        if budget_type == "task":
            cost_tracker.task_budget = value
            await update.message.reply_text(f"✅ Budget per task: ${value:.2f}")
        elif budget_type == "daily":
            cost_tracker.daily_budget = value
            await update.message.reply_text(f"✅ Budget giornaliero: ${value:.2f}")
        else:
            await update.message.reply_text("Tipo non valido. Uso: /budget [task|daily] <valore>")
        return

    # Show current budgets
    daily_cost = cost_tracker.get_daily_cost()
    await update.message.reply_text(
        f"💰 <b>Budget attuale:</b>\n"
        f"Per task: <b>${cost_tracker.task_budget:.2f}</b>\n"
        f"Giornaliero: <b>${cost_tracker.daily_budget:.2f}</b>\n\n"
        f"📊 <b>Spesa di oggi:</b> ${daily_cost:.2f} / ${cost_tracker.daily_budget:.2f}\n"
        f"Totale sessione: ${cost_tracker.total_cost:.2f}",
        parse_mode="HTML",
    )


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
        await _reply_html_safe(update.message, response)

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


# ── Scheduler commands ───────────────────────────────────────────────────


async def cmd_jobs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List all scheduled jobs."""
    if not _is_authorized(update):
        return
    from core.scheduler import scheduler
    jobs = await scheduler.list_jobs()
    if not jobs:
        await update.message.reply_text("Nessun job schedulato.")
        return
    lines = ["📅 <b>Job schedulati:</b>"]
    for j in jobs:
        status = "✅" if j["enabled"] else "⏸️"
        interval_min = j["interval"] // 60
        lines.append(
            f"{status} <b>{j['name']}</b> — ogni {interval_min}min\n"
            f"    Runs: {j['runs']} | Ultimo: {j['last_run']} | Prossimo: {j['next_run']}\n"
            f"    {j['last_result']}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_job_enable(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /job_enable <nome_job>")
        return
    from core.scheduler import scheduler
    found = await scheduler.set_enabled(ctx.args[0], True)
    if found:
        await update.message.reply_text(f"✅ Job <b>{ctx.args[0]}</b> abilitato.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"Job '{ctx.args[0]}' non trovato.")


async def cmd_job_disable(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /job_disable <nome_job>")
        return
    from core.scheduler import scheduler
    found = await scheduler.set_enabled(ctx.args[0], False)
    if found:
        await update.message.reply_text(f"⏸️ Job <b>{ctx.args[0]}</b> disabilitato.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"Job '{ctx.args[0]}' non trovato.")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 <b>Comandi disponibili:</b>\n\n"
        "/start — info iniziale\n"
        "/status — stato agenti e task\n"
        "/tasks — lista task attivi\n"
        "/costs — riepilogo costi API\n"
        "/budget — mostra budget (o /budget task|daily X)\n"
        "/tools — lista tool disponibili\n"
        "/projects — lista progetti registrati\n"
        "/cancel &lt;id&gt; — annulla un task\n"
        "/cancel_all — cancella tutti i task attivi/bloccati\n"
        "/force_cancel &lt;id&gt; — forza cancellazione task bloccato\n"
        "/logs — lista file di log\n"
        "/log &lt;file&gt; — scarica un log\n"
        "/jobs — lista job schedulati\n"
        "/job_enable &lt;nome&gt; — abilita job\n"
        "/job_disable &lt;nome&gt; — disabilita job\n"
        "/help — questo messaggio\n\n"
        "Oppure scrivi qualsiasi richiesta in linguaggio naturale!",
        parse_mode="HTML",
    )


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
        await _reply_html_safe(update.message, response)


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
        await _reply_html_safe(update.message, response)
