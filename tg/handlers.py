"""Telegram message and command handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

import re
from config import config
from tg.notifications import notify
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


def _pm_session_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Cambia progetto", callback_data="pm:change_project"),
            InlineKeyboardButton("🛑 Termina sessione", callback_data="pm:end_session"),
        ]
    ])


def _sanitize_pm_reply(text: str) -> str:
    """Keep PM replies readable in Telegram by stripping large code blocks."""
    if not text:
        return text

    text = re.sub(r"```[\s\S]*?```", "[snippet di codice omesso]", text)
    text = re.sub(r"<pre>[\s\S]*?</pre>", "[snippet di codice omesso]", text)
    suspicious_patterns = [
        r"## Regole operative",
        r"## Contesto operativo",
        r"Tool disponibili:",
        r"calls>",
        r"<tool>",
        r"name>",
        r"parameter name=",
        r"PM_CONTEXT",
    ]
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in suspicious_patterns):
        return (
            "Ho generato un output interno non adatto alla chat, quindi lo blocco qui.\n\n"
            "Non considero confermati push o deploy finché non arrivano da tool reali. "
            "Se vuoi, posso verificare ora lo stato reale del progetto su GitHub e Vercel."
        )
    if len(text) > 3000:
        text = text[:3000].rstrip() + "\n\n[risposta abbreviata per leggibilita']"
    return text


async def _start_project_selection(update: Update, user_id: int, pending_message: str) -> None:
    """Open project selector menu for PM 'change project' action."""
    from core.project_registry import project_registry
    from core.project_selector import start_selector

    selectable_projects = await project_registry.list_selectable_projects(limit=30)
    if not selectable_projects:
        await update.effective_message.reply_text(
            "Nessun progetto esistente trovato. Scrivimi la richiesta e partiamo con un nuovo progetto.",
        )
        return

    selector = start_selector(user_id, selectable_projects, pending_message)
    await update.effective_message.reply_text(selector.format_menu(), parse_mode="HTML")


# ── Commands ─────────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 <b>Agent Infrastructure Online</b>\n"
        "Mandami qualsiasi richiesta e ci penso io.\n\n"
        "/websites — scegli un sito da creare o modificare\n"
        "/verify_site &lt;nome&gt; — verifica repo GitHub e deploy Vercel reali\n"
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


async def cmd_websites(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Open the website selector flow for modifying or creating websites."""
    if not _is_authorized(update):
        return

    user_id = update.effective_user.id
    from core.pm_session import end_pm_session
    from core.project_registry import project_registry
    from core.webdev_planner import PLANNING_QUESTIONS, start_session

    end_pm_session(user_id)
    active_projects = await project_registry.list_selectable_projects(limit=30)

    if active_projects:
        await _start_project_selection(update, user_id, pending_message="Gestione sito web")
        return

    ws = start_session(user_id, "Crea un nuovo sito web")
    q = ws.current_question
    total = len(PLANNING_QUESTIONS)
    await update.message.reply_text(
        f"🌐 <b>Nessun sito registrato: partiamo da zero.</b>\n\n"
        f"Ti farò <b>{total} domande rapide</b> per creare il nuovo sito.\n\n"
        f"{ws.progress_bar()}\n\n"
        f"{q['question']}",
        parse_mode="HTML",
    )


async def cmd_verify_site(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Verify a site's real GitHub/Vercel publish state without going through PM."""
    if not _is_authorized(update):
        return
    if _orchestrator is None:
        await update.message.reply_text("⏳ Orchestrator not ready")
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /verify_site <nome_progetto>")
        return

    project_name = " ".join(ctx.args).strip()
    from core.project_registry import project_registry

    project = await project_registry.get_project(project_name)
    if not project:
        await update.message.reply_text(
            f"Progetto non trovato nel registry: <code>{project_name}</code>",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")

    github_tool = _orchestrator.get_tool("github")
    vercel_tool = _orchestrator.get_tool("vercel")
    if github_tool is None or vercel_tool is None:
        await update.message.reply_text(
            "❌ Tool GitHub/Vercel non disponibili nell'orchestrator. Riavvia il servizio e riprova.",
        )
        return

    github_result = await github_tool.execute(
        action="get_repo",
        repo_name=project.name,
    )
    vercel_project = await vercel_tool.execute(
        action="get_project",
        project_name=project.name,
    )
    vercel_deployments = await vercel_tool.execute(
        action="list_deployments",
        project_name=project.name,
    )

    repo_url = "-"
    repo_ok = False
    repo_error = ""
    if github_result.get("success"):
        repo = github_result.get("repo", {})
        full_name = repo.get("full_name", "")
        repo_url = repo.get("html_url", "-")
        repo_ok = full_name.lower() == f"{config.github_owner}/{project.name}".lower()
        if not repo_ok:
            repo_error = f"owner/repo inatteso: {full_name or 'sconosciuto'}"
    else:
        repo_error = github_result.get("error", "repo non trovato")

    deployments = vercel_deployments.get("deployments", []) if vercel_deployments.get("success") else []
    ready_deployment = next(
        (deployment for deployment in deployments if str(deployment.get("state", "")).lower() == "ready"),
        None,
    )
    deploy_url = f"https://{ready_deployment['url']}" if ready_deployment and ready_deployment.get("url") else "-"
    deploy_ok = bool(vercel_project.get("success") and ready_deployment)
    deploy_error = ""
    if not deploy_ok:
        deploy_error = (
            vercel_project.get("error", "") or
            vercel_deployments.get("error", "") or
            "deployment ready non trovata"
        )

    lines = [
        f"🔎 <b>Verifica sito:</b> <code>{project.name}</code>",
        f"GitHub verificato: <b>{'SI' if repo_ok else 'NO'}</b>",
        f"Repo: <code>{repo_url}</code>",
    ]
    if repo_error:
        lines.append(f"Errore GitHub: <code>{repo_error[:300]}</code>")

    lines.extend([
        f"Vercel verificato: <b>{'SI' if deploy_ok else 'NO'}</b>",
        f"Deploy: <code>{deploy_url}</code>",
    ])
    if deploy_error:
        lines.append(f"Errore Vercel: <code>{deploy_error[:300]}</code>")

    await project_registry.upsert_project(
        name=project.name,
        description=project.description,
        workspace_path=project.workspace_path,
        github_repo=f"{config.github_owner}/{project.name}" if repo_ok else None,
        domain=project.domain,
        deploy_provider="vercel",
        deploy_url=deploy_url if deploy_ok else None,
        status=project.status,
        metadata_json=project.metadata_json,
        mark_deployed=deploy_ok,
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

    user_id = update.effective_user.id

    # ── PM session intercept (sticky routing) ─────────────────────────────
    from core.pm_session import end_pm_session, get_pm_session
    pm_session = get_pm_session(user_id)
    if pm_session:
        t = text.strip().lower()
        if t in ("/pm_exit", "/exit", "esci", "termina sessione", "chiudi sessione"):
            end_pm_session(user_id)
            await update.message.reply_text("🛑 Sessione PM terminata.")
            return

        if t in ("cambia progetto", "/cambia_progetto", "/change_project"):
            end_pm_session(user_id)
            await _start_project_selection(update, user_id, pending_message="Modifica progetto esistente")
            return

        await update.message.chat.send_action("typing")
        pm_session.add_user(text)
        pm_response = await _orchestrator.handle_project_modification_sync(
            user_id=user_id,
            project=pm_session.project,
            user_message=text,
            chat_id=update.effective_chat.id,
            history=pm_session.history,
        )
        pm_response = _sanitize_pm_reply(md_bold_to_html(pm_response))
        pm_session.add_assistant(pm_response)
        try:
            await update.message.reply_text(
                pm_response,
                parse_mode="HTML",
                reply_markup=_pm_session_keyboard(),
            )
        except Exception:
            await update.message.reply_text(
                re.sub(r"<[^>]+>", "", pm_response),
                reply_markup=_pm_session_keyboard(),
            )
        return
    # ─────────────────────────────────────────────────────────────────────

    # ── Planning session intercept ────────────────────────────────────────
    from core.webdev_planner import get_session, abort_session
    session = get_session(user_id)
    if session and not session.completed:
        # Special abort command
        if text.strip().lower() in ("/annulla", "/abort", "annulla", "abort", "/cancel"):
            abort_session(user_id)
            await update.message.reply_text("❌ Pianificazione annullata.")
            return

        response = await _handle_planning_answer(session, text, update)
        if response:
            response = md_bold_to_html(response)
            await _reply_html_safe(update.message, response)
        return
    # ─────────────────────────────────────────────────────────────────────

    # ── Project selector intercept ────────────────────────────────────────
    from core.project_selector import get_selector, end_selector, start_selector
    from core.webdev_planner import get_session as _get_ws, start_session, PLANNING_QUESTIONS
    selector = get_selector(user_id)
    if selector:
        # Abort
        if text.strip().lower() in ("/annulla", "/abort", "annulla", "abort", "/cancel"):
            end_selector(user_id)
            await update.message.reply_text("❌ Selezione annullata.")
            return

        result = selector.resolve(text)
        if result is False:
            await update.message.reply_text(
                "Non ho capito. Rispondi con il <b>numero</b> del progetto "
                "o <b>0</b> per crearne uno nuovo.",
                parse_mode="HTML",
            )
            return  # keep selector active

        end_selector(user_id)

        if result is None:
            # New project — start Q&A
            ws = start_session(user_id, selector.pending_message)
            q = ws.current_question
            total = len(PLANNING_QUESTIONS)
            await update.message.reply_text(
                f"🌐 <b>Perfetto! Creiamo un nuovo sito.</b>\n\n"
                f"Ti farò <b>{total} domande rapide</b> e poi partirò subito.\n\n"
                f"Puoi anche mandare un file JSON/TXT con le specifiche "
                f"e salteremo le domande.\n\n"
                f"{ws.progress_bar()}\n\n"
                f"{q['question']}",
                parse_mode="HTML",
            )
        else:
            # Existing project → start sticky PM session
            from core.pm_session import start_pm_session
            start_pm_session(
                user_id=user_id,
                project=result,
                chat_id=update.effective_chat.id,
            )
            await update.message.reply_text(
                f"🧑‍💼 <b>Sessione PM attiva</b> — progetto <b>{result.name}</b>\n"
                "Dimmi ora che azione vuoi fare (es. aggiornare hero, cambiare copy, deploy, fix bug).\n\n"
                f"Richiesta iniziale rilevata: <i>{selector.pending_message[:180]}</i>",
                parse_mode="HTML",
                reply_markup=_pm_session_keyboard(),
            )
        return
    # ─────────────────────────────────────────────────────────────────────

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
        "/websites — selettore siti web (crea/modifica)\n"
        "/verify_site &lt;nome&gt; — verifica stato reale di un sito\n"
        "/cancel &lt;id&gt; — annulla un task\n"
        "/cancel_all — cancella tutti i task attivi/bloccati\n"
        "/force_cancel &lt;id&gt; — forza cancellazione task bloccato\n"
        "/logs — lista file di log\n"
        "/log &lt;file&gt; — scarica un log\n"
        "/jobs — lista job schedulati\n"
        "/job_enable &lt;nome&gt; — abilita job\n"
        "/job_disable &lt;nome&gt; — disabilita job\n"
        "/pm_exit — termina la sessione PM attiva\n"
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

    data = query.data or ""

    if data == "pm:end_session":
        from core.pm_session import end_pm_session
        end_pm_session(update.effective_user.id)
        await query.answer("Sessione PM terminata")
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("🛑 Sessione PM terminata. Scrivimi pure una nuova richiesta.")
        return

    if data == "pm:change_project":
        from core.pm_session import end_pm_session
        end_pm_session(update.effective_user.id)
        await query.answer("Cambio progetto")
        await query.edit_message_reply_markup(reply_markup=None)
        await _start_project_selection(update, update.effective_user.id, pending_message="Modifica progetto esistente")
        return

    # approval callbacks: "approve:123" / "reject:123"
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
        await notify(
            f"⏰ <b>Approvazione scaduta</b> (task #{task_id})\n"
            f"Nessuna risposta ricevuta entro {int(timeout)}s.\n"
            "Il task è stato sospeso automaticamente. "
            "Puoi riprendere inviando un nuovo messaggio."
        )
        return False


# ── Planning session helpers ────────────────────────────────────────────────


async def _handle_planning_answer(session, text: str, update: Update) -> str | None:
    """Process a user answer in the context of an active planning session.

    Returns the next question string, or None if finalization was triggered.
    """
    session.record_answer(text)

    if session.current_question is None:
        # All questions answered — finalize
        await update.message.reply_text(
            "⏳ <b>Perfetto!</b> Sto generando le specifiche e il design system…\n"
            "(può richiedere 20-30 secondi)",
            parse_mode="HTML",
        )
        from core.webdev_planner import end_session
        try:
            result = await session.finalize()
            end_session(session.user_id)

            # Show specs summary
            specs_summary = session.format_specs_summary()
            await update.message.reply_text(specs_summary, parse_mode="HTML")

            # Trigger the full webdev pipeline with specs
            if _orchestrator:
                await _orchestrator.handle_webdev_task(
                    user_id=session.user_id,
                    initial_message=session.initial_message,
                    specs=result["specs"],
                    design_system=result["design_system"],
                    media_files=result["media"],
                    chat_id=update.effective_chat.id,
                    state_files={
                        "project_md": result.get("project_md", ""),
                        "requirements_md": result.get("requirements_md", ""),
                        "pm_context": result.get("pm_context", ""),
                    },
                )
        except Exception as exc:
            log.error(f"Planning finalization error: {exc}", exc_info=True)
            await update.message.reply_text(f"❌ Errore nella generazione delle specifiche: {exc}")
        return None

    # More questions to ask
    q = session.current_question
    return f"{session.progress_bar()}\n\n{q['question']}"


# ── File/photo handlers ─────────────────────────────────────────────────


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos — route to planning session or orchestrator."""
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

    user_id = update.effective_user.id

    # ── Planning session: inspiration image ──────────────────────────────
    from core.webdev_planner import get_session
    session = get_session(user_id)
    if session and not session.completed and session.accepts_media_now():
        await update.message.reply_text("🔍 Analizzo l'immagine…")
        insight = await session.add_media_with_analysis(str(file_path))
        caption = update.message.caption or ""
        if caption:
            session.answers.setdefault("inspiration", "")
            session.answers["inspiration"] = (
                session.answers["inspiration"] + "\n" + caption
            ).strip()
        insight_summary = ""
        if insight:
            try:
                import json as _json
                ins = _json.loads(insight[insight.find("{"):insight.rfind("}") + 1])
                insight_summary = (
                    f"\n✅ <b>Immagine analizzata:</b> {ins.get('style', '')} — "
                    f"colori: {', '.join(ins.get('colors', [])[:3])}"
                )
            except Exception:
                pass
        await update.message.reply_text(
            f"📸 Immagine salvata come riferimento.{insight_summary}\n\n"
            "Puoi mandarne altre, oppure rispondi alla domanda per continuare.",
            parse_mode="HTML",
        )
        return
    # ─────────────────────────────────────────────────────────────────────

    caption = update.message.caption or "Immagine ricevuta, cosa devo fare?"
    response = await _orchestrator.handle_user_message(
        user_id=user_id,
        message=caption,
        chat_id=update.effective_chat.id,
        attachments=[str(file_path)],
    )
    if response:
        await _reply_html_safe(update.message, response)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle incoming documents/files — route to planning session or orchestrator."""
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

    user_id = update.effective_user.id

    # ── Planning session: specs file upload ──────────────────────────────
    from core.webdev_planner import get_session, end_session, PLANNING_QUESTIONS
    session = get_session(user_id)
    if session and not session.completed:
        fname = doc.file_name or ""
        if fname.endswith((".json", ".txt", ".md")):
            try:
                content = Path(str(file_path)).read_text(encoding="utf-8", errors="replace")
                await update.message.reply_text(
                    f"📄 File specs ricevuto: <code>{fname}</code>\n"
                    "Uso questo come specifiche di progetto, genero il design system…",
                    parse_mode="HTML",
                )
                # Override the initial message with file content
                session.initial_message = content[:3000]
                # Skip remaining questions and finalize immediately
                session.current_question_idx = len(PLANNING_QUESTIONS)  # mark all done
                # Fill remaining answers as "vedi file specs"
                for q in PLANNING_QUESTIONS:
                    session.answers.setdefault(q["key"], "Vedi file specs allegato")

                result = await session.finalize()
                end_session(user_id)
                specs_summary = session.format_specs_summary()
                await update.message.reply_text(specs_summary, parse_mode="HTML")

                if _orchestrator:
                    await _orchestrator.handle_webdev_task(
                        user_id=user_id,
                        initial_message=session.initial_message,
                        specs=result["specs"],
                        design_system=result["design_system"],
                        media_files=result["media"],
                        chat_id=update.effective_chat.id,
                        state_files={
                            "project_md": result.get("project_md", ""),
                            "requirements_md": result.get("requirements_md", ""),
                            "pm_context": result.get("pm_context", ""),
                        },
                    )
                return
            except Exception as exc:
                await update.message.reply_text(f"❌ Errore lettura file: {exc}")
                return
    # ─────────────────────────────────────────────────────────────────────

    caption = update.message.caption or f"File ricevuto: {doc.file_name}. Cosa devo fare?"
    response = await _orchestrator.handle_user_message(
        user_id=user_id,
        message=caption,
        chat_id=update.effective_chat.id,
        attachments=[str(file_path)],
    )
    if response:
        await _reply_html_safe(update.message, response)
