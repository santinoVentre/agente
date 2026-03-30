"""Orchestrator — the central brain that routes requests to sub-agents."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any

from config import config
from core.memory import memory
from core.inventory import get_latest_snapshot_summary
from core.model_router import CLASSIFICATION_PROMPT, TaskType, get_model_for_task
from core.openrouter_client import openrouter
from core.project_registry import project_registry
from core.task_manager import task_manager
from db.models import TaskStatus
from tg.notifications import notify, notify_done, notify_error
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("orchestrator")


class Orchestrator:
    """
    Central orchestrator: always-alive process that:
    1. Receives user messages from Telegram
    2. Classifies intent → picks task type + model
    3. Delegates to the appropriate sub-agent
    4. Can answer status questions while tasks run in background
    """

    def __init__(self):
        self._agents: dict[str, Any] = {}
        self._tools: dict[str, Any] = {}

    def register_agent(self, agent):
        self._agents[agent.name] = agent
        log.info(f"Registered agent: {agent.name}")

    def register_tool(self, tool):
        self._tools[tool.name] = tool

    # ── Intent classification ────────────────────────────────────────

    async def _classify_intent(self, message: str) -> TaskType:
        """Use a cheap model to classify the user's intent."""
        model = get_model_for_task(TaskType.ROUTING)
        response = await openrouter.chat(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()

        # Map string to TaskType
        mapping = {t.value: t for t in TaskType}
        task_type = mapping.get(raw, TaskType.SIMPLE_CHAT)
        log.info(f"Classified intent: '{message[:60]}...' → {task_type.value}")
        return task_type

    def _pick_agent(self, task_type: TaskType) -> str:
        """Pick the best sub-agent for a task type."""
        agent_map = {
            TaskType.WEB_DEV: "webdev",
            TaskType.MEDIA: "media",
            TaskType.CODE_GENERATION: "system",
            TaskType.CODE_REVIEW: "system",
            TaskType.COMPLEX_REASONING: "system",
        }
        agent_name = agent_map.get(task_type, "system")
        if agent_name not in self._agents:
            agent_name = next(iter(self._agents), "system")
        return agent_name

    def _task_memory_key(self, task_id: int, user_message: str) -> str:
        """Build a readable key for long-term task memory entries."""
        raw = user_message.strip().lower()[:80]
        slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if not slug:
            slug = "attivita"
        slug = slug[:48]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{slug}_{ts}"

    # ── Main handler ─────────────────────────────────────────────────

    async def handle_user_message(
        self,
        user_id: int,
        message: str,
        chat_id: int,
        attachments: list[str] | None = None,
    ) -> str:
        """Process a user message end-to-end."""

        # Save user message
        await memory.save_message(user_id, "user", message)

        # Check if it's a status query (respond immediately, no task creation)
        if await self._is_status_query(message):
            return await self._handle_status_query(message, user_id)

        # Classify intent
        task_type = await self._classify_intent(message)

        # Check if it's a tool creation request
        if self._is_tool_creation_request(message, task_type):
            task_type = TaskType.CODE_GENERATION

        # Pick agent and model
        agent_name = self._pick_agent(task_type)
        model = get_model_for_task(task_type)
        agent = self._agents.get(agent_name)

        if not agent:
            return "⚠️ Nessun agente disponibile per gestire questa richiesta."

        # Create task
        task = await task_manager.create_task(
            user_id=user_id,
            description=message[:500],
            task_type=task_type.value,
            agent=agent_name,
            model=model,
        )

        await task_manager.update_task_status(task.id, TaskStatus.IN_PROGRESS, progress=5)

        # Build context
        history = await memory.get_conversation_history(user_id, limit=10)
        memories = await memory.recall_all(user_id)
        memory_context = memory.format_memories_for_prompt(memories)

        system_prompt = await self._build_system_prompt(agent_name, memory_context, attachments)

        # For quick tasks, run synchronously; for complex ones, run in background
        if task_type in (TaskType.SIMPLE_CHAT, TaskType.ROUTING, TaskType.SUMMARIZATION):
            return await self._run_sync(agent, message, task, history, system_prompt, model)
        else:
            return await self._run_async(agent, message, task, history, system_prompt, model)

    async def _run_sync(self, agent, message, task, history, system_prompt, model) -> str:
        """Run agent synchronously — return response directly."""
        try:
            response = await agent.run(
                user_message=message,
                task_id=task.id,
                history=history,
                system_prompt=system_prompt,
                model_override=model,
            )
            await task_manager.complete_task(task.id, cost=cost_tracker.total_cost)
            await memory.save_message(task.user_id, "assistant", response, model=model)
            await memory.remember(
                task.user_id,
                key=self._task_memory_key(task.id, message),
                value=(
                    f"type={task.task_type}; agent={task.agent_assigned}; "
                    f"request={message[:240]}; outcome={response[:600]}"
                ),
                category="task_history",
            )
            return response
        except Exception as e:
            log.error(f"Task #{task.id} failed: {e}", exc_info=True)
            await task_manager.fail_task(task.id, str(e))
            return f"❌ Errore: {str(e)[:500]}"

    async def _run_async(self, agent, message, task, history, system_prompt, model) -> str:
        """Launch agent in background — return acknowledgement immediately."""

        async def _background():
            try:
                response = await agent.run(
                    user_message=message,
                    task_id=task.id,
                    history=history,
                    system_prompt=system_prompt,
                    model_override=model,
                )
                await task_manager.complete_task(task.id, cost=cost_tracker.total_cost)
                await memory.save_message(task.user_id, "assistant", response, model=model)
                await memory.remember(
                    task.user_id,
                    key=self._task_memory_key(task.id, message),
                    value=(
                        f"type={task.task_type}; agent={task.agent_assigned}; "
                        f"request={message[:240]}; outcome={response[:600]}"
                    ),
                    category="task_history",
                )
                await notify_done(message[:60], response)
            except Exception as e:
                log.error(f"Task #{task.id} failed: {e}", exc_info=True)
                await task_manager.fail_task(task.id, str(e))
                await notify_error(message[:60], str(e))
            finally:
                task_manager.unregister_running_task(task.id)

        atask = asyncio.create_task(_background())
        task_manager.register_running_task(task.id, atask)

        # Keep Telegram quiet for operational startup messages.
        return ""

    # ── Status queries ───────────────────────────────────────────────

    async def _is_status_query(self, message: str) -> bool:
        keywords = [
            "a che punto", "stato", "status", "come va", "cosa stai facendo",
            "task attivi", "progresso", "finito", "completato",
        ]
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in keywords)

    async def _handle_status_query(self, message: str, user_id: int) -> str:
        tasks = await task_manager.get_active_tasks()
        if not tasks:
            return "✅ Nessun task attivo al momento. Sono in attesa di istruzioni."

        lines = ["📋 <b>Task attivi:</b>"]
        for t in tasks:
            status_data = await task_manager.get_task_status_from_redis(t.id)
            progress = status_data.get("progress", "?") if status_data else "?"
            emoji = {"pending": "⏳", "in_progress": "🔄", "waiting_approval": "⚠️"}.get(t.status.value, "❓")
            lines.append(
                f"{emoji} <b>#{t.id}</b> [{t.task_type}] {t.description[:60]}... — {progress}%"
            )
        return "\n".join(lines)

    # ── Tool creation detection ──────────────────────────────────────

    def _is_tool_creation_request(self, message: str, task_type: TaskType) -> bool:
        keywords = [
            "crea un tool", "nuovo tool", "crea tool", "creare un tool",
            "aggiungi tool", "modifica tool", "cambia tool",
            "create a tool", "new tool", "add tool",
        ]
        return any(kw in message.lower() for kw in keywords)

    # ── System prompt ────────────────────────────────────────────────

    async def _build_system_prompt(
        self, agent_name: str, memory_context: str, attachments: list[str] | None
    ) -> str:
        try:
            infra_summary = await get_latest_snapshot_summary()
        except Exception as e:
            log.warning(f"Failed to load infrastructure snapshot summary: {e}")
            infra_summary = "Snapshot infrastrutturale non disponibile"

        try:
            projects_summary = await project_registry.get_recent_projects_summary(limit=6)
        except Exception as e:
            log.warning(f"Failed to load projects summary: {e}")
            projects_summary = "Registro progetti non disponibile"

        parts = [
            # ── Identità ──
            "Sei l'agente personale di Santino. Sei un sistema autonomo che vive su un "
            "server VPS Linux (Ubuntu 24.04, 4 vCPU, 4GB RAM, 120GB NVMe, IP 87.106.245.253, IONOS).",
            "Stai comunicando con Santino ESCLUSIVAMENTE tramite le API di Telegram. "
            "La conversazione avviene dentro il bot Telegram: tu ricevi i messaggi e rispondi tramite l'API.",

            # ── Infrastruttura ──
            "\n== INFRASTRUTTURA ==",
            "- Il tuo codice sorgente si trova in: /srv/agent/app/",
            "- Puoi LEGGERE tutto il tuo codice sorgente con il tool filesystem (read, list) o con shell (cat, ls).",
            "- I tuoi file CORE protetti da modifica diretta (richiedono approvazione) sono: "
            "orchestrator.py, security_agent.py, tool_validator.py, config.py, db/models.py",
            "- Puoi SCRIVERE liberamente in: workspaces, media, tools/custom, logs, /tmp",
            "- La tua configurazione (.env) è in: /srv/agent/app/.env — contiene token e segreti.",
            "- Database PostgreSQL 16 su localhost:5432 (Docker container 'agent-postgres')",
            "- Redis 7 su localhost:6379 (Docker container 'agent-redis')",
            "- Sei eseguito come utente 'agent' via systemd (servizio: agent.service)",
            "- SUDO LIMITATO: hai accesso sudo SENZA password per comandi specifici:",
            "  • sudo apt update / sudo apt install -y <pacchetto>",
            "  • sudo systemctl <start|stop|restart|reload|status> <servizio>",
            "  • sudo cp /tmp/agent-* <destinazione> (per copiare config da /tmp a /etc/)",
            "  • sudo ln -sf <sorgente> <link> (per link simbolici, es. nginx sites-enabled)",
            "  • sudo rm /etc/nginx/sites-enabled/<file> (per disabilitare siti nginx)",
            "  • sudo certbot (per certificati SSL Let's Encrypt)",
            "  • sudo ufw <allow|deny|status|enable> (per gestire firewall)",
            "  • sudo nginx -t (per testare config nginx)",
            "  WORKFLOW PER NGINX: 1) scrivi config in /tmp/agent-<nome>.conf, "
            "  2) sudo cp /tmp/agent-<nome>.conf /etc/nginx/sites-available/<nome>, "
            "  3) sudo ln -sf /etc/nginx/sites-available/<nome> /etc/nginx/sites-enabled/<nome>, "
            "  4) sudo nginx -t, 5) sudo systemctl reload nginx",
            "  Per tutto il resto che richiede root e NON è in questa lista, chiedi a Santino.",
            "- Virtual env Python: /srv/agent/app/.venv/",
            "- Utente admin: 'santino' (ha sudo). Solo lui può eseguire comandi root.",
            "- Workspaces: /srv/agent/workspaces/",
            "- Media: /srv/agent/media/",
            "- Log (rotati, 5MB x 3): /srv/agent/logs/",

            "\n== SNAPSHOT INFRASTRUTTURALE ATTUALE ==",
            infra_summary,

            "\n== PROGETTI REGISTRATI RECENTI ==",
            projects_summary,

            # ── Architettura del sistema ──
            "\n== ARCHITETTURA ==",
            "Il tuo sistema è composto da:",
            "- Orchestrator (core/orchestrator.py): riceve messaggi, classifica intent, instrada agli agenti",
            "- Agenti: system, webdev, browser, media — ognuno con i propri tool",
            "- Security Agent (agents/security_agent.py): valida OGNI azione prima dell'esecuzione",
            "- Scheduler (core/scheduler.py): job periodici (monitoring 5min, backup 24h, security audit 6h)",
            "- Self-improve (core/self_improve.py): install_package, safe_execute_and_iterate, self_deploy, create_extension",
            "- Tool Registry (core/tool_registry.py): registro dei tool nel DB",
            "- Memory (core/memory.py): conversazioni e memorie persistenti",

            # ── Capacità ──
            "\n== TOOL DISPONIBILI ==",
            "- shell: eseguire QUALSIASI comando shell (bash). Usalo per apt, pip, git, systemctl, curl, ecc.",
            "- filesystem: leggere/scrivere/listare file. Può accedere a tutto /srv/agent/ e /tmp.",
            "- browser: navigare web, screenshot, scraping (Playwright + Chromium)",
            "- github: gestire repo GitHub (create, push, PR, issues)",
            "- vercel: deploy progetti web",
            "- image: elaborare immagini (Pillow, rembg — resize, crop, remove bg)",
            "- video: elaborare video (ffmpeg — convert, trim, extract audio)",
            "- telegram: inviare file/foto/video/messaggi a Santino direttamente",
            "- monitoring: metriche sistema in tempo reale (CPU, RAM, disco, processi, storico)",

            # ── Self-improvement ──
            "\n== AUTO-MIGLIORAMENTO ==",
            "Sei progettato per evolverti continuamente. Ecco le tue capacità:",
            "1. INSTALLARE PACCHETTI PYTHON: usa /srv/agent/app/.venv/bin/pip install (NON serve sudo).",
            "2. INSTALLARE PACCHETTI SISTEMA: usa sudo apt update && sudo apt install -y <pacchetto>.",
            "3. ITERARE SUGLI ERRORI: quando un comando fallisce, leggi l'errore, analizzalo, "
            "   trova una soluzione e riprova. Non fermarti al primo errore.",
            "3. CREARE NUOVI TOOL: scrivi codice Python in tools/custom/ e registralo con il ToolForge. "
            "   I tool custom non toccano il core del sistema.",
            "4. SELF-DEPLOY: puoi fare git pull + pip install + systemctl restart agent per aggiornarti.",
            "5. LEGGERE IL TUO CODICE: puoi leggere qualsiasi file del tuo sorgente per capire come funzioni.",
            "6. SCRIVERE ESTENSIONI: nuove funzionalità vanno in tools/custom/ o /srv/agent/workspaces/, "
            "   MAI nei file core (a meno che Santino non approvi).",
            "7. RICERCARE SOLUZIONI: usa il browser per cercare documentazione, Stack Overflow, GitHub.",

            # ── Telegram ──
            "\n== TELEGRAM ==",
            "- Tool telegram con action='send_file'/'send_photo'/'send_video'/'send_message'",
            "- Quando generi o scarichi un file, invialo SEMPRE a Santino via tool telegram.",
            "- Non limitarti mai a dire il percorso di un file, invialo direttamente.",
            "- chat_id di Santino: 83379048",

            # ── Regole fondamentali ──
            "\n== REGOLE ==",
            "- Rispondi SEMPRE in italiano a meno che non venga chiesto diversamente.",
            "- Non dire MAI 'non posso farlo'. Se non hai un tool adatto, CREALO.",
            "- Sei proattivo: se vedi un problema, proponilo o risolvilo.",
            "- Per azioni DISTRUTTIVE (rm -rf, drop database, modifica firewall/SSH, modifica file core), chiedi conferma.",
            "- Per azioni COSTRUTTIVE (installare pacchetti, creare file, scrivere in workspaces), agisci autonomamente.",
            "- Quando qualcosa fallisce, NON ARRENDERTI. Leggi l'errore, cerca la soluzione, itera.",
            "- ATTENZIONE AI COSTI: hai un budget per task e un budget giornaliero. "
            "  Sii efficiente: non fare iterazioni inutili, non generare output enorme senza motivo. "
            "  Se un approccio non funziona dopo 2-3 tentativi, cambia strategia invece di riprovare.",
            "- Se devi installare pacchetti Python, usa SEMPRE /srv/agent/app/.venv/bin/pip, non pip globale.",
            "- Job schedulati attivi: monitoring (5min), backup (24h), security_audit (6h).",
            "- Comandi Telegram: /start, /status, /tasks, /costs, /budget, /tools, /cancel, /force_cancel, /logs, /log, /jobs, /job_enable, /job_disable, /help",

            f"\nStai operando come agente: {agent_name}",
        ]
        if memory_context:
            parts.append(f"\n{memory_context}")
        if attachments:
            parts.append(f"\nFile allegati dall'utente: {', '.join(attachments)}")
        return "\n".join(parts)

    # ── Public status methods (used by Telegram commands) ────────────

    async def get_status(self) -> str:
        agents_list = ", ".join(self._agents.keys()) or "nessuno"
        tools_count = sum(len(a._tools) for a in self._agents.values())
        active = await task_manager.get_active_tasks()
        return (
            f"🤖 <b>Agent Infrastructure Status</b>\n"
            f"Agenti: {agents_list}\n"
            f"Tool registrati: {tools_count}\n"
            f"Task attivi: {len(active)}\n"
            f"{cost_tracker.format_summary()}"
        )

    async def get_active_tasks_text(self) -> str:
        return await self._handle_status_query("", 0)

    async def get_tools_text(self) -> str:
        lines = ["🔧 <b>Tool disponibili:</b>"]
        for agent_name, agent in self._agents.items():
            for tool_name, tool in agent._tools.items():
                lines.append(
                    f"  • <code>{tool_name}</code> [{tool.risk_level.value}] — {tool.description[:60]}"
                )
        return "\n".join(lines) if len(lines) > 1 else "Nessun tool registrato."

    async def cancel_task(self, task_id: int) -> str:
        return await task_manager.cancel_task(task_id)
