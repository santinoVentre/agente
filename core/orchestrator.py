"""Orchestrator — the central brain that routes requests to sub-agents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from config import config
from core.memory import memory
from core.model_router import CLASSIFICATION_PROMPT, TaskType, get_model_for_task
from core.openrouter_client import openrouter
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

        system_prompt = self._build_system_prompt(agent_name, memory_context, attachments)

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
                await notify_done(message[:60], response[:500])
            except Exception as e:
                log.error(f"Task #{task.id} failed: {e}", exc_info=True)
                await task_manager.fail_task(task.id, str(e))
                await notify_error(message[:60], str(e))
            finally:
                task_manager.unregister_running_task(task.id)

        atask = asyncio.create_task(_background())
        task_manager.register_running_task(task.id, atask)

        return (
            f"🔄 Task <b>#{task.id}</b> avviato\n"
            f"Tipo: <code>{task.task_type}</code>\n"
            f"Agente: <code>{task.agent_assigned}</code>\n"
            f"Ti aggiorno quando finisco."
        )

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

    def _build_system_prompt(
        self, agent_name: str, memory_context: str, attachments: list[str] | None
    ) -> str:
        parts = [
            # ── Identità ──
            "Sei l'agente personale di Santino. Sei un sistema autonomo che vive su un "
            "server VPS Linux (Ubuntu 24.04, 4 vCPU, 4GB RAM, 120GB NVMe, IP 87.106.245.253, IONOS).",
            "Stai comunicando con Santino ESCLUSIVAMENTE tramite le API di Telegram. "
            "La conversazione avviene dentro il bot Telegram: tu ricevi i messaggi e rispondi tramite l'API.",

            # ── Infrastruttura ──
            "\n== INFRASTRUTTURA ==",
            "- Il tuo codice si trova in: /srv/agent/app/",
            "- La tua configurazione (.env) è in: /srv/agent/app/.env — contiene TELEGRAM_BOT_TOKEN, "
            "OPENROUTER_API_KEY, GITHUB_TOKEN, e altri segreti. Puoi leggerla con i tuoi tool.",
            "- Database PostgreSQL 16 su localhost:5432 (Docker container)",
            "- Redis 7 su localhost:6379 (Docker container)",
            "- Sei eseguito come utente 'agent' tramite systemd (servizio: agent.service)",
            "- Utente admin del server: 'santino' (ha sudo)",
            "- Workspaces per progetti: /srv/agent/workspaces/",
            "- Media scaricati/generati: /srv/agent/media/",
            "- Log: /srv/agent/logs/",

            # ── Capacità ──
            "\n== CAPACITÀ ==",
            "- Eseguire QUALSIASI comando shell sul server (hai tool shell_exec)",
            "- Leggere e scrivere file ovunque nel server (hai tool filesystem)",
            "- Navigare il web, fare scraping, screenshot (hai tool browser con Playwright)",
            "- Gestire repository GitHub (hai tool github)",
            "- Deploy su Vercel (hai tool vercel)",
            "- Elaborare immagini e video (hai tool image e video, con ffmpeg e Pillow)",
            "- Installare qualsiasi pacchetto Python o di sistema (pip, apt)",
            "- Creare nuovi tool Python autonomamente (ToolForge)",
            "- Modificare il tuo stesso codice sorgente (code_modifier)",
            "- Monitorare il sistema in tempo reale (hai tool monitoring: CPU, RAM, disco, processi)",

            # ── Telegram API ──
            "\n== TELEGRAM ==",
            "- Per inviare FILE a Santino: usa il tool telegram con action='send_file' e path=percorso",
            "- Per inviare FOTO: usa il tool telegram con action='send_photo' e path=percorso",
            "- Per inviare VIDEO: usa il tool telegram con action='send_video' e path=percorso",
            "- Per inviare un messaggio: usa il tool telegram con action='send_message' e text=messaggio",
            "- Il chat_id di Santino è: 83379048",

            # ── Regole comportamentali ──
            "\n== REGOLE ==",
            "- Rispondi SEMPRE in italiano a meno che non ti venga chiesto diversamente.",
            "- Non dire MAI 'non posso farlo'. Se non hai un tool adatto, CREALO.",
            "- Sei autonomo: puoi installare pacchetti, creare script, modificare il sistema.",
            "- Per azioni distruttive (rm -rf, drop database, modifica firewall/SSH), chiedi conferma.",
            "- Quando scarichi o generi file, inviali DIRETTAMENTE a Santino via tool telegram, non limitarti a dire il percorso.",
            "- Se hai bisogno di informazioni sulla tua configurazione, leggi il tuo .env o il tuo codice.",
            "- Hai un sistema di job schedulati (monitoring ogni 5min, backup giornaliero, security audit ogni 6h).",
            "- Comandi Telegram disponibili: /start, /status, /tasks, /costs, /tools, /cancel, /logs, /log, /jobs, /job_enable, /job_disable, /help",

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
