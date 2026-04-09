"""Orchestrator — the central brain that routes requests to sub-agents."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from config import config
from core.memory import memory
from core.inventory import get_latest_snapshot_summary
from core.model_router import CLASSIFICATION_PROMPT, TaskType, get_model_for_task
from core.openrouter_client import openrouter
from core.project_registry import project_registry
from core.project_selector import end_selector, get_selector, start_selector
from core.reflection import reflection_engine
from core.task_manager import task_manager
from core.webdev_planner import PLANNING_QUESTIONS, get_session, start_session
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
            TaskType.CODE_FIX: "system",
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

        # ── WebDev: project selector or new Q&A session ──────────────────────
        if task_type == TaskType.WEB_DEV and get_session(user_id) is None:
            # Check for existing active projects
            try:
                all_projects = await project_registry.list_projects(limit=30)
                active_projects = [p for p in all_projects if p.status in ("active", "building")]
            except Exception:
                active_projects = []

            if active_projects:
                # Show project selector — the handler will pick this up next message
                selector = start_selector(user_id, active_projects, message)
                return selector.format_menu()

            # No existing projects → start Q&A immediately
            return self._start_new_webdev_session(user_id, message)
        # ─────────────────────────────────────────────────────────────────────

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

        # Build context — keep history short to save tokens
        history = await memory.get_conversation_history(user_id, limit=4)
        memories = await memory.recall_recent(user_id, limit=5, value_max_chars=200)
        memory_context = memory.format_memories_for_prompt(memories)

        # Inject improvement guidelines for this task type
        improvement_ctx = await reflection_engine.get_improvement_context(user_id, task_type.value)
        system_prompt = await self._build_system_prompt(agent_name, memory_context, attachments, task_type)
        if improvement_ctx:
            system_prompt = system_prompt + improvement_ctx[:800]

        # For quick tasks, run synchronously; for complex ones, run in background
        if task_type in (TaskType.SIMPLE_CHAT, TaskType.ROUTING, TaskType.SUMMARIZATION):
            return await self._run_sync(agent, message, task, history, system_prompt, model)
        else:
            return await self._run_async(agent, message, task, history, system_prompt, model)

    async def _run_sync(self, agent, message, task, history, system_prompt, model) -> str:
        """Run agent synchronously — return response directly."""
        t0 = time.monotonic()
        try:
            response = await agent.run(
                user_message=message,
                task_id=task.id,
                history=history,
                system_prompt=system_prompt,
                model_override=model,
            )
            duration = time.monotonic() - t0
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
            # Self-improvement: analyze completed task
            asyncio.create_task(reflection_engine.analyze_task(
                user_id=task.user_id,
                task_id=task.id,
                task_type=task.task_type,
                user_message=message,
                outcome=response,
                success=True,
                cost=cost_tracker.get_task_cost(task.id),
                duration_seconds=duration,
            ))
            return response
        except Exception as e:
            duration = time.monotonic() - t0
            log.error(f"Task #{task.id} failed: {e}", exc_info=True)
            await task_manager.fail_task(task.id, str(e))
            asyncio.create_task(reflection_engine.analyze_task(
                user_id=task.user_id,
                task_id=task.id,
                task_type=task.task_type,
                user_message=message,
                outcome=str(e),
                success=False,
                cost=cost_tracker.get_task_cost(task.id),
                duration_seconds=duration,
            ))
            return f"❌ Errore: {str(e)[:500]}"

    async def _run_async(self, agent, message, task, history, system_prompt, model) -> str:
        """Launch agent in background — return acknowledgement immediately."""

        async def _background():
            t0 = time.monotonic()
            try:
                response = await agent.run(
                    user_message=message,
                    task_id=task.id,
                    history=history,
                    system_prompt=system_prompt,
                    model_override=model,
                )
                duration = time.monotonic() - t0
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
                # Self-improvement: analyze completed task
                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=task.user_id,
                    task_id=task.id,
                    task_type=task.task_type,
                    user_message=message,
                    outcome=response,
                    success=True,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_done(message[:60], response)
            except Exception as e:
                duration = time.monotonic() - t0
                log.error(f"Task #{task.id} failed: {e}", exc_info=True)
                await task_manager.fail_task(task.id, str(e))
                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=task.user_id,
                    task_id=task.id,
                    task_type=task.task_type,
                    user_message=message,
                    outcome=str(e),
                    success=False,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_error(message[:60], str(e))
            finally:
                task_manager.unregister_running_task(task.id)

        atask = asyncio.create_task(_background())
        task_manager.register_running_task(task.id, atask)

        # Keep Telegram quiet for operational startup messages.
        return ""

    # ── WebDev helpers ────────────────────────────────────────────────

    def _start_new_webdev_session(self, user_id: int, message: str) -> str:
        """Start a Q&A planning session for a brand-new project and return the first question."""
        session = start_session(user_id, message)
        q = session.current_question
        total = len(PLANNING_QUESTIONS)
        return (
            f"🌐 <b>Perfetto! Creiamo il tuo sito.</b>\n\n"
            f"Prima ho bisogno di capire meglio il progetto. "
            f"Ti farò <b>{total} domande rapide</b> e poi partirò subito.\n\n"
            f"Puoi anche mandare un file JSON/TXT con le specifiche "
            f"e salteremo le domande.\n\n"
            f"{session.progress_bar()}\n\n"
            f"{q['question']}"
        )

    def _create_pm_agent(self, project):
        """Instantiate a ProjectManagerAgent for an existing project with all tools."""
        from agents.project_manager_agent import ProjectManagerAgent
        return ProjectManagerAgent.from_project(tools=dict(self._tools), project=project)

    async def handle_project_modification(
        self,
        user_id: int,
        project,  # ProjectRegistry
        user_message: str,
        chat_id: int,
    ) -> None:
        """Run the PM agent for a modification request on an existing project.

        Always runs in background. The PM agent is the sole user-facing actor.
        """
        from agents.project_manager_agent import ProjectManagerAgent

        pm_agent = self._create_pm_agent(project)
        if pm_agent is None:
            await notify(
                f"⚠️ Impossibile caricare il PM agent per <b>{project.name}</b>.\n"
                "Procedo con l'agente WebDev base."
            )
            # Fallback: treat as new task with the webdev agent
            agent = self._agents.get("webdev")
            if not agent:
                await notify("❌ Nessun agente webdev disponibile.")
                return
        else:
            agent = pm_agent

        model = get_model_for_task(TaskType.WEB_DEV)
        task = await task_manager.create_task(
            user_id=user_id,
            description=f"[pm:{project.name}] {user_message[:460]}",
            task_type=TaskType.WEB_DEV.value,
            agent="project_manager",
            model=model,
        )

        async def _bg():
            t0 = time.monotonic()
            try:
                response = await agent.run(
                    user_message=user_message,
                    task_id=task.id,
                    model_override=model,
                )
                duration = time.monotonic() - t0
                await task_manager.complete_task(task.id, cost=cost_tracker.total_cost)
                await memory.save_message(user_id, "assistant", response, model=model)
                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=user_id,
                    task_id=task.id,
                    task_type=TaskType.WEB_DEV.value,
                    user_message=user_message,
                    outcome=response,
                    success=True,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_done(user_message[:60], response)
            except Exception as exc:
                duration = time.monotonic() - t0
                log.error(f"PM task #{task.id} failed: {exc}", exc_info=True)
                await task_manager.fail_task(task.id, str(exc))
                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=user_id,
                    task_id=task.id,
                    task_type=TaskType.WEB_DEV.value,
                    user_message=user_message,
                    outcome=str(exc),
                    success=False,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_error(user_message[:60], str(exc))
            finally:
                task_manager.unregister_running_task(task.id)

        atask = asyncio.create_task(_bg())
        task_manager.register_running_task(task.id, atask)

    # ── WebDev task (called after Q&A planning completes) ────────────

    async def handle_webdev_task(
        self,
        user_id: int,
        initial_message: str,
        specs: dict,
        design_system: dict,
        media_files: list[str],
        chat_id: int,
        state_files: dict | None = None,
    ) -> None:
        """Run the WebDev pipeline with pre-built specs and design system.

        Called by tg/handlers.py after a planning session completes.
        Always runs in background.
        """
        agent = self._agents.get("webdev")
        if not agent:
            await notify("❌ WebDev agent non disponibile.")
            return

        model = get_model_for_task(TaskType.WEB_DEV)
        task = await task_manager.create_task(
            user_id=user_id,
            description=f"[webdev:planned] {initial_message[:480]}",
            task_type=TaskType.WEB_DEV.value,
            agent="webdev",
            model=model,
        )

        memories = await memory.recall_recent(user_id, limit=5, value_max_chars=200)
        memory_context = memory.format_memories_for_prompt(memories)
        improvement_ctx = await reflection_engine.get_improvement_context(user_id, TaskType.WEB_DEV.value)
        system_prompt = await self._build_system_prompt("webdev", memory_context, media_files or None, TaskType.WEB_DEV)
        if improvement_ctx:
            system_prompt = system_prompt + improvement_ctx[:800]

        async def _bg():
            t0 = time.monotonic()
            try:
                response = await agent.run(
                    user_message=initial_message,
                    task_id=task.id,
                    specs=specs,
                    design_system=design_system,
                    state_files=state_files or {},
                    system_prompt=system_prompt,
                    model_override=model,
                )
                duration = time.monotonic() - t0
                await task_manager.complete_task(task.id, cost=cost_tracker.total_cost)
                await memory.save_message(user_id, "assistant", response, model=model)

                # Register project in registry (pm_context stored in metadata)
                project_name = specs.get("project_name", "")
                if project_name:
                    pm_ctx = (state_files or {}).get("pm_context", "")
                    workdir = f"/srv/agent/workspaces/{project_name}"
                    try:
                        await project_registry.upsert_project(
                            name=project_name,
                            description=specs.get("description", "")[:500],
                            workspace_path=workdir,
                            deploy_provider="vercel",
                            status="active",
                            metadata_json={"pm_context": pm_ctx[:8000]} if pm_ctx else None,
                        )
                        log.info(f"[orchestrator] Project '{project_name}' registered in registry")
                    except Exception as reg_exc:
                        log.warning(f"[orchestrator] Failed to register project: {reg_exc}")

                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=user_id,
                    task_id=task.id,
                    task_type=TaskType.WEB_DEV.value,
                    user_message=initial_message,
                    outcome=response,
                    success=True,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_done(initial_message[:60], response)
            except Exception as exc:
                duration = time.monotonic() - t0
                log.error(f"WebDev task #{task.id} failed: {exc}", exc_info=True)
                await task_manager.fail_task(task.id, str(exc))
                asyncio.create_task(reflection_engine.analyze_task(
                    user_id=user_id,
                    task_id=task.id,
                    task_type=TaskType.WEB_DEV.value,
                    user_message=initial_message,
                    outcome=str(exc),
                    success=False,
                    cost=cost_tracker.get_task_cost(task.id),
                    duration_seconds=duration,
                ))
                await notify_error(initial_message[:60], str(exc))
            finally:
                task_manager.unregister_running_task(task.id)

        atask = asyncio.create_task(_bg())
        task_manager.register_running_task(task.id, atask)

    # ── Status queries ───────────────────────────────────────────────

    async def _is_status_query(self, message: str) -> bool:
        # Exact multi-word phrases that unambiguously request a status update
        exact_phrases = [
            "a che punto", "cosa stai facendo", "cosa stai facend",
            "task attivi", "task in corso", "quanti task",
            "sei occupato", "sei libero", "stai lavorando",
            "qual è lo stato", "qual e' lo stato", "che stato ha",
            "dimmi lo stato", "mostrami i task", "lista task",
        ]
        # Single words only when they appear alone or as the whole message
        standalone_words = ["status", "/tasks"]
        msg_lower = message.lower().strip()
        if any(phrase in msg_lower for phrase in exact_phrases):
            return True
        if msg_lower in standalone_words:
            return True
        return False

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
        self,
        agent_name: str,
        memory_context: str,
        attachments: list[str] | None,
        task_type: TaskType = TaskType.SIMPLE_CHAT,
    ) -> str:
        # ── Tier classification ──────────────────────────────
        _needs_infra = task_type in (
            TaskType.SYSTEM, TaskType.CODE_GENERATION, TaskType.CODE_REVIEW,
            TaskType.CODE_FIX, TaskType.COMPLEX_REASONING,
        )
        _needs_projects = task_type in (TaskType.WEB_DEV, TaskType.CODE_GENERATION)
        _needs_tools_detail = task_type not in (TaskType.SIMPLE_CHAT, TaskType.ROUTING, TaskType.SUMMARIZATION)
        _needs_self_improve = task_type in (
            TaskType.CODE_GENERATION, TaskType.CODE_REVIEW, TaskType.CODE_FIX, TaskType.SYSTEM,
        )

        # ── Core identity (always included) ────────────────────
        parts = [
            "Sei l'agente personale di Santino su VPS Linux (Ubuntu 24.04, /srv/agent/app/). "
            "Comunichi SOLO via Telegram. Rispondi in italiano. "
            "Non dire MAI 'non posso farlo': se manca un tool, crealo. "
            "Per azioni distruttive chiedi conferma; per azioni costruttive agisci autonomamente.",
            f"Agente attivo: {agent_name}",
        ]

        # ── Infrastructure (only for system/code tasks) ──────
        if _needs_infra:
            infra_static = (
                "VPS: 4vCPU/4GB/120GB, IP 87.106.245.253. "
                "Codice: /srv/agent/app/ | Workspaces: /srv/agent/workspaces/ | venv: .venv/bin/pip. "
                "DB: PostgreSQL:5432, Redis:6379. Servizio: agent.service (systemd). "
                "Sudo limitato: apt install, systemctl, certbot, ufw, nginx-t, cp/ln in /etc/nginx/. "
                "File core protetti (richiedono approvazione): orchestrator.py, security_agent.py, config.py, db/models.py."
            )
            parts.append(f"\n== INFRA ==\n{infra_static}")
            try:
                snap = (await get_latest_snapshot_summary())[:1000]
                parts.append(f"Snapshot attuale: {snap}")
            except Exception:
                pass

        # ── Tools list (most tasks, not trivial chat) ──────
        if _needs_tools_detail:
            parts.append(
                "\n== TOOL ==\n"
                "shell(cmd), filesystem(read/write/list), browser(web/scraping), "
                "github(repo/push/PR), vercel(deploy), image(resize/rembg), "
                "video(ffmpeg), telegram(send_file/photo/video), monitoring(CPU/RAM/disk). "
                "Invia SEMPRE i file generati via tool telegram. chat_id Santino: 83379048."
            )

        # ── Self-improvement (code/system tasks only) ─────
        if _needs_self_improve:
            parts.append(
                "\n== AUTO-IMPROVE ==\n"
                "Installa pypackages: .venv/bin/pip install. Sistema: sudo apt install -y. "
                "Itera sugli errori: leggi, analizza, riprova. Crea tool custom in tools/custom/. "
                "Self-deploy: git pull + pip install + systemctl restart agent."
            )

        # ── Projects (webdev / code gen only) ────────────
        if _needs_projects:
            try:
                proj = (await project_registry.get_recent_projects_summary(limit=3))[:600]
                if proj:
                    parts.append(f"\n== PROGETTI RECENTI ==\n{proj}")
            except Exception:
                pass

        # ── Memories (always, but capped) ────────────────
        if memory_context:
            parts.append(f"\n{memory_context[:600]}")

        # ── Attachments ───────────────────────────────
        if attachments:
            parts.append(f"Allegati: {', '.join(attachments)}")

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
