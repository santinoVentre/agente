"""Base agent class — shared logic for all sub-agents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from core.openrouter_client import openrouter
from core.model_router import TaskType, get_model_for_task
from core.execution_controller import execution_controller, MAX_TOKENS_PER_TASK
from core.context_compressor import compress_messages
from core.task_manager import task_manager
from config import config
from db.models import ActionVerdict, RiskLevel, TaskStatus
from agents.security_agent import security_agent
from tg.notifications import notify, notify_approval_needed
from tg.handlers import request_approval
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("base_agent")

MAX_CONTINUATIONS = 2  # Max times user can approve continuing past iteration limit
HEARTBEAT_SECONDS = 45
MAX_SAME_TOOL_CALLS = 3
MAX_CONSECUTIVE_TOOL_ERRORS = 4


class BaseAgent:
    """Base class for specialised sub-agents."""

    name: str = "base"
    description: str = ""
    default_task_type: TaskType = TaskType.TOOL_EXECUTION
    max_iterations: int | None = 15
    max_steps_per_task: int | None = None
    max_continuations: int = MAX_CONTINUATIONS
    max_same_tool_calls: int = MAX_SAME_TOOL_CALLS
    loop_approval_threshold: int = MAX_SAME_TOOL_CALLS
    ask_approval_on_loop: bool = False
    ask_approval_on_iteration_limit: bool = True

    def __init__(self, tools: dict[str, Any] | None = None):
        self._tools: dict[str, Any] = tools or {}

    def register_tool(self, tool):
        self._tools[tool.name] = tool

    def _iteration_limit_enabled(self) -> bool:
        return self.max_iterations is not None and self.max_iterations > 0

    def get_tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    async def _request_user_approval(
        self,
        task_id: int,
        message_html: str,
        timeout: float = 600.0,
        progress: int | None = None,
    ) -> bool:
        """Set explicit waiting state and ask the owner to approve/reject."""
        await task_manager.set_waiting_approval(task_id, reason=message_html, progress=progress)
        await notify_approval_needed(message_html, task_id=task_id)
        approved = await request_approval(task_id, timeout=timeout)
        if approved:
            await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=progress)
        return approved

    async def execute_tool(
        self, tool_name: str, parameters: dict, task_id: int | None = None
    ) -> dict[str, Any]:
        """Execute a tool with security checks and logging."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found."}

        # Security check
        risk = tool.risk_level
        if tool_name == "shell":
            risk = tool.get_command_risk(parameters.get("command", ""))

        verdict, assessed_risk, block_reason = await security_agent.evaluate_action(
            tool_name=tool_name,
            action=f"{tool_name}({json.dumps(parameters, default=str)[:200]})",
            parameters=parameters,
            risk_override=risk,
        )

        if verdict == ActionVerdict.BLOCKED:
            await task_manager.log_action(
                task_id, self.name, tool_name,
                f"BLOCKED: {block_reason}",
                parameters, assessed_risk, verdict,
            )
            return {"success": False, "error": f"Security blocked: {block_reason}"}

        if verdict == ActionVerdict.PENDING:
            # Request approval via Telegram
            desc = f"<b>{tool_name}</b>\n<code>{json.dumps(parameters, default=str)[:300]}</code>"
            if task_id is not None:
                approved = await self._request_user_approval(task_id, desc, timeout=600.0)
            else:
                await notify_approval_needed(desc, task_id=task_id or 0)
                approved = await request_approval(task_id or 0)
            if not approved:
                await task_manager.log_action(
                    task_id, self.name, tool_name,
                    "User rejected action",
                    parameters, assessed_risk, ActionVerdict.BLOCKED,
                )
                return {"success": False, "error": "Action rejected by user."}
            verdict = ActionVerdict.APPROVED

        # Execute
        try:
            result = await tool.execute(**parameters)
            await task_manager.log_action(
                task_id, self.name, tool_name,
                f"Executed {tool_name}",
                parameters, assessed_risk, verdict,
                result=result,
            )
            return result
        except Exception as e:
            error_msg = str(e)
            await task_manager.log_action(
                task_id, self.name, tool_name,
                f"Error: {error_msg}",
                parameters, assessed_risk, verdict,
                error=error_msg,
            )
            return {"success": False, "error": error_msg}

    async def run(
        self,
        user_message: str,
        task_id: int,
        history: list[dict] | None = None,
        system_prompt: str = "",
        model_override: str | None = None,
    ) -> str:
        """
        Main agent loop: send prompt to LLM with tool schemas,
        handle tool calls iteratively until the LLM returns a final text response.
        Enforces per-task and daily cost budgets.
        """
        model = model_override or get_model_for_task(self.default_task_type)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_schemas = self.get_tool_schemas()
        max_iterations = self.max_iterations or 0
        continuations = 0
        consecutive_tool_errors = 0
        same_tool_call_streak = 0
        last_tool_signature = ""
        loop = asyncio.get_running_loop()
        last_heartbeat = loop.time()
        length_continuations = 0
        max_length_continuations = 2

        _task_budget_approved = False   # True once user approves exceeding task budget
        _daily_budget_approved = False  # True once user approves exceeding daily budget

        # ── Execution controller state for this run ──────────────────
        exec_state = execution_controller.get(task_id)
        run_start_steps = exec_state.steps
        _effective_max_steps = self.max_steps_per_task or config.max_steps_per_task

        iteration = 0
        while True:
            if self._iteration_limit_enabled() and iteration >= max_iterations:
                # Reached iteration limit — optionally ask for continuation
                if not self.ask_approval_on_iteration_limit:
                    iteration = 0
                    continue

                continuations += 1
                if continuations > self.max_continuations:
                    task_cost = cost_tracker.get_task_cost(task_id)
                    await notify(
                        f"🛑 <b>Task #{task_id} fermato definitivamente</b>\n"
                        f"Raggiunte {max_iterations * (continuations)} iterazioni totali.\n"
                        f"Costo task: ${task_cost:.2f}"
                    )
                    return (
                        "Ho raggiunto il limite massimo di continuazioni. "
                        "Non posso proseguire oltre per evitare costi eccessivi."
                    )

                task_cost = cost_tracker.get_task_cost(task_id)
                approved = await self._request_user_approval(
                    task_id,
                    f"⏸️ Limite iterazioni raggiunto ({max_iterations})\n"
                    f"Agente: <b>{self.name}</b> — Task #{task_id}\n"
                    f"💰 Costo finora: <b>${task_cost:.2f}</b> / ${cost_tracker.task_budget:.2f}\n"
                    f"Continuazione {continuations}/{self.max_continuations}\n"
                    f"Posso continuare?",
                    timeout=600.0,
                    progress=95,
                )
                if not approved:
                    return "Task fermato su tua richiesta (limite iterazioni raggiunto)."
                iteration = 0
                continue

            iteration += 1
            exec_state.tick()
            now = loop.time()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                progress = min(90, 10 + iteration * 5)
                await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=progress)
                last_heartbeat = now

            # ── Execution controller guards ─────────────────────────────

            # Hard step limit (MAX_STEPS from config, default 8)
            if not self.ask_approval_on_iteration_limit and exec_state.is_step_limit_reached(
                max_steps=_effective_max_steps,
                start_steps=run_start_steps,
            ):
                log.warning(f"[{self.name}] Task #{task_id} reached MAX_STEPS ({_effective_max_steps})")
                await notify(
                    f"⚠️ Task #{task_id} (<b>{self.name}</b>): step limit reached ({exec_state.steps}). Stopping."
                )
                return (
                    f"Stopped: reached the maximum step limit ({_effective_max_steps}). "
                    "Task may be incomplete."
                )

            # Token budget guard
            if exec_state.is_token_budget_exceeded():
                log.warning(
                    f"[{self.name}] Task #{task_id} token budget exceeded "
                    f"({exec_state.total_tokens} / {MAX_TOKENS_PER_TASK})"
                )
                return (
                    f"Stopped: token budget exceeded ({exec_state.total_tokens} tokens used, "
                    f"limit {MAX_TOKENS_PER_TASK}). Task may be incomplete."
                )

            # Context compression every SUMMARY_INTERVAL steps — uses cheap model
            if exec_state.should_compress():
                messages = await compress_messages(messages, task_id=task_id)

            # Check per-task cost limit (ask user, don't auto-block)
            task_cost = cost_tracker.get_task_cost(task_id)
            if task_cost >= cost_tracker.task_budget and not _task_budget_approved:
                log.warning(f"[{self.name}] Task #{task_id} exceeded budget: ${task_cost:.4f}")
                approved = await self._request_user_approval(
                    task_id,
                    f"⚠️ <b>Budget task superato</b>\n"
                    f"Task #{task_id} — Agente: <b>{self.name}</b>\n"
                    f"💰 Speso: <b>${task_cost:.2f}</b> / ${cost_tracker.task_budget:.2f}\n"
                    f"Iterazione: {iteration}\n\n"
                    f"Vuoi continuare o fermare il task?",
                    timeout=600.0,
                    progress=min(90, 10 + iteration * 5),
                )
                if not approved:
                    return (
                        f"Task fermato su tua richiesta. "
                        f"Costo: ${task_cost:.2f}. Ecco quello che ho fatto finora."
                    )
                _task_budget_approved = True
                log.info(f"[{self.name}] User approved exceeding task budget for #{task_id}")

            # Check daily budget (ask user, don't auto-block)
            daily_cost = cost_tracker.get_daily_cost()
            if daily_cost >= cost_tracker.daily_budget and not _daily_budget_approved:
                log.warning(f"[{self.name}] Daily budget exceeded: ${daily_cost:.4f}")
                approved = await self._request_user_approval(
                    task_id,
                    f"⚠️ <b>Budget giornaliero superato</b>\n"
                    f"📅 Speso oggi: <b>${daily_cost:.2f}</b> / ${cost_tracker.daily_budget:.2f}\n"
                    f"Task attuale: #{task_id}\n\n"
                    f"Vuoi continuare o fermare?",
                    timeout=600.0,
                    progress=min(90, 10 + iteration * 5),
                )
                if not approved:
                    return "Task fermato su tua richiesta (budget giornaliero superato)."
                _daily_budget_approved = True
                log.info(f"[{self.name}] User approved exceeding daily budget")

            response = await openrouter.chat(
                model=model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=0.3,
                max_tokens=4096,
                task_id=task_id,
            )

            # Track tokens consumed by this call against the task token budget
            _usage = response.get("usage", {})
            exec_state.record_tokens(
                _usage.get("total_tokens", 0) or
                _usage.get("prompt_tokens", 0) + _usage.get("completion_tokens", 0)
            )

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # If the model wants to call tools
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                messages.append(message)  # assistant message with tool_calls
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    tool_signature = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)[:500]}"
                    if tool_signature == last_tool_signature:
                        same_tool_call_streak += 1
                    else:
                        same_tool_call_streak = 1
                    last_tool_signature = tool_signature

                    if same_tool_call_streak > self.max_same_tool_calls:
                        if self.ask_approval_on_loop and same_tool_call_streak >= self.loop_approval_threshold:
                            approved = await self._request_user_approval(
                                task_id,
                                f"⚠️ <b>Possibile loop rilevato</b>\n"
                                f"Tool: <b>{tool_name}</b>\n"
                                f"Ripetizioni simili: <b>{same_tool_call_streak}</b>\n"
                                f"Vuoi che continui comunque?",
                                timeout=600.0,
                                progress=min(95, 10 + iteration * 5),
                            )
                            if not approved:
                                return (
                                    "Mi sono fermato per evitare loop silenziosi "
                                    f"sul tool {tool_name}."
                                )
                            same_tool_call_streak = 0
                            last_tool_signature = ""
                        else:
                            await notify(
                                f"⚠️ Task #{task_id}: rilevata possibile iterazione ripetitiva su tool "
                                f"<b>{tool_name}</b>. Mi fermo e chiedo nuova direzione."
                            )
                            return (
                                "Mi sono fermato per evitare loop silenziosi: "
                                f"lo stesso tool ({tool_name}) e' stato chiamato troppe volte con input simile."
                            )

                    log.info(f"[{self.name}] Tool call: {tool_name}({args})")
                    result = await self.execute_tool(tool_name, args, task_id)

                    if not result.get("success", True):
                        consecutive_tool_errors += 1
                        # Model escalation: if repeated failures, escalate tier
                        if consecutive_tool_errors >= 2:
                            escalated = exec_state.escalate_model()
                            if escalated and model != escalated:
                                log.info(
                                    f"[{self.name}] Task #{task_id}: escalating model "
                                    f"{model} → {escalated} after {consecutive_tool_errors} errors"
                                )
                                model = escalated
                    else:
                        consecutive_tool_errors = 0

                    if consecutive_tool_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                        await notify(
                            f"⚠️ Task #{task_id}: troppi errori consecutivi ({consecutive_tool_errors}) "
                            "durante le tool call. Richiedo tua indicazione."
                        )
                        return (
                            "Mi sono fermato dopo diversi errori consecutivi nelle azioni automatiche. "
                            "Indicami se vuoi che cambi strategia o proceda in modo diverso."
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result, default=str)[:8000],
                    })

                    # Update progress
                    progress = min(90, 10 + iteration * 10)
                    await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=progress)
                continue

            # Final text response
            content = message.get("content", "")

            # If output was truncated by token limit, ask model to continue seamlessly.
            if finish_reason == "length" and content and length_continuations < max_length_continuations:
                length_continuations += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Continua esattamente dal punto in cui eri rimasto, "
                        "senza ripetere le parti gia' inviate."
                    ),
                })
                continue

            return content
