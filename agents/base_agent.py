"""Base agent class — shared logic for all sub-agents."""

from __future__ import annotations

import json
from typing import Any

from core.openrouter_client import openrouter
from core.model_router import TaskType, get_model_for_task
from core.task_manager import task_manager
from db.models import ActionVerdict, RiskLevel, TaskStatus
from agents.security_agent import security_agent
from tg.notifications import notify, notify_approval_needed, notify_progress
from tg.handlers import request_approval
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("base_agent")

MAX_CONTINUATIONS = 2  # Max times user can approve continuing past iteration limit


class BaseAgent:
    """Base class for specialised sub-agents."""

    name: str = "base"
    description: str = ""
    default_task_type: TaskType = TaskType.TOOL_EXECUTION

    def __init__(self, tools: dict[str, Any] | None = None):
        self._tools: dict[str, Any] = tools or {}

    def register_tool(self, tool):
        self._tools[tool.name] = tool

    def get_tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

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
        max_iterations = 15
        continuations = 0

        _task_budget_approved = False   # True once user approves exceeding task budget
        _daily_budget_approved = False  # True once user approves exceeding daily budget

        while True:
            for iteration in range(max_iterations):
                # Check per-task cost limit (ask user, don't auto-block)
                task_cost = cost_tracker.get_task_cost(task_id)
                if task_cost >= cost_tracker.task_budget and not _task_budget_approved:
                    log.warning(f"[{self.name}] Task #{task_id} exceeded budget: ${task_cost:.4f}")
                    await notify_approval_needed(
                        f"⚠️ <b>Budget task superato</b>\n"
                        f"Task #{task_id} — Agente: <b>{self.name}</b>\n"
                        f"💰 Speso: <b>${task_cost:.2f}</b> / ${cost_tracker.task_budget:.2f}\n"
                        f"Iterazione: {iteration}\n\n"
                        f"Vuoi continuare o fermare il task?",
                        task_id=task_id,
                    )
                    approved = await request_approval(task_id, timeout=600.0)
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
                    await notify_approval_needed(
                        f"⚠️ <b>Budget giornaliero superato</b>\n"
                        f"📅 Speso oggi: <b>${daily_cost:.2f}</b> / ${cost_tracker.daily_budget:.2f}\n"
                        f"Task attuale: #{task_id}\n\n"
                        f"Vuoi continuare o fermare?",
                        task_id=task_id,
                    )
                    approved = await request_approval(task_id, timeout=600.0)
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

                        log.info(f"[{self.name}] Tool call: {tool_name}({args})")
                        result = await self.execute_tool(tool_name, args, task_id)

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
                return content

            # Reached iteration limit — check if we can ask for continuation
            continuations += 1
            if continuations > MAX_CONTINUATIONS:
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
            await notify_approval_needed(
                f"⏸️ Limite iterazioni raggiunto ({max_iterations})\n"
                f"Agente: <b>{self.name}</b> — Task #{task_id}\n"
                f"💰 Costo finora: <b>${task_cost:.2f}</b> / ${cost_tracker.task_budget:.2f}\n"
                f"Continuazione {continuations}/{MAX_CONTINUATIONS}\n"
                f"Posso continuare?",
                task_id=task_id,
            )
            approved = await request_approval(task_id, timeout=600.0)
            if not approved:
                return "Ho raggiunto il limite massimo di iterazioni. Ecco quello che ho fatto finora."
            log.info(f"[{self.name}] User approved continuation {continuations}/{MAX_CONTINUATIONS} for task #{task_id}")
