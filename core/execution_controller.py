"""Execution controller — enforces step limits, token budgets, loop detection, and model escalation.

Hierarchy:
  Level 0 (CHEAP)    — execution, parsing, tool calls, repetitive tasks
  Level 1 (MID)      — reasoning, planning, code generation
  Level 2 (EXPENSIVE)— last resort: task failed multiple times or extreme complexity
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import config
from utils.logging import setup_logging

log = setup_logging("execution_controller")

MAX_STEPS: int = config.max_steps_per_task
MAX_TOKENS_PER_TASK: int = config.max_tokens_per_task
MAX_SAME_ACTION: int = 2
SUMMARY_INTERVAL: int = 3   # compress context every N steps
MAX_RETRIES: int = 3        # per-action retry cap before model escalation


@dataclass
class ExecutionState:
    task_id: int
    steps: int = 0
    total_tokens: int = 0
    base_model: str = ""
    base_model_level: int = 0
    model_level: int = 0                                # 0=cheap, 1=mid, 2=expensive
    retry_count: int = 0
    _action_counts: dict[str, int] = field(default_factory=dict)

    # ── step tracking ────────────────────────────────────────────────

    def tick(self) -> None:
        self.steps += 1

    def record_tokens(self, tokens: int) -> None:
        self.total_tokens += tokens

    # ── loop detection ───────────────────────────────────────────────

    def record_action(self, signature: str) -> bool:
        """Count action; return True if it has been repeated ≥ MAX_SAME_ACTION times."""
        count = self._action_counts.get(signature, 0) + 1
        self._action_counts[signature] = count
        return count > MAX_SAME_ACTION

    # ── guards ───────────────────────────────────────────────────────

    def is_step_limit_reached(self, max_steps: int | None = None, start_steps: int = 0) -> bool:
        limit = max_steps if max_steps is not None else MAX_STEPS
        return (self.steps - start_steps) >= limit

    def is_token_budget_exceeded(self, max_tokens: int | None = None, start_tokens: int = 0) -> bool:
        limit = max_tokens if max_tokens is not None else MAX_TOKENS_PER_TASK
        return (self.total_tokens - start_tokens) >= limit

    def should_compress(self) -> bool:
        """True every SUMMARY_INTERVAL steps (but not at step 0)."""
        return self.steps > 0 and self.steps % SUMMARY_INTERVAL == 0

    # ── retry / escalation ───────────────────────────────────────────

    def _level_for_model(self, model: str) -> int:
        if model == config.model_expensive:
            return 2
        if model == config.model_mid or model == config.model_mid_fallback:
            return 1
        return 0

    def configure_base_model(self, model: str) -> None:
        self.base_model = model
        self.base_model_level = self._level_for_model(model)
        self.model_level = self.base_model_level

    def reset_model(self) -> str:
        self.model_level = self.base_model_level
        return self.base_model or config.model_cheap

    def escalate_model(self) -> str | None:
        """Escalate to the next model tier; return the new model string or None if already maxed."""
        if self.model_level == 0:
            self.model_level = 1
            model = config.model_mid
            log.info(f"[task {self.task_id}] Model escalated → MID ({model})")
            return model
        if self.model_level == 1:
            self.model_level = 2
            model = config.model_expensive
            log.info(f"[task {self.task_id}] Model escalated → EXPENSIVE ({model})")
            return model
        log.warning(f"[task {self.task_id}] Already at max model level — cannot escalate further")
        return None

    def current_model_label(self) -> str:
        return ["CHEAP", "MID", "EXPENSIVE"][self.model_level]


class ExecutionController:
    """Manages per-task ExecutionState instances."""

    def __init__(self) -> None:
        self._states: dict[int, ExecutionState] = {}

    def get(self, task_id: int) -> ExecutionState:
        if task_id not in self._states:
            self._states[task_id] = ExecutionState(task_id=task_id)
        return self._states[task_id]

    def clear(self, task_id: int) -> None:
        self._states.pop(task_id, None)

    def step(self, task_id: int) -> ExecutionState:
        """Tick the step counter and return the updated state."""
        state = self.get(task_id)
        state.tick()
        return state


# Singleton used by BaseAgent and callsites
execution_controller = ExecutionController()
