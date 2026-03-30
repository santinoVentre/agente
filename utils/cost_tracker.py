"""Cost tracking for OpenRouter API usage with budget enforcement."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field

from utils.logging import setup_logging

log = setup_logging("cost_tracker")

# Budget defaults (can be overridden via env vars)
DEFAULT_TASK_BUDGET = float(os.getenv("COST_LIMIT_PER_TASK", "0.50"))
DEFAULT_DAILY_BUDGET = float(os.getenv("COST_LIMIT_DAILY", "3.00"))

# Model pricing in USD per 1M tokens (input/output).
# Keep this compact and focused on models commonly used in this project.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "google/gemini-2.0-flash-001": {"input": 0.0, "output": 0.0},
    "google/gemini-2.0-flash-lite-001": {"input": 0.0, "output": 0.0},
    "anthropic/claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "anthropic/claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.0},
    "qwen/qwen3-235b-a22b": {"input": 0.90, "output": 0.90},
    "moonshotai/kimi-k2": {"input": 0.60, "output": 2.40},
    "minimax/minimax-m2.5": {"input": 0.20, "output": 0.80},
}


def _find_pricing(model: str) -> dict[str, float] | None:
    pricing = MODEL_PRICING.get(model)
    if pricing:
        return pricing

    # Fallback fuzzy match for provider/model variants.
    normalized = model.lower()
    for known_model, p in MODEL_PRICING.items():
        km = known_model.lower()
        if km in normalized or normalized in km:
            return p
    return None


def calculate_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    """Estimate USD cost from token usage and pricing table."""
    pricing = _find_pricing(model)
    if not pricing:
        log.warning(f"No pricing found for model '{model}', assuming $0")
        return 0.0

    input_cost = (tokens_input / 1_000_000) * pricing["input"]
    output_cost = (tokens_output / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class CostLimitExceeded(Exception):
    """Raised when a cost limit is exceeded."""
    pass


@dataclass
class UsageRecord:
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    task_id: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Track API costs per model/task and enforce budgets."""

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._lock = asyncio.Lock()
        self.task_budget: float = DEFAULT_TASK_BUDGET
        self.daily_budget: float = DEFAULT_DAILY_BUDGET

    async def record(
        self, model: str, tokens_input: int, tokens_output: int, cost_usd: float = 0.0,
        task_id: int | None = None,
    ):
        if cost_usd <= 0.0:
            cost_usd = calculate_cost(model, tokens_input, tokens_output)

        async with self._lock:
            rec = UsageRecord(
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                task_id=task_id,
            )
            self._records.append(rec)
            log.info(f"API cost: ${cost_usd:.6f} | {model} | in={tokens_input} out={tokens_output} | task={task_id}")

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def get_task_cost(self, task_id: int) -> float:
        """Get total cost for a specific task."""
        return sum(r.cost_usd for r in self._records if r.task_id == task_id)

    def get_daily_cost(self) -> float:
        """Get total cost for today (UTC)."""
        today = datetime.now(timezone.utc).date()
        return sum(r.cost_usd for r in self._records if r.timestamp.date() == today)

    def check_task_budget(self, task_id: int) -> bool:
        """Returns True if task is within budget."""
        return self.get_task_cost(task_id) < self.task_budget

    def check_daily_budget(self) -> bool:
        """Returns True if daily spend is within budget."""
        return self.get_daily_cost() < self.daily_budget

    def summary_by_model(self) -> dict[str, dict]:
        models: dict[str, dict] = {}
        for r in self._records:
            if r.model not in models:
                models[r.model] = {"cost": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}
            m = models[r.model]
            m["cost"] += r.cost_usd
            m["calls"] += 1
            m["tokens_in"] += r.tokens_input
            m["tokens_out"] += r.tokens_output
        return models

    def format_summary(self) -> str:
        daily = self.get_daily_cost()
        lines = [
            f"💰 Costo totale: ${self.total_cost:.4f}",
            f"📅 Oggi: ${daily:.4f} / ${self.daily_budget:.2f} budget",
        ]
        for model, data in sorted(self.summary_by_model().items(), key=lambda x: -x[1]["cost"]):
            lines.append(
                f"  {model}: ${data['cost']:.4f} ({data['calls']} calls, "
                f"{data['tokens_in']+data['tokens_out']} tokens)"
            )
        return "\n".join(lines)


# Singleton
cost_tracker = CostTracker()
