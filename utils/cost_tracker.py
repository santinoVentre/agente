"""Cost tracking for OpenRouter API usage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, field

from utils.logging import setup_logging

log = setup_logging("cost_tracker")


@dataclass
class UsageRecord:
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Track API costs per model and provide summaries."""

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._lock = asyncio.Lock()

    async def record(self, model: str, tokens_input: int, tokens_output: int, cost_usd: float):
        async with self._lock:
            rec = UsageRecord(
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
            )
            self._records.append(rec)
            log.info(f"API cost: ${cost_usd:.6f} | {model} | in={tokens_input} out={tokens_output}")

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._records)

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
        lines = [f"💰 Total cost: ${self.total_cost:.4f}"]
        for model, data in sorted(self.summary_by_model().items(), key=lambda x: -x[1]["cost"]):
            lines.append(
                f"  {model}: ${data['cost']:.4f} ({data['calls']} calls, "
                f"{data['tokens_in']+data['tokens_out']} tokens)"
            )
        return "\n".join(lines)


# Singleton
cost_tracker = CostTracker()
