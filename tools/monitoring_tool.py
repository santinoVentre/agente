"""Monitoring Tool — on-demand system metrics for the agent to query."""

from __future__ import annotations

from typing import Any

from db.models import RiskLevel
from tools.base_tool import BaseTool


class MonitoringTool(BaseTool):
    name = "monitoring"
    description = (
        "Ottieni metriche di sistema in tempo reale: CPU, RAM, disco, uptime, "
        "processi più pesanti, e storico metriche. "
        "Usa action='metrics' per stato attuale, 'top' per processi, 'history' per storico."
    )
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["metrics", "top", "history"],
                    "description": "metrics=stato attuale, top=processi pesanti, history=storico metriche",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action", "metrics")

        if action == "metrics":
            from core.monitoring import collect_metrics
            metrics = await collect_metrics()
            return {"success": True, "metrics": metrics}

        if action == "top":
            import psutil
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = p.info
                    if info["cpu_percent"] and info["cpu_percent"] > 0.5:
                        procs.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
            return {"success": True, "processes": procs[:15]}

        if action == "history":
            import json
            import redis.asyncio as aioredis
            from config import config
            try:
                r = aioredis.from_url(config.redis_url, decode_responses=True)
                raw = await r.lrange("metrics:history", 0, 19)
                await r.close()
                history = [json.loads(item) for item in raw]
                return {"success": True, "history": history}
            except Exception as e:
                return {"success": False, "error": f"Errore Redis: {e}"}

        return {"success": False, "error": f"Azione non supportata: {action}"}
