"""Monitoring — collects system metrics and sends alerts on threshold breaches."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import psutil
import redis.asyncio as aioredis

from config import config
from tg.notifications import notify
from utils.logging import setup_logging

log = setup_logging("monitoring")

# Alert cooldown: don't spam Telegram for the same metric
_last_alert: dict[str, float] = {}
ALERT_COOLDOWN = 900  # 15 minutes

# Thresholds
THRESHOLDS = {
    "cpu_percent": 80.0,
    "ram_percent": 85.0,
    "disk_percent": 90.0,
}


async def collect_metrics() -> dict:
    """Collect system metrics (CPU, RAM, disk, uptime). No AI, pure local."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime = datetime.now(timezone.utc) - boot

    return {
        "cpu_percent": cpu,
        "ram_total_mb": round(mem.total / 1024 / 1024),
        "ram_used_mb": round(mem.used / 1024 / 1024),
        "ram_percent": mem.percent,
        "disk_total_gb": round(disk.total / 1024 / 1024 / 1024, 1),
        "disk_used_gb": round(disk.used / 1024 / 1024 / 1024, 1),
        "disk_percent": disk.percent,
        "uptime_hours": round(uptime.total_seconds() / 3600, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _store_metrics(metrics: dict):
    """Store metrics in Redis for history (last 100 points)."""
    try:
        import json
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        await r.lpush("metrics:history", json.dumps(metrics))
        await r.ltrim("metrics:history", 0, 99)
        await r.close()
    except Exception as e:
        log.warning(f"Failed to store metrics in Redis: {e}")


async def _check_alerts(metrics: dict):
    """Check thresholds and send Telegram alerts (with cooldown)."""
    now = time.time()
    alerts = []

    for key, threshold in THRESHOLDS.items():
        value = metrics.get(key, 0)
        if value >= threshold:
            last = _last_alert.get(key, 0)
            if now - last > ALERT_COOLDOWN:
                alerts.append((key, value, threshold))
                _last_alert[key] = now

    if alerts:
        msg_parts = ["⚠️ <b>Alert Sistema</b>"]
        for key, value, threshold in alerts:
            label = {"cpu_percent": "CPU", "ram_percent": "RAM", "disk_percent": "Disco"}.get(key, key)
            msg_parts.append(f"• {label}: <b>{value:.1f}%</b> (soglia: {threshold}%)")
        await notify("\n".join(msg_parts))


async def monitoring_job() -> str:
    """Scheduled job: collect metrics, store, check alerts. No AI cost."""
    metrics = await collect_metrics()
    await _store_metrics(metrics)
    await _check_alerts(metrics)
    return f"CPU={metrics['cpu_percent']}% RAM={metrics['ram_percent']}% Disk={metrics['disk_percent']}%"
