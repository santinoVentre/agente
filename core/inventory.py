"""Infrastructure inventory service - snapshots VPS runtime context for prompt grounding."""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil
from sqlalchemy import select

from config import config
from db.connection import async_session
from db.models import InfrastructureSnapshot
from utils.logging import setup_logging

log = setup_logging("inventory")


def _run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
        out = (proc.stdout or proc.stderr or "").strip()
        return out[:400]
    except Exception:
        return ""


def _detect_service_status() -> dict[str, str]:
    services = {
        "docker": "unknown",
        "nginx": "unknown",
        "postgres": "unknown",
        "redis": "unknown",
    }

    if _run_cmd(["docker", "--version"]):
        services["docker"] = "installed"
    if _run_cmd(["nginx", "-v"]):
        services["nginx"] = "installed"

    # Best effort checks for local service ports.
    try:
        conns = psutil.net_connections(kind="tcp")
        listening = {c.laddr.port for c in conns if c.status == psutil.CONN_LISTEN and c.laddr}
        services["postgres"] = "listening" if 5432 in listening else services["postgres"]
        services["redis"] = "listening" if 6379 in listening else services["redis"]
    except Exception:
        pass

    return services


async def collect_infrastructure_snapshot(source: str = "scheduler") -> dict:
    """Collect a full infrastructure snapshot and persist it to DB."""
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime = datetime.now(timezone.utc) - boot

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "resources": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "uptime_hours": round(uptime.total_seconds() / 3600, 2),
        },
        "directories": {
            "workspaces": str(config.workspaces_dir),
            "logs": str(config.logs_dir),
            "media": str(config.media_dir),
            "tools_custom": str(config.tools_custom_dir),
            "tool_backups": str(config.tool_backups_dir),
        },
        "integrations": {
            "github_owner": config.github_owner,
            "has_github_token": bool(config.github_token),
            "has_vercel_token": bool(config.vercel_token),
            "has_openrouter_key": bool(config.openrouter_api_key),
            "telegram_user_id": config.allowed_telegram_user_id,
        },
        "services": _detect_service_status(),
        "docker_ps": _run_cmd(["docker", "ps", "--format", "{{.Names}}"]),
    }

    async with async_session() as session:
        session.add(
            InfrastructureSnapshot(
                source=source,
                hostname=data["hostname"],
                os_name=data["platform"]["system"],
                kernel=data["platform"]["release"],
                python_version=data["platform"]["python"],
                data=data,
            )
        )
        await session.commit()

    return data


async def get_latest_snapshot() -> InfrastructureSnapshot | None:
    async with async_session() as session:
        result = await session.execute(
            select(InfrastructureSnapshot).order_by(InfrastructureSnapshot.created_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()


async def get_latest_snapshot_summary() -> str:
    row = await get_latest_snapshot()
    if not row:
        return "Nessuno snapshot infrastrutturale disponibile."

    data = row.data or {}
    resources = data.get("resources", {})
    services = data.get("services", {})

    return (
        f"host={row.hostname or '-'}; os={row.os_name or '-'} {row.kernel or ''}; "
        f"cpu={resources.get('cpu_percent', '?')}%; ram={resources.get('ram_percent', '?')}%; "
        f"disk={resources.get('disk_percent', '?')}%; "
        f"services=docker:{services.get('docker', 'unknown')},nginx:{services.get('nginx', 'unknown')},"
        f"postgres:{services.get('postgres', 'unknown')},redis:{services.get('redis', 'unknown')}"
    )


async def inventory_job() -> str:
    """Scheduler job for periodic infrastructure snapshots."""
    snapshot = await collect_infrastructure_snapshot(source="scheduler")
    resources = snapshot.get("resources", {})
    return (
        f"snapshot ok | CPU={resources.get('cpu_percent', '?')}% "
        f"RAM={resources.get('ram_percent', '?')}% DISK={resources.get('disk_percent', '?')}%"
    )
