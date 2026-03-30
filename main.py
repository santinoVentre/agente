"""Entry point — initialises all components and starts the Telegram bot."""

from __future__ import annotations

import asyncio
import signal
import sys

from config import config
from db.connection import init_db, close_db
from core.orchestrator import Orchestrator
from core.openrouter_client import openrouter
from core.task_manager import task_manager
from core.tool_registry import tool_registry
from core.scheduler import scheduler
from core.inventory import collect_infrastructure_snapshot, inventory_job
from tg.bot import build_app
from tg.handlers import set_orchestrator
from tg.notifications import set_app
from utils.logging import setup_logging

# Tools
from tools.shell_tool import ShellTool
from tools.filesystem_tool import FileSystemTool
from tools.browser_tool import BrowserTool
from tools.github_tool import GitHubTool
from tools.vercel_tool import VercelTool
from tools.image_tool import ImageTool
from tools.video_tool import VideoTool
from tools.telegram_tool import TelegramTool
from tools.monitoring_tool import MonitoringTool
from tools.project_registry_tool import ProjectRegistryTool

# Agents
from agents.webdev_agent import WebDevAgent
from agents.browser_agent import BrowserAgent
from agents.media_agent import MediaAgent
from agents.system_agent import SystemAgent

log = setup_logging("main")


def _create_directories():
    """Ensure workspace directories exist."""
    for d in (config.workspaces_dir, config.logs_dir, config.media_dir,
              config.tools_custom_dir, config.tool_backups_dir):
        d.mkdir(parents=True, exist_ok=True)
        log.debug(f"Dir ensured: {d}")


def _build_agents() -> dict[str, object]:
    """Instantiate agents and assign tools."""
    shell = ShellTool()
    filesystem = FileSystemTool()
    browser = BrowserTool()
    github = GitHubTool()
    vercel = VercelTool()
    image = ImageTool()
    video = VideoTool()
    telegram = TelegramTool()
    monitoring = MonitoringTool()
    project_registry = ProjectRegistryTool()

    # System agent — shell + filesystem + telegram + monitoring
    system_agent = SystemAgent()
    system_agent.register_tool(shell)
    system_agent.register_tool(filesystem)
    system_agent.register_tool(telegram)
    system_agent.register_tool(monitoring)
    system_agent.register_tool(project_registry)

    # WebDev agent — filesystem + github + vercel + shell + telegram
    webdev_agent = WebDevAgent()
    webdev_agent.register_tool(filesystem)
    webdev_agent.register_tool(github)
    webdev_agent.register_tool(vercel)
    webdev_agent.register_tool(shell)
    webdev_agent.register_tool(telegram)
    webdev_agent.register_tool(project_registry)

    # Browser agent — browser + filesystem + telegram
    browser_agent = BrowserAgent()
    browser_agent.register_tool(browser)
    browser_agent.register_tool(filesystem)
    browser_agent.register_tool(telegram)

    # Media agent — image + video + filesystem + shell + telegram
    media_agent = MediaAgent()
    media_agent.register_tool(image)
    media_agent.register_tool(video)
    media_agent.register_tool(filesystem)
    media_agent.register_tool(shell)
    media_agent.register_tool(telegram)
    media_agent.register_tool(project_registry)

    return {
        a.name: a
        for a in (system_agent, webdev_agent, browser_agent, media_agent)
    }


async def startup():
    """Async startup sequence."""
    log.info("=== Agent starting up ===")

    # 1. Directories
    _create_directories()

    # 2. Database
    await init_db()
    log.info("Database initialized")

    # 3. Load DB-registered tools
    await tool_registry.load_all()
    log.info(f"Tool registry loaded ({tool_registry.count()} tools)")

    # 3b. Refresh infrastructure snapshot early (prompt grounding)
    try:
        await collect_infrastructure_snapshot(source="startup")
        log.info("Infrastructure snapshot captured")
    except Exception as e:
        log.warning(f"Infrastructure snapshot failed at startup: {e}")

    # 4. Build agents + orchestrator
    agents = _build_agents()
    orchestrator = Orchestrator()
    for agent in agents.values():
        orchestrator.register_agent(agent)
    log.info(f"Orchestrator ready — {len(agents)} agents")

    # 5. Inject orchestrator into Telegram handlers
    set_orchestrator(orchestrator)

    # 6. Register scheduler jobs (before Telegram, so failures don't leave orphan tasks)
    from core.monitoring import monitoring_job
    from core.backup import backup_job
    from core.security_hardening import security_audit_job

    scheduler.register_handler("monitoring", monitoring_job)
    scheduler.register_handler("backup", backup_job)
    scheduler.register_handler("security_audit", security_audit_job)
    scheduler.register_handler("inventory", inventory_job)

    await scheduler.ensure_job(
        name="monitoring",
        handler_name="monitoring",
        interval_seconds=300,          # every 5 minutes
        description="Raccolta metriche sistema e alert soglie",
    )
    await scheduler.ensure_job(
        name="backup",
        handler_name="backup",
        interval_seconds=86400,        # daily
        description="Backup DB + .env + custom tools su GitHub",
    )
    await scheduler.ensure_job(
        name="security_audit",
        handler_name="security_audit",
        interval_seconds=21600,        # every 6 hours
        description="Controllo sicurezza VPS (SSH, firewall, fail2ban, porte)",
    )
    await scheduler.ensure_job(
        name="inventory",
        handler_name="inventory",
        interval_seconds=900,          # every 15 minutes
        description="Snapshot infrastrutturale persistente (servizi, risorse, integrazioni)",
    )

    scheduler.start()
    log.info("Scheduler started with 3 jobs")

    # 7. Build & start Telegram (last, so everything else is ready)
    app = build_app()
    log.info("Telegram bot starting polling…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    log.info("=== Agent is LIVE ===")
    return app


async def shutdown(app):
    """Graceful shutdown."""
    log.info("Shutting down…")

    scheduler.stop()

    try:
        if app.updater and app.updater.running:
            await app.updater.stop()
        if app.running:
            await app.stop()
        await app.shutdown()
    except Exception as e:
        log.warning(f"Telegram shutdown: {e}")

    await openrouter.close()
    await close_db()
    log.info("=== Agent stopped ===")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = None

    stop_event = asyncio.Event()

    def _handle_signal():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _handle_signal())

    try:
        app = loop.run_until_complete(startup())
        loop.run_until_complete(stop_event.wait())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")
    finally:
        if app:
            loop.run_until_complete(shutdown(app))
        # Let remaining tasks finish cleanly before closing the loop
        pending = asyncio.all_tasks(loop)
        if pending:
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
