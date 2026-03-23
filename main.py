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

    # System agent — shell + filesystem
    system_agent = SystemAgent()
    system_agent.register_tool(shell)
    system_agent.register_tool(filesystem)

    # WebDev agent — filesystem + github + vercel + shell
    webdev_agent = WebDevAgent()
    webdev_agent.register_tool(filesystem)
    webdev_agent.register_tool(github)
    webdev_agent.register_tool(vercel)
    webdev_agent.register_tool(shell)

    # Browser agent — browser + filesystem
    browser_agent = BrowserAgent()
    browser_agent.register_tool(browser)
    browser_agent.register_tool(filesystem)

    # Media agent — image + video + filesystem + shell
    media_agent = MediaAgent()
    media_agent.register_tool(image)
    media_agent.register_tool(video)
    media_agent.register_tool(filesystem)
    media_agent.register_tool(shell)

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

    # 4. Build agents + orchestrator
    agents = _build_agents()
    orchestrator = Orchestrator()
    for agent in agents.values():
        orchestrator.register_agent(agent)
    log.info(f"Orchestrator ready — {len(agents)} agents")

    # 5. Inject orchestrator into Telegram handlers
    set_orchestrator(orchestrator)

    # 6. Build & start Telegram
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
        loop.close()


if __name__ == "__main__":
    main()
