"""Browser tool — Playwright-based headless browsing, screenshots, scraping."""

from __future__ import annotations

from typing import Any

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_browser")


class BrowserTool(BaseTool):
    name = "browser"
    description = (
        "Navigate to URLs, take screenshots, extract page text, download files "
        "using a headless Chromium browser (Playwright)."
    )
    risk_level = RiskLevel.MEDIUM

    def __init__(self):
        self._browser = None
        self._playwright = None

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "screenshot", "get_text", "get_html", "download", "click", "fill"],
                    "description": "Browser action to perform.",
                },
                "url": {"type": "string", "description": "URL to navigate to."},
                "selector": {"type": "string", "description": "CSS selector for click/fill actions."},
                "value": {"type": "string", "description": "Value for fill action."},
                "save_path": {"type": "string", "description": "Path to save downloaded file or screenshot."},
            },
            "required": ["action"],
        }

    async def _ensure_browser(self):
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs["action"]
        try:
            await self._ensure_browser()
            page = await self._browser.new_page()

            if action == "navigate":
                url = kwargs.get("url", "")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                title = await page.title()
                await page.close()
                return {"success": True, "title": title, "url": url}

            elif action == "screenshot":
                url = kwargs.get("url", "")
                save_path = kwargs.get("save_path", str(config.media_dir / "screenshot.png"))
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.screenshot(path=save_path, full_page=True)
                await page.close()
                return {"success": True, "path": save_path}

            elif action == "get_text":
                url = kwargs.get("url", "")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                text = await page.inner_text("body")
                await page.close()
                return {"success": True, "text": text[:20000]}

            elif action == "get_html":
                url = kwargs.get("url", "")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                html = await page.content()
                await page.close()
                return {"success": True, "html": html[:50000]}

            elif action == "download":
                url = kwargs.get("url", "")
                save_path = kwargs.get("save_path", str(config.media_dir / "download"))
                resp = await page.request.get(url)
                body = await resp.body()
                from pathlib import Path
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
                await page.close()
                return {"success": True, "path": save_path, "size": len(body)}

            else:
                await page.close()
                return {"success": False, "error": f"Action '{action}' not fully implemented yet."}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
