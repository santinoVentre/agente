"""Browser tool — Playwright-based headless browsing, screenshots, scraping."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_browser")

# Patterns that match internal / metadata / loopback addresses
_SSRF_BLOCKED_HOST_RE = re.compile(
    r"^("
    r"localhost"
    r"|127\.\d+\.\d+\.\d+"
    r"|0\.0\.0\.0"
    r"|::1"
    r"|169\.254\.\d+\.\d+"          # AWS/GCP link-local metadata
    r"|10\.\d+\.\d+\.\d+"
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+"
    r"|192\.168\.\d+\.\d+"
    r"|fd[0-9a-f:]+"                  # ULA IPv6
    r")$",
    re.IGNORECASE,
)


def _is_ssrf_blocked(url: str) -> bool:
    """Return True if the URL targets an internal or metadata address."""
    try:
        host = urlparse(url).hostname or ""
        return bool(_SSRF_BLOCKED_HOST_RE.match(host))
    except Exception:
        return True  # fail-safe: block unparseable URLs


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
        action = kwargs.get("action")
        if not action:
            return {"success": False, "error": "Missing required parameter(s): action.", "failure_kind": "invalid_args"}

        # SSRF guard: reject requests to internal/metadata addresses
        url_param = kwargs.get("url", "")
        if url_param and _is_ssrf_blocked(url_param):
            log.warning(f"[browser] SSRF attempt blocked: {url_param}")
            return {
                "success": False,
                "error": f"URL bloccato: l'indirizzo '{url_param}' punta a una rete interna o di metadati.",
                "failure_kind": "blocked",
            }

        try:
            await self._ensure_browser()
            page = await self._browser.new_page()

            if action == "navigate":
                url = kwargs.get("url", "")
                if not url:
                    await page.close()
                    return {"success": False, "error": "Missing required parameter(s): url.", "failure_kind": "invalid_args"}
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                title = await page.title()
                await page.close()
                return {"success": True, "title": title, "url": url}

            elif action == "screenshot":
                url = kwargs.get("url", "")
                if not url:
                    await page.close()
                    return {"success": False, "error": "Missing required parameter(s): url.", "failure_kind": "invalid_args"}
                save_path = kwargs.get("save_path", str(config.media_dir / "screenshot.png"))
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.screenshot(path=save_path, full_page=True)
                await page.close()
                return {"success": True, "path": save_path}

            elif action == "get_text":
                url = kwargs.get("url", "")
                if not url:
                    await page.close()
                    return {"success": False, "error": "Missing required parameter(s): url.", "failure_kind": "invalid_args"}
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                text = await page.inner_text("body")
                await page.close()
                return {"success": True, "text": text[:20000]}

            elif action == "get_html":
                url = kwargs.get("url", "")
                if not url:
                    await page.close()
                    return {"success": False, "error": "Missing required parameter(s): url.", "failure_kind": "invalid_args"}
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                html = await page.content()
                await page.close()
                return {"success": True, "html": html[:50000]}

            elif action == "download":
                url = kwargs.get("url", "")
                if not url:
                    await page.close()
                    return {"success": False, "error": "Missing required parameter(s): url.", "failure_kind": "invalid_args"}
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
                return {"success": False, "error": f"Action '{action}' not fully implemented yet.", "failure_kind": "invalid_args"}

        except Exception as e:
            return {"success": False, "error": str(e), "failure_kind": "runtime_error"}

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
