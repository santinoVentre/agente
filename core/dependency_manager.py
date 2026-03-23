"""DependencyManager — install pip packages required by generated tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from config import config
from tg.notifications import notify_approval_needed
from tg.handlers import request_approval
from utils.logging import setup_logging

log = setup_logging("dependency_manager")

REQUIREMENTS_CUSTOM = Path(__file__).parent.parent / "requirements-custom.txt"


class DependencyManager:
    """Manages pip dependencies for custom tools."""

    async def check_installed(self, package: str) -> bool:
        """Check if a pip package is installed."""
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", f"import importlib; importlib.import_module('{package}')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def install_packages(
        self,
        packages: list[str],
        task_id: int | None = None,
        require_approval: bool = True,
    ) -> tuple[bool, str]:
        """
        Install pip packages, optionally requiring user approval.
        Returns (success, message).
        """
        if not packages:
            return True, "No packages to install"

        # Filter already installed
        to_install = []
        for pkg in packages:
            # normalize package name for import check
            import_name = pkg.split("==")[0].split(">=")[0].replace("-", "_").lower()
            if not await self.check_installed(import_name):
                to_install.append(pkg)

        if not to_install:
            return True, "All packages already installed"

        # Request approval
        if require_approval:
            desc = f"Installare pacchetti pip:\n<code>{', '.join(to_install)}</code>"
            await notify_approval_needed(desc, task_id=task_id or 0)
            approved = await request_approval(task_id or 0)
            if not approved:
                return False, "Package installation rejected by user"

        # Install
        log.info(f"Installing packages: {to_install}")
        proc = await asyncio.create_subprocess_exec(
            "pip", "install", *to_install,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            error = stderr.decode("utf-8", errors="replace")
            log.error(f"pip install failed: {error}")
            return False, f"Installation failed: {error[:500]}"

        # Update requirements-custom.txt
        self._update_requirements_file(to_install)

        log.info(f"Successfully installed: {to_install}")
        return True, f"Installed: {', '.join(to_install)}"

    def _update_requirements_file(self, packages: list[str]):
        """Append newly installed packages to requirements-custom.txt."""
        existing = set()
        if REQUIREMENTS_CUSTOM.exists():
            existing = {
                line.strip().split("==")[0].lower()
                for line in REQUIREMENTS_CUSTOM.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            }

        with REQUIREMENTS_CUSTOM.open("a", encoding="utf-8") as f:
            for pkg in packages:
                pkg_name = pkg.split("==")[0].split(">=")[0].lower()
                if pkg_name not in existing:
                    f.write(f"{pkg}\n")


# Singleton
dependency_manager = DependencyManager()
