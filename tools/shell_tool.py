"""Shell tool — execute shell commands on the VPS with safety checks."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_shell")

# Commands that always require approval
_DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r /", "mkfs", "dd if=", "format",
    "> /dev/", "chmod 777", "curl | sh", "wget | sh",
    "DROP TABLE", "DROP DATABASE", "TRUNCATE",
    "shutdown", "reboot", "init 0", "init 6",
    "iptables -F", "ufw disable",
    "passwd", "useradd", "userdel", "usermod",
    ":(){:|:&};:",  # fork bomb
    "cat .env", "cat /.env", "less .env", "more .env",  # block reading secrets files
]


def _assess_risk(command: str) -> RiskLevel:
    cmd_lower = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.lower() in cmd_lower:
            return RiskLevel.CRITICAL
    if any(kw in cmd_lower for kw in ["rm ", "kill ", "pkill ", "systemctl stop", "apt remove"]):
        return RiskLevel.HIGH
    if any(kw in cmd_lower for kw in ["apt install", "pip install", "npm install", "systemctl restart", "docker"]):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class ShellTool(BaseTool):
    name = "shell"
    description = "Execute a shell command on the VPS and return stdout/stderr."
    risk_level = RiskLevel.MEDIUM  # base risk; actual risk is assessed per-command

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60).",
                    "default": 60,
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: agent workspaces dir).",
                },
            },
            "required": ["command"],
        }

    def get_command_risk(self, command: str) -> RiskLevel:
        return _assess_risk(command)

    async def execute(self, **kwargs) -> dict[str, Any]:
        command: str = kwargs["command"]
        timeout: int = min(kwargs.get("timeout", 60), config.max_shell_timeout)
        cwd: str = kwargs.get("cwd", str(config.workspaces_dir))

        log.info(f"Shell exec: {command} (cwd={cwd}, timeout={timeout}s)")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[:10000],
                "stderr": stderr.decode("utf-8", errors="replace")[:5000],
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
