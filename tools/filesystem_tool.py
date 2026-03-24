"""File system tool — read, write, list, delete files and directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_fs")


def _is_safe_path(path: str, action: str = "read") -> bool:
    """Ensure the path is under allowed directories.
    
    Read/list/exists are allowed on the agent's own source tree too,
    so the agent can inspect and understand its own code.
    Write/delete/move operations on core files go through security_agent approval.
    """
    resolved = Path(path).resolve()

    # Always writable roots
    writable_roots = [
        config.workspaces_dir.resolve(),
        config.media_dir.resolve(),
        config.tools_custom_dir.resolve(),
        config.tool_backups_dir.resolve(),
        Path(config.logs_dir).resolve(),
    ]
    if any(str(resolved).startswith(str(root)) for root in writable_roots):
        return True

    # Agent source tree: readable (read/list/exists), writes checked by security_agent
    agent_root = Path("/srv/agent/app").resolve()
    if str(resolved).startswith(str(agent_root)):
        return True  # security_agent handles write protection via PROTECTED_PATHS

    # /tmp is allowed for transient operations
    if str(resolved).startswith("/tmp"):
        return True

    return False


class FileSystemTool(BaseTool):
    name = "filesystem"
    description = "Read, write, list, or delete files and directories on the VPS."
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "append", "list", "delete", "mkdir", "exists", "move"],
                    "description": "The file operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write/append actions).",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path (for move action).",
                },
            },
            "required": ["action", "path"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action: str = kwargs["action"]
        path: str = kwargs["path"]

        if not _is_safe_path(path, action):
            return {"success": False, "error": f"Access denied: path '{path}' is outside allowed directories."}

        p = Path(path)

        try:
            if action == "read":
                if not p.exists():
                    return {"success": False, "error": "File not found."}
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                return {"success": True, "content": content}

            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"success": True, "message": f"Written {p}"}

            elif action == "append":
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(kwargs.get("content", ""))
                return {"success": True, "message": f"Appended to {p}"}

            elif action == "list":
                if not p.exists():
                    return {"success": False, "error": "Directory not found."}
                entries = []
                for entry in sorted(p.iterdir()):
                    entries.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size": entry.stat().st_size if entry.is_file() else None,
                    })
                return {"success": True, "entries": entries}

            elif action == "delete":
                if not p.exists():
                    return {"success": False, "error": "Path not found."}
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                return {"success": True, "message": f"Deleted {p}"}

            elif action == "mkdir":
                p.mkdir(parents=True, exist_ok=True)
                return {"success": True, "message": f"Created directory {p}"}

            elif action == "exists":
                return {"success": True, "exists": p.exists(), "is_dir": p.is_dir() if p.exists() else None}

            elif action == "move":
                dest = kwargs.get("destination")
                if not dest or not _is_safe_path(dest):
                    return {"success": False, "error": "Invalid or unsafe destination."}
                shutil.move(str(p), dest)
                return {"success": True, "message": f"Moved {p} → {dest}"}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
