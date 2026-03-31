"""File system tool — read, write, list, delete files and directories."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_fs")
ACTIVITY_FILE = "AGENT_ACTIVITY.md"


def _is_safe_path(path: str, action: str = "read") -> bool:
    """Ensure the path is under allowed directories.
    
    Read/list/exists are allowed on the agent's own source tree too,
    so the agent can inspect and understand its own code.
    Write/delete/move operations on core files go through security_agent approval.
    """
    resolved = Path(path).resolve()

    # Never expose secrets files under any circumstance
    if resolved.name == ".env" or resolved.suffix == ".env":
        return False

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

    # /etc/nginx is readable (agent can inspect configs); writes go via sudo cp workflow
    if str(resolved).startswith("/etc/nginx"):
        if action in ("read", "list", "exists"):
            return True

    return False


def _activity_dir_for(path: Path, action: str) -> Path:
    if action in ("list", "mkdir"):
        return path
    return path.parent


def _append_activity_log(action: str, target: Path, details: str = ""):
    """Append a compact operation trace in AGENT_ACTIVITY.md for the affected folder."""
    if target.name == ACTIVITY_FILE:
        return

    try:
        entry_dir = _activity_dir_for(target, action)
        if not _is_safe_path(str(entry_dir), "write"):
            return

        entry_dir.mkdir(parents=True, exist_ok=True)
        activity_path = entry_dir / ACTIVITY_FILE
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        detail_suffix = f" | {details}" if details else ""
        line = f"- {now} | action={action} | target={target}{detail_suffix}\n"
        with activity_path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        # Never fail the main filesystem action because of activity logging.
        log.debug(f"Activity log append skipped: {e}")


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
                _append_activity_log("read", p)
                return {"success": True, "content": content}

            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                _append_activity_log("write", p)
                return {"success": True, "message": f"Written {p}"}

            elif action == "append":
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(kwargs.get("content", ""))
                _append_activity_log("append", p)
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
                _append_activity_log("list", p, details=f"entries={len(entries)}")
                return {"success": True, "entries": entries}

            elif action == "delete":
                if not p.exists():
                    return {"success": False, "error": "Path not found."}
                _append_activity_log("delete", p)
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                return {"success": True, "message": f"Deleted {p}"}

            elif action == "mkdir":
                p.mkdir(parents=True, exist_ok=True)
                _append_activity_log("mkdir", p)
                return {"success": True, "message": f"Created directory {p}"}

            elif action == "exists":
                _append_activity_log("exists", p, details=f"exists={p.exists()}")
                return {"success": True, "exists": p.exists(), "is_dir": p.is_dir() if p.exists() else None}

            elif action == "move":
                dest = kwargs.get("destination")
                if not dest or not _is_safe_path(dest):
                    return {"success": False, "error": "Invalid or unsafe destination."}
                shutil.move(str(p), dest)
                _append_activity_log("move", p, details=f"destination={dest}")
                _append_activity_log("moved_here", Path(dest), details=f"source={p}")
                return {"success": True, "message": f"Moved {p} → {dest}"}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
