"""CodeModifier — safely modify agent's own source code with backup + rollback."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from agents.security_agent import security_agent
from telegram.notifications import notify, notify_approval_needed
from telegram.handlers import request_approval
from utils.logging import setup_logging

log = setup_logging("code_modifier")


class CodeModifier:
    """Allows the agent to modify its own codebase safely."""

    async def modify_file(
        self,
        file_path: str,
        instructions: str,
        task_id: int | None = None,
    ) -> tuple[bool, str]:
        """
        Modify a source file according to instructions.
        Protected files require Telegram approval.
        Creates backup before modifying.
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"

        # Check if protected
        is_protected = any(
            file_path.endswith(p) or p in file_path
            for p in config.protected_paths
        )

        if is_protected:
            desc = (
                f"⚠️ <b>Modifica file protetto</b>\n"
                f"<code>{file_path}</code>\n"
                f"Istruzioni: {instructions[:200]}"
            )
            await notify_approval_needed(desc, task_id=task_id or 0)
            approved = await request_approval(task_id or 0)
            if not approved:
                return False, "Modifica rifiutata dall'utente"

        # Read current content
        current_code = path.read_text(encoding="utf-8")

        # Backup
        backup_dir = config.tool_backups_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = path.name.replace(".", f"_{ts}.")
        backup_path = backup_dir / backup_name
        shutil.copy2(path, backup_path)
        log.info(f"Backup created: {backup_path}")

        # Generate modified code
        model = get_model_for_task(TaskType.CODE_GENERATION)
        response = await openrouter.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code editor. Given source code and modification instructions, "
                        "return ONLY the complete modified file content. No explanations, no markdown "
                        "code blocks, just the raw Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"File: {file_path}\n\n"
                        f"Current code:\n```python\n{current_code}\n```\n\n"
                        f"Instructions: {instructions}\n\n"
                        f"Return the complete modified file."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=16384,
        )

        new_code = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Strip markdown code fences if present
        if new_code.startswith("```"):
            lines = new_code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_code = "\n".join(lines)

        # Security check on new code
        is_safe, issues = security_agent.validate_generated_code(new_code)
        if not is_safe:
            log.warning(f"Security issues in modified code: {issues}")
            return False, f"Modifiche bloccate per problemi di sicurezza:\n" + "\n".join(issues)

        # Write modified file
        path.write_text(new_code, encoding="utf-8")
        log.info(f"Modified file: {file_path}")

        return True, f"File modificato con successo. Backup: {backup_path}"

    async def rollback(self, file_path: str) -> tuple[bool, str]:
        """Rollback a file to its most recent backup."""
        path = Path(file_path)
        backup_dir = config.tool_backups_dir
        stem = path.stem

        # Find latest backup
        backups = sorted(
            backup_dir.glob(f"{stem}_*.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not backups:
            return False, f"Nessun backup trovato per {file_path}"

        latest = backups[0]
        shutil.copy2(latest, path)
        log.info(f"Rolled back {file_path} from {latest}")
        return True, f"Rollback completato da {latest.name}"


# Singleton
code_modifier = CodeModifier()
