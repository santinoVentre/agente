"""ToolFactory — generates new tools from natural language descriptions using LLMs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import config
from core.dependency_manager import dependency_manager
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from core.tool_registry import tool_registry
from core.tool_validator import tool_validator
from db.models import RiskLevel
from tg.notifications import notify, notify_done, notify_error
from utils.logging import setup_logging

log = setup_logging("tool_factory")

TOOL_GENERATION_PROMPT = """\
You are a Python tool generator for an agent infrastructure. You generate self-contained Python tool classes.

RULES:
1. The tool MUST subclass BaseTool from tools.base_tool
2. The tool class MUST define: name, description, risk_level, get_parameters_schema(), execute()
3. execute() is an async method that receives **kwargs and returns dict[str, Any]
4. Return {"success": True, ...} on success, {"success": False, "error": "..."} on failure
5. Use standard library or well-known pip packages only
6. NEVER use os.system(), eval(), exec(), or raw subprocess calls
7. For HTTP requests use httpx (async)
8. For file operations use pathlib
9. Include proper error handling with try/except
10. Include type hints

You must respond with a JSON object containing EXACTLY these keys:
{
  "name": "tool_name_snake_case",
  "description": "What the tool does",
  "risk_level": "low|medium|high",
  "dependencies": ["list", "of", "pip", "packages"],
  "code": "full Python source code as a string",
  "test_code": "async test code (will be run inside an async function)"
}
"""


class ToolFactory:
    """Generates new Python tools from natural language descriptions."""

    async def create_tool(
        self,
        description: str,
        task_id: int | None = None,
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """
        Generate a new tool from a natural language description.
        Uses expensive model for code generation.
        Returns (success, message).
        """
        model = get_model_for_task(TaskType.CODE_GENERATION)
        log.info(f"Generating tool: '{description}' using {model}")

        last_error = ""
        for attempt in range(max_retries):
            messages = [
                {"role": "system", "content": TOOL_GENERATION_PROMPT},
                {"role": "user", "content": f"Create a tool that: {description}"},
            ]
            if attempt > 0 and last_error:
                messages.append({
                    "role": "user",
                    "content": f"The previous attempt failed with:\n{last_error}\nPlease fix the issues and try again.",
                })

            await notify(f"🔧 Generazione tool (tentativo {attempt+1}/{max_retries})...")

            response = await openrouter.chat(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            tool_spec = self._extract_json(content)
            if not tool_spec:
                last_error = "Failed to parse JSON from LLM response"
                log.warning(f"Attempt {attempt+1}: {last_error}")
                continue

            name = tool_spec.get("name", "")
            code = tool_spec.get("code", "")
            test_code = tool_spec.get("test_code", "pass")
            deps = tool_spec.get("dependencies", [])
            risk_str = tool_spec.get("risk_level", "low")
            risk = getattr(RiskLevel, risk_str.upper(), RiskLevel.LOW)

            if not name or not code:
                last_error = "Missing name or code in response"
                continue

            # Validate
            passed, details = await tool_validator.full_validation(code, test_code)
            if not passed:
                last_error = details
                log.warning(f"Attempt {attempt+1} validation failed: {details}")
                continue

            # Install dependencies if needed
            if deps:
                success, msg = await dependency_manager.install_packages(
                    deps, task_id=task_id, require_approval=True
                )
                if not success:
                    return False, f"Dependency installation failed: {msg}"

            # Save tool file
            tool_dir = config.tools_custom_dir
            tool_dir.mkdir(parents=True, exist_ok=True)
            tool_path = tool_dir / f"{name}.py"
            tool_path.write_text(code, encoding="utf-8")

            # Register in DB
            schema = {
                "name": name,
                "description": tool_spec.get("description", description),
                "parameters": self._extract_params_from_code(code),
            }
            await tool_registry.register_generated(
                name=name,
                description=tool_spec.get("description", description),
                schema=schema,
                source_path=str(tool_path),
                risk_level=risk,
                created_by_model=model,
                dependencies=deps,
            )

            await notify_done(
                f"Tool '{name}' creato",
                f"v1 | Risk: {risk.value} | Deps: {deps or 'none'}"
            )
            log.info(f"Tool '{name}' created successfully at {tool_path}")
            return True, f"Tool '{name}' creato e registrato con successo."

        return False, f"Creazione tool fallita dopo {max_retries} tentativi. Ultimo errore: {last_error}"

    async def modify_tool(
        self,
        tool_name: str,
        modification_desc: str,
        task_id: int | None = None,
    ) -> tuple[bool, str]:
        """Modify an existing generated tool."""
        tool_path = config.tools_custom_dir / f"{tool_name}.py"
        if not tool_path.exists():
            return False, f"Tool '{tool_name}' non trovato in {tool_path}"

        current_code = tool_path.read_text(encoding="utf-8")
        model = get_model_for_task(TaskType.CODE_GENERATION)

        response = await openrouter.chat(
            model=model,
            messages=[
                {"role": "system", "content": TOOL_GENERATION_PROMPT},
                {"role": "user", "content": (
                    f"Modify the following tool according to these instructions: {modification_desc}\n\n"
                    f"Current code:\n```python\n{current_code}\n```"
                )},
            ],
            temperature=0.2,
            max_tokens=8192,
        )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_spec = self._extract_json(content)
        if not tool_spec or not tool_spec.get("code"):
            return False, "Failed to generate modified code"

        new_code = tool_spec["code"]
        test_code = tool_spec.get("test_code", "pass")

        passed, details = await tool_validator.full_validation(new_code, test_code)
        if not passed:
            return False, f"Validation failed: {details}"

        # Backup old version
        backup_dir = config.tool_backups_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(tool_path, backup_dir / f"{tool_name}_{ts}.py")

        # Write new version
        tool_path.write_text(new_code, encoding="utf-8")

        # Update registry
        deps = tool_spec.get("dependencies", [])
        if deps:
            await dependency_manager.install_packages(deps, task_id=task_id)

        schema = {
            "name": tool_name,
            "description": tool_spec.get("description", ""),
            "parameters": self._extract_params_from_code(new_code),
        }
        await tool_registry.register_generated(
            name=tool_name,
            description=tool_spec.get("description", ""),
            schema=schema,
            source_path=str(tool_path),
            risk_level=getattr(RiskLevel, tool_spec.get("risk_level", "low").upper(), RiskLevel.LOW),
            created_by_model=model,
            dependencies=deps,
        )

        log.info(f"Tool '{tool_name}' modified and re-registered")
        return True, f"Tool '{tool_name}' modificato con successo."

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON object from LLM response (may be wrapped in markdown)."""
        # Try to find JSON block in markdown
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)

        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _extract_params_from_code(self, code: str) -> dict:
        """Best-effort extraction of parameters schema from tool code."""
        # Look for get_parameters_schema method and try to extract the return dict
        match = re.search(
            r"def get_parameters_schema.*?return\s+(\{.*?\})\s*$",
            code,
            re.DOTALL | re.MULTILINE,
        )
        if match:
            try:
                # This won't always work for complex dicts, fallback to basic schema
                return json.loads(match.group(1).replace("'", '"'))
            except (json.JSONDecodeError, Exception):
                pass
        return {"type": "object", "properties": {}}


# Singleton
tool_factory = ToolFactory()
