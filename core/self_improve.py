"""Self-improvement engine — safe framework for the agent to evolve its own capabilities.

Provides:
- safe_execute_and_iterate: run a command, detect errors, retry with fixes
- install_package: install pip/apt packages in the venv/system
- self_deploy: pull latest code, install deps, restart the service
- create_extension: safely add new tools/modules without touching core
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from config import config
from tg.notifications import notify
from utils.logging import setup_logging

log = setup_logging("self_improve")

VENV_PIP = "/srv/agent/app/.venv/bin/pip"
VENV_PYTHON = "/srv/agent/app/.venv/bin/python"
APP_DIR = "/srv/agent/app"


async def _run(cmd: str, cwd: str | None = None, timeout: int = 120) -> tuple[str, int]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd or APP_DIR,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(errors="replace")[:8000], proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        return "Command timed out", -1


async def install_package(package: str, system: bool = False) -> dict:
    """Install a Python package into the venv, or a system package via apt.

    Returns {"success": bool, "output": str}.
    """
    if system:
        cmd = f"apt-get install -y {package}"
    else:
        cmd = f"{VENV_PIP} install {package}"

    log.info(f"Installing package: {cmd}")
    out, rc = await _run(cmd, timeout=180)  
    success = rc == 0
    if success:
        log.info(f"Package installed: {package}")
    else:
        log.warning(f"Package install failed: {package} — {out[:200]}")
    return {"success": success, "output": out}


async def safe_execute_and_iterate(
    command: str,
    max_retries: int = 3,
    cwd: str | None = None,
) -> dict:
    """Execute a command. If it fails, return the error for the agent to analyze.

    This is meant to be called BY the agent via tool calls in a loop:
    1. Agent calls shell with a command
    2. If it fails, agent reads the error, decides a fix
    3. Agent applies the fix and retries

    This helper provides structured output to make the loop easier.
    """
    out, rc = await _run(command, cwd=cwd)
    return {
        "success": rc == 0,
        "returncode": rc,
        "output": out,
        "can_retry": rc != 0,
        "suggestion": _suggest_fix(out, rc) if rc != 0 else None,
    }


def _suggest_fix(output: str, rc: int) -> str | None:
    """Provide hints based on common error patterns."""
    out_lower = output.lower()

    if "modulenotfounderror" in out_lower or "no module named" in out_lower:
        # Extract module name
        for line in output.splitlines():
            if "No module named" in line:
                mod = line.split("No module named")[-1].strip().strip("'\"")
                return f"Missing module: {mod}. Try: {VENV_PIP} install {mod}"

    if "permission denied" in out_lower:
        return "Permission denied. The agent user may need access. Check file ownership."

    if "command not found" in out_lower:
        return "Command not found. May need to install the package: apt-get install <package>"

    if "disk quota" in out_lower or "no space left" in out_lower:
        return "Disk full. Clean up /tmp or old logs."

    if "connection refused" in out_lower:
        return "Connection refused. Check if the target service (DB/Redis) is running."

    if "syntax error" in out_lower or "syntaxerror" in out_lower:
        return "Python syntax error. Review the code for typos."

    return None


async def self_deploy() -> str:
    """Pull latest code from GitHub, install deps, restart the service.

    Returns a summary string.
    """
    steps = []

    # 1. Git pull
    out, rc = await _run("git pull --ff-only", cwd=APP_DIR)
    if rc == 0:
        steps.append("✅ git pull")
    else:
        steps.append(f"❌ git pull: {out[:150]}")
        return " | ".join(steps)

    # 2. Install requirements
    out, rc = await _run(f"{VENV_PIP} install -r requirements.txt", cwd=APP_DIR)
    if rc == 0:
        steps.append("✅ pip install")
    else:
        steps.append(f"⚠️ pip install: {out[:150]}")

    # 3. Restart service
    out, rc = await _run("systemctl restart agent")
    if rc == 0:
        steps.append("✅ riavviato")
    else:
        steps.append(f"❌ restart: {out[:150]}")

    summary = " | ".join(steps)
    await notify(f"🔄 <b>Self-deploy</b>\n{summary}")
    return summary


async def create_extension(
    name: str,
    code: str,
    extension_type: str = "tool",
) -> dict:
    """Safely create a new tool/module in the extensions directory.

    The extension is written to tools/custom/ (not core), so it can't
    break the base system. The agent can then register it dynamically.
    """
    if extension_type == "tool":
        target_dir = config.tools_custom_dir
    else:
        target_dir = Path(APP_DIR) / "extensions"

    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / f"{name}.py"

    # Safety: don't overwrite existing without backup
    if filepath.exists():
        backup = target_dir / f"{name}.py.bak"
        filepath.rename(backup)
        log.info(f"Backed up existing {filepath} → {backup}")

    filepath.write_text(code, encoding="utf-8")
    log.info(f"Created extension: {filepath}")

    # Validate syntax
    out, rc = await _run(f"{VENV_PYTHON} -c \"import ast; ast.parse(open('{filepath}').read())\"")
    if rc != 0:
        # Syntax error — restore backup if exists
        backup = target_dir / f"{name}.py.bak"
        if backup.exists():
            backup.rename(filepath)
        return {"success": False, "error": f"Syntax error in generated code: {out[:300]}"}

    return {"success": True, "path": str(filepath), "message": f"Extension '{name}' created at {filepath}"}
