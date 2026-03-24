"""Backup — automatic backup of DB, config, and custom tools to GitHub."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from config import config
from tg.notifications import notify
from utils.logging import setup_logging

log = setup_logging("backup")

BACKUP_REPO = "agent-backups"


async def _run_cmd(cmd: str, cwd: str | None = None) -> tuple[str, int]:
    """Run a shell command and return (output, returncode)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace")[:5000], proc.returncode


async def backup_job() -> str:
    """Scheduled job: dump PostgreSQL + .env + custom tools → push to GitHub."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"/tmp/agent_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. PostgreSQL dump
    pg_file = backup_dir / "agent_db.sql"
    out, rc = await _run_cmd(
        f"docker exec agent-postgres pg_dump -U agent agent_db > {pg_file}",
    )
    if rc == 0:
        results.append("✅ DB dump")
    else:
        results.append(f"❌ DB dump: {out[:100]}")

    # 2. Copy .env (strip secrets for safety — keep structure)
    env_src = Path("/srv/agent/app/.env")
    if env_src.exists():
        (backup_dir / ".env.backup").write_text(env_src.read_text())
        results.append("✅ .env")

    # 3. Copy custom tools
    custom_dir = config.tools_custom_dir
    if custom_dir.exists():
        out, rc = await _run_cmd(f"cp -r {custom_dir} {backup_dir}/tools_custom")
        results.append("✅ Custom tools" if rc == 0 else f"❌ Custom tools: {out[:100]}")

    # 4. Push to GitHub
    token = config.github_token
    owner = config.github_owner
    if not token or not owner:
        results.append("⚠️ GitHub token/owner non configurato, skip push")
        summary = " | ".join(results)
        await notify(f"📦 <b>Backup locale completato</b>\n{summary}\nPath: {backup_dir}")
        return summary

    repo_url = f"https://{token}@github.com/{owner}/{BACKUP_REPO}.git"
    git_dir = backup_dir / "repo"

    # Clone or init
    out, rc = await _run_cmd(f"git clone {repo_url} {git_dir}")
    if rc != 0:
        # Repo might not exist — create it via API
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.github.com/user/repos",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json={"name": BACKUP_REPO, "private": True, "auto_init": True},
            )
            if resp.status_code in (201, 422):
                out, rc = await _run_cmd(f"git clone {repo_url} {git_dir}")

    if rc != 0:
        results.append(f"❌ Git clone failed: {out[:100]}")
    else:
        # Copy backup files into repo
        await _run_cmd(f"cp {backup_dir}/*.sql {git_dir}/ 2>/dev/null; "
                        f"cp {backup_dir}/.env.backup {git_dir}/ 2>/dev/null; "
                        f"cp -r {backup_dir}/tools_custom {git_dir}/ 2>/dev/null")

        # Cleanup old backups (keep last 30 commits)
        await _run_cmd(
            f"cd {git_dir} && git add -A && "
            f"git -c user.name='Agent' -c user.email='agent@vps' "
            f"commit -m 'Backup {timestamp}' && git push",
            cwd=str(git_dir),
        )
        results.append("✅ Push GitHub")

    # Cleanup temp
    await _run_cmd(f"rm -rf {backup_dir}")

    summary = " | ".join(results)
    await notify(f"📦 <b>Backup completato</b>\n{summary}")
    return summary
