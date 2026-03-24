"""Security Hardening — one-time setup and periodic audit for VPS security."""

from __future__ import annotations

import asyncio

from tg.notifications import notify
from utils.logging import setup_logging

log = setup_logging("security_hardening")


async def _run_cmd(cmd: str) -> tuple[str, int]:
    """Run a shell command and return (output, returncode)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace")[:5000], proc.returncode


async def security_audit_job() -> str:
    """Scheduled job: check security posture. No AI cost."""
    issues = []
    checks_ok = []

    # 1. SSH password auth check
    out, _ = await _run_cmd("grep -i '^PasswordAuthentication' /etc/ssh/sshd_config 2>/dev/null || echo 'NOT_FOUND'")
    if "yes" in out.lower():
        issues.append("🔴 SSH password auth abilitata")
    elif "no" in out.lower():
        checks_ok.append("SSH password auth disabilitata")
    else:
        issues.append("🟡 SSH PasswordAuthentication non configurata esplicitamente")

    # 2. Firewall status
    out, rc = await _run_cmd("ufw status 2>/dev/null || echo 'UFW_NOT_INSTALLED'")
    if "inactive" in out.lower() or "UFW_NOT_INSTALLED" in out:
        issues.append("🔴 Firewall (UFW) non attivo")
    elif "active" in out.lower():
        checks_ok.append("UFW attivo")

    # 3. Fail2ban
    out, rc = await _run_cmd("systemctl is-active fail2ban 2>/dev/null || echo 'NOT_ACTIVE'")
    if "active" in out.strip() and "NOT_ACTIVE" not in out:
        checks_ok.append("fail2ban attivo")
    else:
        issues.append("🟡 fail2ban non attivo")

    # 4. Docker containers running
    out, rc = await _run_cmd("docker ps --format '{{.Names}}: {{.Status}}' 2>/dev/null")
    if rc == 0 and out.strip():
        checks_ok.append(f"Docker: {out.strip().count(chr(10)) + 1} container")

    # 5. Open ports
    out, rc = await _run_cmd("ss -tlnp 2>/dev/null | grep LISTEN | wc -l")
    port_count = out.strip()
    checks_ok.append(f"Porte in ascolto: {port_count}")

    # 6. Disk usage warning
    out, rc = await _run_cmd("df -h / | tail -1 | awk '{print $5}' | tr -d '%'")
    try:
        disk_pct = int(out.strip())
        if disk_pct > 90:
            issues.append(f"🔴 Disco quasi pieno: {disk_pct}%")
        elif disk_pct > 80:
            issues.append(f"🟡 Disco al {disk_pct}%")
        else:
            checks_ok.append(f"Disco: {disk_pct}%")
    except ValueError:
        pass

    # 7. Failed SSH logins (last hour)
    out, rc = await _run_cmd(
        "journalctl -u ssh --since '1 hour ago' --no-pager 2>/dev/null | grep -c 'Failed password' || echo 0"
    )
    try:
        failed = int(out.strip())
        if failed > 10:
            issues.append(f"🔴 {failed} tentativi SSH falliti nell'ultima ora")
        elif failed > 0:
            issues.append(f"🟡 {failed} tentativi SSH falliti nell'ultima ora")
    except ValueError:
        pass

    # Send alert only if there are issues
    if issues:
        msg = "🔒 <b>Security Audit</b>\n"
        msg += "\n".join(issues)
        if checks_ok:
            msg += "\n\n✅ " + "\n✅ ".join(checks_ok)
        await notify(msg)

    return f"{len(checks_ok)} ok, {len(issues)} issues"
