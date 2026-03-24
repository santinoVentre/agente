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
    """Scheduled job: check security posture and suggest fixes. No AI cost."""
    issues = []
    checks_ok = []
    fixes = []

    # 1. SSH password auth check
    out, _ = await _run_cmd("grep -i '^PasswordAuthentication' /etc/ssh/sshd_config 2>/dev/null || echo 'NOT_FOUND'")
    if "yes" in out.lower():
        issues.append("🔴 SSH password auth abilitata")
        fixes.append("Fix: <code>sudo sed -i 's/^PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && sudo systemctl restart sshd</code>")
    elif "no" in out.lower():
        checks_ok.append("SSH password auth disabilitata")
    else:
        issues.append("🟡 SSH PasswordAuthentication non configurata esplicitamente")

    # 2. Firewall status
    out, rc = await _run_cmd("ufw status 2>/dev/null || echo 'UFW_NOT_INSTALLED'")
    if "UFW_NOT_INSTALLED" in out:
        issues.append("🔴 UFW non installato")
        fixes.append("Fix: <code>sudo apt-get install -y ufw && sudo ufw default deny incoming && sudo ufw default allow outgoing && sudo ufw allow 22/tcp && sudo ufw --force enable</code>")
    elif "inactive" in out.lower():
        issues.append("🔴 Firewall (UFW) non attivo")
        fixes.append("Fix: <code>sudo ufw default deny incoming && sudo ufw default allow outgoing && sudo ufw allow 22/tcp && sudo ufw --force enable</code>")
    elif "active" in out.lower():
        checks_ok.append("UFW attivo")

    # 3. Fail2ban
    out, rc = await _run_cmd("systemctl is-active fail2ban 2>/dev/null || echo 'NOT_ACTIVE'")
    if "active" in out.strip() and "NOT_ACTIVE" not in out:
        checks_ok.append("fail2ban attivo")
    else:
        # Check if installed
        out2, _ = await _run_cmd("which fail2ban-server 2>/dev/null || echo 'NOT_INSTALLED'")
        if "NOT_INSTALLED" in out2:
            issues.append("🟡 fail2ban non installato")
            fixes.append("Fix: <code>sudo apt-get install -y fail2ban && sudo systemctl enable --now fail2ban</code>")
        else:
            issues.append("🟡 fail2ban installato ma non attivo")
            fixes.append("Fix: <code>sudo systemctl enable --now fail2ban</code>")

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
            fixes.append("Fix: pulisci /tmp, vecchi log, o contenuto non necessario")
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

    # 8. Agent service health
    out, rc = await _run_cmd("systemctl is-active agent 2>/dev/null")
    if "active" in out.strip():
        checks_ok.append("Servizio agent attivo")
    else:
        issues.append("🔴 Servizio agent non attivo!")

    # Build report — always send, with both issues and OK checks
    msg = "🔒 <b>Security Audit</b>\n"
    if issues:
        msg += "\n".join(issues) + "\n"
    if fixes:
        msg += "\n<b>Fix suggeriti:</b>\n" + "\n".join(fixes) + "\n"
    if checks_ok:
        msg += "\n✅ " + "\n✅ ".join(checks_ok)

    if issues:
        await notify(msg)

    return f"{len(checks_ok)} ok, {len(issues)} issues"
