"""Security Agent (Guardian) — validates ALL actions before execution."""

from __future__ import annotations

import re
from typing import Any

from config import config
from db.models import ActionVerdict, RiskLevel
from utils.logging import setup_logging

log = setup_logging("security_agent")

# Patterns that indicate dangerous shell operations
_SHELL_BLOCKLIST = [
    r"rm\s+-rf\s+/(?!\s*srv/agent)",  # rm -rf / (but allow within /srv/agent)
    r"mkfs\.",
    r"dd\s+if=",
    r"\d*>\s*/dev/(?!null\b)",
    r"chmod\s+777\s+/",
    r"curl\s+.*\|\s*(?:ba)?sh",
    r"wget\s+.*\|\s*(?:ba)?sh",
    r":\(\)\{.*\}",  # fork bomb
    r"init\s+[06]",
    r"shutdown",
    r"reboot",
    r"iptables\s+-F",
]

# Whitelisted sudo commands — regex-anchored to prevent prefix-bypass attacks
_SUDO_REGEX_WHITELIST: list[re.Pattern] = [
    re.compile(r"^sudo\s+apt(?:-get)?\s+(update|install\s+-y)\b"),
    re.compile(r"^sudo\s+systemctl\s+(reload|restart|start|stop|status)\s+\w"),
    # cp only from /tmp/agent-<safe-name> to specific nginx paths
    re.compile(r"^sudo\s+cp\s+/tmp/agent-[\w.-]+\s+/etc/nginx/(?:sites-available|conf\.d)/[\w.-]+$"),
    re.compile(r"^sudo\s+ln\s+-sf\s+/etc/nginx/sites-available/[\w.-]+\s+/etc/nginx/sites-enabled/[\w.-]+$"),
    re.compile(r"^sudo\s+rm\s+/etc/nginx/sites-enabled/[\w.-]+$"),
    re.compile(r"^sudo\s+certbot\b"),
    re.compile(r"^sudo\s+nginx\s+-t$"),
    re.compile(r"^sudo\s+ufw\s+(allow|deny|status|enable)\b"),
]

# Imports that are forbidden in generated tool code
_FORBIDDEN_IMPORTS = [
    "os.system",
    "subprocess.call(",
    "subprocess.Popen(",
    "__import__",
    "exec(",
    "eval(",
    "compile(",
    "importlib.import_module",
]

# File paths the agent must never touch without explicit approval
_ALWAYS_PROTECTED_PATTERNS = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/root/", "/boot/",
    "id_rsa", "id_ed25519",
    ".env",          # Never expose environment secrets
]

_READ_ONLY_DIAGNOSTIC_TOKENS = {
    "find",
    "grep",
    "egrep",
    "fgrep",
    "cat",
    "head",
    "tail",
    "sed",
    "awk",
    "ls",
    "pwd",
    "stat",
    "file",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "realpath",
    "readlink",
    "which",
    "whereis",
    "env",
    "printenv",
    "echo",
    "node",
    "npm",
    "pnpm",
    "yarn",
    "git",
}

_READ_ONLY_BLOCKED_SNIPPETS = (
    " rm ",
    "mv ",
    "cp ",
    "chmod ",
    "chown ",
    "mkdir ",
    "rmdir ",
    "touch ",
    "tee ",
    "install ",
    "apt ",
    "apt-get ",
    "pip ",
    "npm install",
    "pnpm add",
    "yarn add",
    "systemctl ",
    "docker ",
    "curl ",
    "wget ",
)


class SecurityAgent:
    """Intercepts and validates every action before it runs."""

    def _is_read_only_diagnostic_command(self, command: str) -> bool:
        cmd = command.strip().lower()
        if not cmd:
            return False
        if any(snippet in f" {cmd} " for snippet in _READ_ONLY_BLOCKED_SNIPPETS):
            return False

        normalized = re.sub(r"\b\w+>/dev/null\b", " ", cmd)
        normalized = re.sub(r"\s*&&\s*", " ; ", normalized)
        normalized = re.sub(r"\s*\|\|\s*", " ; ", normalized)
        normalized = re.sub(r"\s*\|\s*", " ; ", normalized)
        normalized = normalized.replace("(", " ").replace(")", " ")
        normalized = normalized.replace("{", " ").replace("}", " ")

        segments = [segment.strip() for segment in normalized.split(";") if segment.strip()]
        if not segments:
            return False

        for segment in segments:
            segment = re.sub(r"^cd\s+[^;]+", "", segment).strip()
            if not segment:
                continue
            token = segment.split()[0]
            if token not in _READ_ONLY_DIAGNOSTIC_TOKENS:
                return False
        return True

    def assess_shell_command(self, command: str) -> tuple[RiskLevel, str | None]:
        """Assess a shell command's risk. Returns (risk_level, block_reason_or_none)."""
        cmd_lower = command.lower()

        # Safe diagnostic / inspection commands in read-only mode should stay cheap.
        if self._is_read_only_diagnostic_command(command):
            for path in _ALWAYS_PROTECTED_PATTERNS:
                if path in cmd_lower:
                    return RiskLevel.CRITICAL, f"Access to protected system path: {path}"
            return RiskLevel.LOW, None

        for pattern in _SHELL_BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE):
                return RiskLevel.CRITICAL, f"Blocked pattern detected: {pattern}"

        # Check for protected system paths
        for path in _ALWAYS_PROTECTED_PATTERNS:
            if path in cmd_lower:
                return RiskLevel.CRITICAL, f"Access to protected system path: {path}"

        # Whitelisted sudo commands — regex-anchored to prevent prefix/suffix bypass
        if any(p.search(cmd_lower.strip()) for p in _SUDO_REGEX_WHITELIST):
            return RiskLevel.MEDIUM, None

        # Self-management (agent restarting itself, installing in its own venv)
        if "systemctl restart agent" in cmd_lower:
            return RiskLevel.LOW, None
        if ".venv/bin/pip install" in cmd_lower:
            return RiskLevel.LOW, None
        if "git pull" in cmd_lower and "/srv/agent" in cmd_lower:
            return RiskLevel.LOW, None

        # Destructive commands
        if any(kw in cmd_lower for kw in ["rm ", "kill -9", "pkill", "systemctl stop"]):
            return RiskLevel.HIGH, None

        if any(kw in cmd_lower for kw in [
            "apt install", "apt-get install", "apt remove", "pip install", "npm install",
            "systemctl restart", "docker rm", "docker stop",
        ]):
            return RiskLevel.MEDIUM, None

        return RiskLevel.LOW, None

    def assess_file_access(self, path: str, action: str) -> tuple[RiskLevel, str | None]:
        """Assess risk of a file system operation."""
        # Check protected system paths
        for protected in _ALWAYS_PROTECTED_PATTERNS:
            if protected in path:
                return RiskLevel.CRITICAL, f"Access to protected path: {protected}"

        # Check if it's a protected agent file
        if any(path.endswith(p) or p in path for p in config.protected_paths):
            if action in ("write", "delete", "move"):
                return RiskLevel.HIGH, "Modification of protected agent file"
            if action == "read":
                return RiskLevel.CRITICAL, "Read access to protected file is denied"

        return RiskLevel.LOW, None

    def validate_generated_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate code generated by ToolFactory. Returns (is_safe, list_of_issues)."""
        issues = []

        for pattern in _FORBIDDEN_IMPORTS:
            if pattern in code:
                issues.append(f"Forbidden pattern: {pattern}")

        # Check for network access that isn't declared
        if "socket.socket" in code:
            issues.append("Raw socket access detected — use httpx instead")

        # Check for file access outside workspace
        if re.search(r"open\(['\"]/(etc|root|boot|var/log)", code):
            issues.append("Attempted access to system directories")

        # Check for infinite loops (basic heuristic)
        if "while True" in code and "break" not in code and "await" not in code:
            issues.append("Potential infinite loop without break/await")

        return len(issues) == 0, issues

    def needs_approval(self, risk_level: RiskLevel) -> bool:
        """Whether this risk level requires user approval via Telegram."""
        return risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    async def evaluate_action(
        self,
        tool_name: str,
        action: str,
        parameters: dict[str, Any],
        risk_override: RiskLevel | None = None,
    ) -> tuple[ActionVerdict, RiskLevel, str | None]:
        """
        Evaluate an action before execution.
        Returns (verdict, risk_level, block_reason).
        """
        reason = None

        # Tool-specific risk assessment
        if tool_name == "shell":
            risk, reason = self.assess_shell_command(parameters.get("command", ""))
        elif tool_name == "filesystem":
            risk, reason = self.assess_file_access(
                parameters.get("path", ""),
                parameters.get("action", "read"),
            )
        else:
            risk = risk_override or RiskLevel.LOW

        # Critical with a reason = block immediately
        if risk == RiskLevel.CRITICAL and reason:
            log.warning(f"BLOCKED: {tool_name} — {reason}")
            return ActionVerdict.BLOCKED, risk, reason

        # High/Critical without block reason = needs approval
        if self.needs_approval(risk):
            return ActionVerdict.PENDING, risk, reason

        return ActionVerdict.AUTO_APPROVED, risk, None


# Singleton
security_agent = SecurityAgent()
