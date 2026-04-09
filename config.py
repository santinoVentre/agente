"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_list(key: str) -> list[str]:
    raw = os.getenv(key, "")
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Config:
    # --- Core ---
    env: str = field(default_factory=lambda: _env("AGENT_ENV", "prod"))
    workspaces_dir: Path = field(default_factory=lambda: Path(_env("WORKSPACES_DIR", "/srv/agent/workspaces")))
    logs_dir: Path = field(default_factory=lambda: Path(_env("LOGS_DIR", "/srv/agent/logs")))
    media_dir: Path = field(default_factory=lambda: Path(_env("MEDIA_DIR", "/srv/agent/media")))
    tools_custom_dir: Path = field(default_factory=lambda: Path(_env("TOOLS_CUSTOM_DIR", "/srv/agent/app/tools/custom")))
    tool_backups_dir: Path = field(default_factory=lambda: Path(_env("TOOL_BACKUPS_DIR", "/srv/agent/app/tool_backups")))

    # --- Telegram ---
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    allowed_telegram_user_id: int = field(default_factory=lambda: _env_int("ALLOWED_TELEGRAM_USER_ID"))

    # --- OpenRouter ---
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))

    # --- GitHub ---
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    github_owner: str = field(default_factory=lambda: _env("GITHUB_OWNER", "santinosagent"))

    # --- Vercel ---
    vercel_token: str = field(default_factory=lambda: _env("VERCEL_TOKEN"))
    vercel_team_id: str = field(default_factory=lambda: _env("VERCEL_TEAM_ID"))

    # --- Database ---
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))

    # --- Redis ---
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))

    # --- Model routing ---
    # CHEAP  — execution, parsing, tool calls, repetitive tasks
    model_cheap: str = field(default_factory=lambda: _env("MODEL_ROUTING_CHEAP", "google/gemini-2.0-flash-001"))
    # MID    — reasoning, planning, validation, code generation (PRIMARY for websites)
    model_mid: str = field(default_factory=lambda: _env("MODEL_ROUTING_MID", "anthropic/claude-sonnet-4-5"))
    # EXPENSIVE — last resort: task fails multiple times / extreme complexity
    model_expensive: str = field(default_factory=lambda: _env("MODEL_ROUTING_EXPENSIVE", "anthropic/claude-opus-4-5"))
    # Legacy / specialised overrides (env-configurable)
    model_autonomous: str = field(default_factory=lambda: _env("MODEL_AUTONOMOUS", "anthropic/claude-sonnet-4-5"))
    model_code_generation: str = field(default_factory=lambda: _env("MODEL_CODE_GENERATION", "anthropic/claude-sonnet-4-5"))
    model_tool_execution: str = field(default_factory=lambda: _env("MODEL_TOOL_EXECUTION", "google/gemini-2.0-flash-001"))

    # --- Execution controller ---
    max_steps_per_task: int = field(default_factory=lambda: _env_int("MAX_STEPS_PER_TASK", 8))
    max_tokens_per_task: int = field(default_factory=lambda: _env_int("MAX_TOKENS_PER_TASK", 100_000))
    context_summary_interval: int = field(default_factory=lambda: _env_int("CONTEXT_SUMMARY_INTERVAL", 3))

    # --- Security ---
    max_shell_timeout: int = field(default_factory=lambda: _env_int("MAX_SHELL_TIMEOUT", 300))
    protected_paths: list[str] = field(default_factory=lambda: _env_list("PROTECTED_PATHS"))

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


# Singleton
config = Config()
