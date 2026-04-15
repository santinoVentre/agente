"""Runtime PM chat session state.

Keeps a per-user active Project Manager conversation so every new message is
routed to the PM agent until the session is explicitly terminated.
Sessions are persisted to Redis so they survive process restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import redis.asyncio as aioredis

from config import config
from db.models import ProjectRegistry
from utils.logging import setup_logging

log = setup_logging("pm_session")

_REDIS_PREFIX = "pm_session:"
_SESSION_TTL = 86400  # 24 hours


@dataclass
class PMSessionState:
    user_id: int
    project: ProjectRegistry
    chat_id: int
    history: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text[:4000]})
        self.history = self.history[-12:]

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text[:4000]})
        self.history = self.history[-12:]

    def to_redis_dict(self) -> dict:
        """Serialize to a dict suitable for Redis storage."""
        return {
            "user_id": self.user_id,
            "project_name": self.project.name,
            "project_id": self.project.id,
            "chat_id": self.chat_id,
            "history": json.dumps(self.history, ensure_ascii=False),
        }


_active_pm_sessions: dict[int, PMSessionState] = {}


def get_pm_session(user_id: int) -> PMSessionState | None:
    return _active_pm_sessions.get(user_id)


def start_pm_session(user_id: int, project: ProjectRegistry, chat_id: int) -> PMSessionState:
    state = PMSessionState(user_id=user_id, project=project, chat_id=chat_id)
    _active_pm_sessions[user_id] = state
    # Fire-and-forget Redis persistence
    _schedule_redis_save(user_id, state)
    return state


def end_pm_session(user_id: int) -> None:
    _active_pm_sessions.pop(user_id, None)
    _schedule_redis_delete(user_id)


async def save_pm_session_to_redis(user_id: int, state: PMSessionState) -> None:
    """Persist PM session state to Redis."""
    try:
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        key = f"{_REDIS_PREFIX}{user_id}"
        await r.hset(key, mapping=state.to_redis_dict())
        await r.expire(key, _SESSION_TTL)
        await r.aclose()
    except Exception as e:
        log.warning(f"Failed to save PM session to Redis: {e}")


async def restore_pm_sessions_from_redis() -> int:
    """Restore PM sessions from Redis at startup. Returns count restored."""
    try:
        from core.project_registry import project_registry

        r = aioredis.from_url(config.redis_url, decode_responses=True)
        keys = []
        async for key in r.scan_iter(match=f"{_REDIS_PREFIX}*"):
            keys.append(key)

        restored = 0
        for key in keys:
            try:
                data = await r.hgetall(key)
                if not data:
                    continue
                user_id = int(data["user_id"])
                project_name = data["project_name"]
                chat_id = int(data["chat_id"])
                history = json.loads(data.get("history", "[]"))

                project = await project_registry.get_project(project_name)
                if project is None:
                    await r.delete(key)
                    continue

                state = PMSessionState(
                    user_id=user_id, project=project, chat_id=chat_id, history=history
                )
                _active_pm_sessions[user_id] = state
                restored += 1
            except Exception as e:
                log.warning(f"Failed to restore PM session from {key}: {e}")
                await r.delete(key)

        await r.aclose()
        if restored:
            log.info(f"Restored {restored} PM sessions from Redis")
        return restored
    except Exception as e:
        log.warning(f"Failed to restore PM sessions from Redis: {e}")
        return 0


def _schedule_redis_save(user_id: int, state: PMSessionState) -> None:
    """Schedule async Redis save without blocking."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(save_pm_session_to_redis(user_id, state))
    except RuntimeError:
        pass


def _schedule_redis_delete(user_id: int) -> None:
    """Schedule async Redis key deletion."""
    import asyncio

    async def _delete():
        try:
            r = aioredis.from_url(config.redis_url, decode_responses=True)
            await r.delete(f"{_REDIS_PREFIX}{user_id}")
            await r.aclose()
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_delete())
    except RuntimeError:
        pass
