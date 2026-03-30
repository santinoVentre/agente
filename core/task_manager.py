"""Task manager — creates, tracks, and updates tasks using DB + Redis."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import select, update

from config import config
from db.connection import async_session
from db.models import ActionLog, ActionVerdict, RiskLevel, Task, TaskStatus
from utils.logging import setup_logging

log = setup_logging("task_manager")


class TaskManager:
    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._running_tasks: dict[int, asyncio.Task] = {}

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(config.redis_url, decode_responses=True)
        return self._redis

    # ── Task CRUD ─────────────────────────────────────────────────────

    async def create_task(
        self,
        user_id: int,
        description: str,
        task_type: str,
        agent: str | None = None,
        model: str | None = None,
    ) -> Task:
        async with async_session() as session:
            task = Task(
                user_id=user_id,
                description=description,
                task_type=task_type,
                agent_assigned=agent,
                model_used=model,
                status=TaskStatus.PENDING,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            log.info(f"Task #{task.id} created: {task_type} — {description[:80]}")

            # Publish to Redis for real-time status
            r = await self._get_redis()
            await r.hset(f"task:{task.id}", mapping={
                "status": task.status.value,
                "type": task_type,
                "description": description[:200],
                "progress": "0",
                "agent": agent or "",
            })
            return task

    async def update_task_status(
        self, task_id: int, status: TaskStatus, progress: int | None = None, error: str | None = None
    ):
        async with async_session() as session:
            values: dict[str, Any] = {"status": status}
            if progress is not None:
                values["progress"] = progress
            if error:
                values["error"] = error
            if status == TaskStatus.IN_PROGRESS:
                values["started_at"] = datetime.now(timezone.utc)
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                values["completed_at"] = datetime.now(timezone.utc)
            await session.execute(update(Task).where(Task.id == task_id).values(**values))
            await session.commit()

        r = await self._get_redis()
        update_map = {"status": status.value}
        if progress is not None:
            update_map["progress"] = str(progress)
        await r.hset(f"task:{task_id}", mapping=update_map)

    async def set_waiting_approval(
        self,
        task_id: int,
        reason: str | None = None,
        progress: int | None = None,
    ):
        """Mark task as waiting for owner approval and expose reason in Redis."""
        await self.update_task_status(task_id, TaskStatus.WAITING_APPROVAL, progress=progress)
        if reason:
            r = await self._get_redis()
            await r.hset(f"task:{task_id}", mapping={"waiting_reason": reason[:500]})

    async def complete_task(self, task_id: int, result: dict | None = None, cost: float = 0.0):
        async with async_session() as session:
            await session.execute(
                update(Task).where(Task.id == task_id).values(
                    status=TaskStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                    result=result,
                    progress=100,
                    cost_usd=cost,
                )
            )
            await session.commit()

        r = await self._get_redis()
        await r.hset(f"task:{task_id}", mapping={"status": "completed", "progress": "100"})
        await r.expire(f"task:{task_id}", 3600)  # keep for 1h then cleanup

    async def fail_task(self, task_id: int, error: str):
        await self.update_task_status(task_id, TaskStatus.FAILED, error=error)

    async def cancel_task(self, task_id: int) -> str:
        # Cancel the asyncio task if running
        atask = self._running_tasks.pop(task_id, None)
        if atask and not atask.done():
            atask.cancel()
        await self.update_task_status(task_id, TaskStatus.CANCELLED)
        return f"Task #{task_id} annullato."

    # ── Queries ───────────────────────────────────────────────────────

    async def get_active_tasks(self) -> list[Task]:
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.WAITING_APPROVAL])
                ).order_by(Task.created_at.desc())
            )
            return list(result.scalars().all())

    async def get_task_status_from_redis(self, task_id: int) -> dict | None:
        r = await self._get_redis()
        data = await r.hgetall(f"task:{task_id}")
        return data if data else None

    # ── Action logging ────────────────────────────────────────────────

    async def log_action(
        self,
        task_id: int | None,
        agent: str,
        tool_name: str,
        action: str,
        parameters: dict | None,
        risk_level: RiskLevel,
        verdict: ActionVerdict,
        result: dict | None = None,
        error: str | None = None,
    ):
        async with async_session() as session:
            entry = ActionLog(
                task_id=task_id,
                agent=agent,
                tool_name=tool_name,
                action=action,
                parameters=parameters,
                risk_level=risk_level,
                verdict=verdict,
                result=result,
                error=error,
            )
            session.add(entry)
            await session.commit()

    # ── Async task tracking ───────────────────────────────────────────

    def register_running_task(self, task_id: int, atask: asyncio.Task):
        self._running_tasks[task_id] = atask

    def unregister_running_task(self, task_id: int):
        self._running_tasks.pop(task_id, None)

    async def close(self):
        if self._redis:
            await self._redis.close()


# Singleton
task_manager = TaskManager()
