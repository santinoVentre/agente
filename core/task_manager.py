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

    async def cleanup_stale_tasks(self) -> int:
        """
        Called at startup: mark all IN_PROGRESS / PENDING / WAITING_APPROVAL
        tasks as CANCELLED. These are orphans from a previous agent run.
        Returns the number of tasks cleaned up.
        """
        stale_statuses = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.WAITING_APPROVAL]
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status.in_(stale_statuses))
            )
            stale = list(result.scalars().all())
            for task in stale:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now(timezone.utc)
                task.error = "Annullato automaticamente al riavvio dell'agente"
            await session.commit()

        # Also cancel any asyncio tasks still tracked (shouldn't be any at startup)
        for atask in list(self._running_tasks.values()):
            if not atask.done():
                atask.cancel()
        self._running_tasks.clear()

        if stale:
            log.info(f"Startup cleanup: {len(stale)} task orfani cancellati: {[t.id for t in stale]}")
        return len(stale)

    async def cancel_all_active(self) -> int:
        """Cancel all currently active tasks (manual operator command)."""
        tasks = await self.get_active_tasks()
        for t in tasks:
            atask = self._running_tasks.pop(t.id, None)
            if atask and not atask.done():
                atask.cancel()
            await self.update_task_status(t.id, TaskStatus.CANCELLED)
        return len(tasks)

    async def close(self):
        if self._redis:
            await self._redis.close()

    async def get_stuck_waiting_tasks(self, older_than_minutes: int = 30) -> list[Task]:
        """Find tasks stuck in WAITING_APPROVAL for longer than the threshold."""
        threshold = datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=older_than_minutes)
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(
                    Task.status == TaskStatus.WAITING_APPROVAL,
                    Task.created_at < threshold,
                ).order_by(Task.created_at.asc())
            )
            return list(result.scalars().all())


# Singleton
task_manager = TaskManager()


async def task_recovery_job():
    """Scheduler job: notify owner about tasks stuck in WAITING_APPROVAL."""
    from tg.notifications import notify

    stuck = await task_manager.get_stuck_waiting_tasks(older_than_minutes=30)
    if not stuck:
        return

    lines = [f"⚠️ <b>{len(stuck)} task in attesa di approvazione da oltre 30 min:</b>\n"]
    for t in stuck[:10]:
        age_min = int((datetime.now(timezone.utc) - t.created_at).total_seconds() / 60)
        lines.append(f"  • #{t.id} — {t.description[:60]} ({age_min} min fa)")
    lines.append("\nRispondi /tasks per gestirli, oppure verranno annullati fra 2h.")

    await notify("\n".join(lines))

    # Auto-cancel tasks stuck for > 2 hours
    for t in stuck:
        age_hours = (datetime.now(timezone.utc) - t.created_at).total_seconds() / 3600
        if age_hours > 2:
            await task_manager.update_task_status(t.id, TaskStatus.CANCELLED, error="Auto-cancellato: nessuna approvazione ricevuta entro 2h")
            log.info(f"Task #{t.id} auto-cancelled after {age_hours:.1f}h waiting")
