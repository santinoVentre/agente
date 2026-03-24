"""Scheduler — asyncio-based cron-like job scheduler for periodic tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from sqlalchemy import select, update

from db.connection import async_session
from db.models import ScheduledJob
from utils.logging import setup_logging

log = setup_logging("scheduler")


class Scheduler:
    """Lightweight asyncio scheduler that runs jobs based on interval expressions."""

    def __init__(self):
        self._handlers: dict[str, Callable[[], Awaitable[Any]]] = {}
        self._task: asyncio.Task | None = None

    def register_handler(self, name: str, handler: Callable[[], Awaitable[Any]]):
        """Register a named handler function that can be referenced by scheduled jobs."""
        self._handlers[name] = handler
        log.debug(f"Registered scheduler handler: {name}")

    async def ensure_job(
        self,
        name: str,
        handler_name: str,
        interval_seconds: int,
        description: str = "",
        enabled: bool = True,
    ):
        """Create a job in the DB if it doesn't exist yet."""
        async with async_session() as session:
            existing = await session.execute(
                select(ScheduledJob).where(ScheduledJob.name == name)
            )
            if existing.scalar_one_or_none():
                return
            job = ScheduledJob(
                name=name,
                handler_name=handler_name,
                interval_seconds=interval_seconds,
                description=description,
                enabled=enabled,
            )
            session.add(job)
            await session.commit()
            log.info(f"Scheduled job created: {name} (every {interval_seconds}s)")

    async def _run_due_jobs(self):
        """Check DB for due jobs and execute them."""
        now = datetime.now(timezone.utc)
        async with async_session() as session:
            result = await session.execute(
                select(ScheduledJob).where(ScheduledJob.enabled.is_(True))
            )
            jobs = list(result.scalars().all())

        for job in jobs:
            if job.next_run_at and job.next_run_at > now:
                continue

            handler = self._handlers.get(job.handler_name)
            if not handler:
                log.warning(f"No handler for job '{job.name}' (handler: {job.handler_name})")
                continue

            log.info(f"Running scheduled job: {job.name}")
            try:
                result_text = await handler()
                async with async_session() as session:
                    from datetime import timedelta
                    next_run = now + timedelta(seconds=job.interval_seconds)
                    await session.execute(
                        update(ScheduledJob)
                        .where(ScheduledJob.id == job.id)
                        .values(
                            last_run_at=now,
                            next_run_at=next_run,
                            last_result=str(result_text)[:500] if result_text else "ok",
                            run_count=ScheduledJob.run_count + 1,
                        )
                    )
                    await session.commit()
                log.info(f"Job '{job.name}' completed")
            except Exception as e:
                log.error(f"Job '{job.name}' failed: {e}", exc_info=True)
                async with async_session() as session:
                    from datetime import timedelta
                    next_run = now + timedelta(seconds=job.interval_seconds)
                    await session.execute(
                        update(ScheduledJob)
                        .where(ScheduledJob.id == job.id)
                        .values(
                            last_run_at=now,
                            next_run_at=next_run,
                            last_result=f"ERROR: {str(e)[:400]}",
                            run_count=ScheduledJob.run_count + 1,
                        )
                    )
                    await session.commit()

    async def _loop(self):
        """Main loop — check every 30 seconds for due jobs."""
        log.info("Scheduler loop started")
        while True:
            try:
                await self._run_due_jobs()
            except Exception as e:
                log.error(f"Scheduler loop error: {e}", exc_info=True)
            await asyncio.sleep(30)

    def start(self):
        """Start the scheduler background loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            log.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self._task and not self._task.done():
            self._task.cancel()
            log.info("Scheduler stopped")

    async def list_jobs(self) -> list[dict]:
        """Return all jobs for display."""
        async with async_session() as session:
            result = await session.execute(
                select(ScheduledJob).order_by(ScheduledJob.name)
            )
            jobs = result.scalars().all()
            return [
                {
                    "name": j.name,
                    "description": j.description,
                    "enabled": j.enabled,
                    "interval": j.interval_seconds,
                    "last_run": j.last_run_at.strftime("%H:%M:%S") if j.last_run_at else "mai",
                    "next_run": j.next_run_at.strftime("%H:%M:%S") if j.next_run_at else "subito",
                    "runs": j.run_count,
                    "last_result": (j.last_result or "")[:80],
                }
                for j in jobs
            ]

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a job. Returns True if found."""
        async with async_session() as session:
            result = await session.execute(
                select(ScheduledJob).where(ScheduledJob.name == name)
            )
            job = result.scalar_one_or_none()
            if not job:
                return False
            await session.execute(
                update(ScheduledJob)
                .where(ScheduledJob.id == job.id)
                .values(enabled=enabled)
            )
            await session.commit()
            log.info(f"Job '{name}' {'enabled' if enabled else 'disabled'}")
            return True


# Singleton
scheduler = Scheduler()
