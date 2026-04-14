"""Project registry service - tracks deployed projects and key metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from db.connection import async_session
from db.models import ProjectRegistry
from utils.logging import setup_logging

log = setup_logging("project_registry")


class ProjectRegistryService:
    _HIDDEN_STATUSES = {"deleted", "archived"}

    async def upsert_project(
        self,
        name: str,
        description: str | None = None,
        workspace_path: str | None = None,
        github_repo: str | None = None,
        domain: str | None = None,
        deploy_provider: str | None = None,
        deploy_url: str | None = None,
        status: str = "active",
        metadata_json: dict | None = None,
        mark_deployed: bool = False,
    ) -> int:
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRegistry).where(ProjectRegistry.name == name)
            )
            row = result.scalar_one_or_none()
            now = datetime.now(timezone.utc)

            if row:
                if description is not None:
                    row.description = description
                if workspace_path is not None:
                    row.workspace_path = workspace_path
                if github_repo is not None:
                    row.github_repo = github_repo
                if domain is not None:
                    row.domain = domain
                if deploy_provider is not None:
                    row.deploy_provider = deploy_provider
                if deploy_url is not None:
                    row.deploy_url = deploy_url
                row.status = status
                if metadata_json is not None:
                    row.metadata_json = metadata_json
                row.last_checked_at = now
                if mark_deployed:
                    row.last_deployed_at = now
                await session.commit()
                return row.id

            entry = ProjectRegistry(
                name=name,
                description=description,
                workspace_path=workspace_path,
                github_repo=github_repo,
                domain=domain,
                deploy_provider=deploy_provider,
                deploy_url=deploy_url,
                status=status,
                metadata_json=metadata_json,
                last_checked_at=now,
                last_deployed_at=now if mark_deployed else None,
            )
            session.add(entry)
            await session.commit()
            await session.refresh(entry)
            return entry.id

    async def list_projects(self, limit: int = 30) -> list[ProjectRegistry]:
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRegistry)
                .order_by(ProjectRegistry.updated_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_selectable_projects(self, limit: int = 30) -> list[ProjectRegistry]:
        """Projects that should appear in the website selector.

        A site may be deployato and still have a status different from `active`, so
        the selector must not hide valid projects only because of status drift.
        """
        rows = await self.list_projects(limit=limit)
        selectable: list[ProjectRegistry] = []
        for row in rows:
            status = (row.status or "").strip().lower()
            if status in self._HIDDEN_STATUSES:
                continue
            if row.workspace_path or row.github_repo or row.deploy_url or row.last_deployed_at:
                selectable.append(row)
                continue
            if status in ("active", "building", "deployed", "paused", "failed"):
                selectable.append(row)
        return selectable

    async def get_project(self, name: str) -> ProjectRegistry | None:
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRegistry).where(ProjectRegistry.name == name)
            )
            return result.scalar_one_or_none()

    async def get_recent_projects_summary(self, limit: int = 6) -> str:
        rows = await self.list_projects(limit=limit)
        if not rows:
            return "Nessun progetto registrato."

        lines = []
        for r in rows:
            lines.append(
                f"{r.name} [status={r.status}] repo={r.github_repo or '-'} "
                f"domain={r.domain or '-'} deploy={r.deploy_url or '-'}"
            )
        return " | ".join(lines)


project_registry = ProjectRegistryService()
