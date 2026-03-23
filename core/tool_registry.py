"""ToolRegistry — DB-backed registry of all tools with hot-reload capability."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from config import config
from db.connection import async_session
from db.models import RiskLevel, ToolRegistry as ToolRegistryModel, ToolSource, ToolStatus
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_registry")


class ToolRegistryManager:
    """Manages discovery, registration, and hot-loading of tools."""

    def __init__(self):
        self._loaded_tools: dict[str, BaseTool] = {}

    async def register_builtin(self, tool: BaseTool):
        """Register a built-in tool in the DB if not already present."""
        async with async_session() as session:
            existing = await session.execute(
                select(ToolRegistryModel).where(ToolRegistryModel.name == tool.name)
            )
            if existing.scalar_one_or_none():
                log.debug(f"Tool '{tool.name}' already registered")
            else:
                entry = ToolRegistryModel(
                    name=tool.name,
                    description=tool.description,
                    schema_json=tool.to_openai_schema()["function"],
                    source=ToolSource.BUILTIN,
                    source_path=None,
                    risk_level=tool.risk_level,
                    status=ToolStatus.ACTIVE,
                    is_protected=True,
                )
                session.add(entry)
                await session.commit()
                log.info(f"Registered builtin tool: {tool.name}")
        self._loaded_tools[tool.name] = tool

    async def register_generated(
        self,
        name: str,
        description: str,
        schema: dict,
        source_path: str,
        risk_level: RiskLevel,
        created_by_model: str,
        dependencies: list[str] | None = None,
    ) -> int:
        """Register a newly generated tool in the DB."""
        async with async_session() as session:
            existing = await session.execute(
                select(ToolRegistryModel).where(ToolRegistryModel.name == name)
            )
            row = existing.scalar_one_or_none()
            if row:
                # Update existing (new version)
                await session.execute(
                    update(ToolRegistryModel)
                    .where(ToolRegistryModel.id == row.id)
                    .values(
                        description=description,
                        schema_json=schema,
                        source_path=source_path,
                        risk_level=risk_level,
                        created_by_model=created_by_model,
                        dependencies=dependencies,
                        version=row.version + 1,
                        status=ToolStatus.ACTIVE,
                    )
                )
                await session.commit()
                log.info(f"Updated generated tool: {name} → v{row.version + 1}")
                return row.id
            else:
                entry = ToolRegistryModel(
                    name=name,
                    description=description,
                    schema_json=schema,
                    source=ToolSource.GENERATED,
                    source_path=source_path,
                    risk_level=risk_level,
                    status=ToolStatus.ACTIVE,
                    created_by_model=created_by_model,
                    dependencies=dependencies,
                )
                session.add(entry)
                await session.commit()
                await session.refresh(entry)
                log.info(f"Registered generated tool: {name}")
                return entry.id

    async def load_custom_tool(self, name: str) -> BaseTool | None:
        """Dynamically load a custom tool from its source file."""
        async with async_session() as session:
            result = await session.execute(
                select(ToolRegistryModel).where(
                    ToolRegistryModel.name == name,
                    ToolRegistryModel.status == ToolStatus.ACTIVE,
                )
            )
            entry = result.scalar_one_or_none()
            if not entry or not entry.source_path:
                return None

        path = Path(entry.source_path)
        if not path.exists():
            log.error(f"Custom tool source not found: {path}")
            return None

        try:
            module_name = f"tools.custom.{name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find the tool class (subclass of BaseTool)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    tool = attr()
                    self._loaded_tools[name] = tool
                    log.info(f"Hot-loaded custom tool: {name}")
                    return tool

            log.error(f"No BaseTool subclass found in {path}")
            return None
        except Exception as e:
            log.error(f"Failed to load custom tool '{name}': {e}")
            return None

    async def get_all_active_schemas(self) -> list[dict]:
        """Get OpenAI function schemas for all active tools."""
        async with async_session() as session:
            result = await session.execute(
                select(ToolRegistryModel).where(ToolRegistryModel.status == ToolStatus.ACTIVE)
            )
            entries = result.scalars().all()
            schemas = []
            for entry in entries:
                schemas.append({
                    "type": "function",
                    "function": entry.schema_json,
                })
            return schemas

    async def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name, loading it dynamically if needed."""
        if name in self._loaded_tools:
            return self._loaded_tools[name]
        return await self.load_custom_tool(name)

    async def record_invocation(self, name: str):
        """Increment invocation counter."""
        from datetime import datetime, timezone
        async with async_session() as session:
            await session.execute(
                update(ToolRegistryModel)
                .where(ToolRegistryModel.name == name)
                .values(
                    invocation_count=ToolRegistryModel.invocation_count + 1,
                    last_used_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    async def list_tools_summary(self) -> list[dict]:
        async with async_session() as session:
            result = await session.execute(
                select(ToolRegistryModel).order_by(ToolRegistryModel.name)
            )
            return [
                {
                    "name": t.name,
                    "description": t.description[:80],
                    "source": t.source.value,
                    "risk": t.risk_level.value,
                    "version": t.version,
                    "uses": t.invocation_count,
                    "status": t.status.value,
                }
                for t in result.scalars().all()
            ]


# Singleton
tool_registry = ToolRegistryManager()
