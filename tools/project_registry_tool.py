"""Project Registry Tool - lets agents track and query project deployment metadata."""

from __future__ import annotations

from typing import Any

from db.models import RiskLevel
from tools.base_tool import BaseTool


class ProjectRegistryTool(BaseTool):
    name = "project_registry"
    description = (
        "Gestisce il registro progetti persistente: crea/aggiorna progetto, "
        "legge dettagli, elenca progetti recenti."
    )
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["upsert", "get", "list"],
                    "description": "Azione da eseguire sul registro progetti",
                },
                "name": {
                    "type": "string",
                    "description": "Nome progetto (obbligatorio per upsert/get)",
                },
                "description": {"type": "string"},
                "workspace_path": {"type": "string"},
                "github_repo": {"type": "string"},
                "domain": {"type": "string"},
                "deploy_provider": {"type": "string"},
                "deploy_url": {"type": "string"},
                "status": {
                    "type": "string",
                    "description": "es: active, deployed, paused, failed",
                },
                "mark_deployed": {
                    "type": "boolean",
                    "description": "Se true, aggiorna last_deployed_at",
                },
                "metadata": {
                    "type": "object",
                    "description": "Metadati arbitrari in JSON",
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo risultati per list",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        from core.project_registry import project_registry

        action = kwargs.get("action")

        if action == "upsert":
            name = kwargs.get("name", "").strip()
            if not name:
                return {"success": False, "error": "name obbligatorio per upsert"}

            pid = await project_registry.upsert_project(
                name=name,
                description=kwargs.get("description"),
                workspace_path=kwargs.get("workspace_path"),
                github_repo=kwargs.get("github_repo"),
                domain=kwargs.get("domain"),
                deploy_provider=kwargs.get("deploy_provider"),
                deploy_url=kwargs.get("deploy_url"),
                status=kwargs.get("status", "active"),
                metadata_json=kwargs.get("metadata"),
                mark_deployed=bool(kwargs.get("mark_deployed", False)),
            )
            return {"success": True, "id": pid, "name": name}

        if action == "get":
            name = kwargs.get("name", "").strip()
            if not name:
                return {"success": False, "error": "name obbligatorio per get"}

            row = await project_registry.get_project(name)
            if not row:
                return {"success": False, "error": f"Progetto '{name}' non trovato"}

            return {
                "success": True,
                "project": {
                    "id": row.id,
                    "name": row.name,
                    "description": row.description,
                    "workspace_path": row.workspace_path,
                    "github_repo": row.github_repo,
                    "domain": row.domain,
                    "deploy_provider": row.deploy_provider,
                    "deploy_url": row.deploy_url,
                    "status": row.status,
                    "metadata": row.metadata_json,
                    "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
                    "last_deployed_at": row.last_deployed_at.isoformat() if row.last_deployed_at else None,
                },
            }

        if action == "list":
            limit = int(kwargs.get("limit", 20))
            rows = await project_registry.list_projects(limit=limit)
            return {
                "success": True,
                "projects": [
                    {
                        "name": r.name,
                        "status": r.status,
                        "github_repo": r.github_repo,
                        "domain": r.domain,
                        "deploy_url": r.deploy_url,
                        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    }
                    for r in rows
                ],
            }

        return {"success": False, "error": f"Azione non supportata: {action}"}
