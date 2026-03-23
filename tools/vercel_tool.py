"""Vercel tool — deploy projects and manage deployments via Vercel API."""

from __future__ import annotations

from typing import Any

import httpx

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_vercel")

VERCEL_API = "https://api.vercel.com"


class VercelTool(BaseTool):
    name = "vercel"
    description = "Deploy projects to Vercel, list deployments, manage projects."
    risk_level = RiskLevel.MEDIUM

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {config.vercel_token}", "Content-Type": "application/json"}
        return h

    def _params(self) -> dict[str, str]:
        params = {}
        if config.vercel_team_id:
            params["teamId"] = config.vercel_team_id
        return params

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["deploy_from_github", "list_projects", "list_deployments", "get_project", "delete_project"],
                    "description": "Vercel action to perform.",
                },
                "project_name": {"type": "string", "description": "Vercel project name."},
                "repo_name": {"type": "string", "description": "GitHub repo name to deploy from."},
                "framework": {
                    "type": "string",
                    "description": "Framework preset (nextjs, vite, etc.).",
                    "default": "nextjs",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs["action"]
        async with httpx.AsyncClient(headers=self._headers(), timeout=60.0) as client:
            try:
                if action == "deploy_from_github":
                    owner = config.github_owner
                    repo = kwargs["repo_name"]
                    project_name = kwargs.get("project_name", repo)

                    resp = await client.post(
                        f"{VERCEL_API}/v10/projects",
                        params=self._params(),
                        json={
                            "name": project_name,
                            "framework": kwargs.get("framework", "nextjs"),
                            "gitRepository": {
                                "type": "github",
                                "repo": f"{owner}/{repo}",
                            },
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return {
                        "success": True,
                        "project_id": data.get("id"),
                        "url": f"https://{project_name}.vercel.app",
                    }

                elif action == "list_projects":
                    resp = await client.get(f"{VERCEL_API}/v9/projects", params=self._params())
                    resp.raise_for_status()
                    projects = [
                        {"name": p["name"], "id": p["id"], "url": f"https://{p['name']}.vercel.app"}
                        for p in resp.json().get("projects", [])
                    ]
                    return {"success": True, "projects": projects}

                elif action == "list_deployments":
                    params = {**self._params()}
                    if kwargs.get("project_name"):
                        project_resp = await client.get(
                            f"{VERCEL_API}/v9/projects/{kwargs['project_name']}", params=self._params()
                        )
                        if project_resp.status_code == 200:
                            params["projectId"] = project_resp.json()["id"]

                    resp = await client.get(f"{VERCEL_API}/v6/deployments", params=params)
                    resp.raise_for_status()
                    deps = [
                        {"url": d.get("url"), "state": d.get("state"), "created": d.get("created")}
                        for d in resp.json().get("deployments", [])[:10]
                    ]
                    return {"success": True, "deployments": deps}

                elif action == "get_project":
                    resp = await client.get(
                        f"{VERCEL_API}/v9/projects/{kwargs['project_name']}", params=self._params()
                    )
                    resp.raise_for_status()
                    return {"success": True, "project": resp.json()}

                elif action == "delete_project":
                    resp = await client.delete(
                        f"{VERCEL_API}/v9/projects/{kwargs['project_name']}", params=self._params()
                    )
                    resp.raise_for_status()
                    return {"success": True, "message": f"Deleted project {kwargs['project_name']}"}

                else:
                    return {"success": False, "error": f"Unknown action: {action}"}

            except httpx.HTTPStatusError as e:
                return {"success": False, "error": f"Vercel API error: {e.response.status_code} {e.response.text[:500]}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
