"""Vercel tool — deploy projects and manage deployments via Vercel API."""

from __future__ import annotations

import asyncio
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

    async def _get_project(self, client: httpx.AsyncClient, project_name: str) -> httpx.Response:
        return await client.get(f"{VERCEL_API}/v9/projects/{project_name}", params=self._params())

    async def _latest_deployment(self, client: httpx.AsyncClient, project_id: str | None) -> dict[str, Any] | None:
        if not project_id:
            return None
        resp = await client.get(
            f"{VERCEL_API}/v6/deployments",
            params={**self._params(), "projectId": project_id, "limit": 1},
        )
        resp.raise_for_status()
        deployments = resp.json().get("deployments", [])
        if not deployments:
            return None
        deployment = deployments[0]
        return {
            "id": deployment.get("uid"),
            "url": deployment.get("url"),
            "state": deployment.get("state"),
            "created": deployment.get("created"),
        }

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["validate_auth", "deploy_from_github", "list_projects", "list_deployments", "get_project", "delete_project"],
                    "description": "Vercel action to perform.",
                },
                "owner": {"type": "string", "description": "GitHub owner/org. Defaults to configured owner."},
                "project_name": {"type": "string", "description": "Vercel project name."},
                "repo_name": {"type": "string", "description": "GitHub repo name to deploy from."},
                "framework": {
                    "type": "string",
                    "description": "Framework preset (nextjs, vite, etc.).",
                    "default": "nextjs",
                },
                "wait_for_deployment": {
                    "type": "boolean",
                    "description": "Whether to briefly wait for a deployment to appear after linking the project.",
                    "default": True,
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action")
        if not action:
            return {"success": False, "error": "Missing required parameter(s): action.", "failure_kind": "invalid_args"}
        async with httpx.AsyncClient(headers=self._headers(), timeout=60.0) as client:
            try:
                if action == "validate_auth":
                    resp = await client.get(f"{VERCEL_API}/v2/user", params=self._params())
                    resp.raise_for_status()
                    data = resp.json().get("user", {})
                    return {
                        "success": True,
                        "user_id": data.get("id"),
                        "username": data.get("username"),
                        "email": data.get("email"),
                        "team_id": config.vercel_team_id or None,
                    }

                if action == "deploy_from_github":
                    owner = kwargs.get("owner") or config.github_owner
                    repo = kwargs["repo_name"]
                    project_name = kwargs.get("project_name", repo)
                    project_resp = await self._get_project(client, project_name)

                    if project_resp.status_code == 200:
                        data = project_resp.json()
                        created = False
                    else:
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

                        if resp.status_code == 409:
                            project_resp = await self._get_project(client, project_name)
                            project_resp.raise_for_status()
                            data = project_resp.json()
                            created = False
                        else:
                            resp.raise_for_status()
                            data = resp.json()
                            created = True

                    latest = None
                    if kwargs.get("wait_for_deployment", True):
                        for _ in range(5):
                            latest = await self._latest_deployment(client, data.get("id"))
                            if latest:
                                break
                            await asyncio.sleep(2)

                    if not latest:
                        return {
                            "success": False,
                            "project_id": data.get("id"),
                            "project_name": project_name,
                            "project_url": f"https://{project_name}.vercel.app",
                            "created": created,
                            "failure_kind": "remote_api_error",
                            "error": (
                                "Project linked on Vercel but no deployment was detected yet. "
                                "Verify that the GitHub repository is accessible to the Vercel account/team."
                            ),
                        }

                    return {
                        "success": True,
                        "project_id": data.get("id"),
                        "project_name": project_name,
                        "created": created,
                        "url": f"https://{latest['url']}",
                        "deployment_state": latest.get("state"),
                        "deployment_id": latest.get("id"),
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
                    return {"success": False, "error": f"Unknown action: {action}", "failure_kind": "invalid_args"}

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (401, 403):
                    failure_kind = "auth_required"
                elif status == 402:
                    failure_kind = "billing_required"
                elif status == 404:
                    failure_kind = "not_found"
                else:
                    failure_kind = "remote_api_error"
                return {
                    "success": False,
                    "error": f"Vercel API error: {status} {e.response.text[:500]}",
                    "failure_kind": failure_kind,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "failure_kind": "runtime_error"}
