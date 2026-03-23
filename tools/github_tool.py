"""GitHub tool — create repos, push code, manage files via GitHub API."""

from __future__ import annotations

from typing import Any

import httpx

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_github")

GITHUB_API = "https://api.github.com"


class GitHubTool(BaseTool):
    name = "github"
    description = "Create repositories, push files, and manage GitHub resources."
    risk_level = RiskLevel.MEDIUM

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_repo", "push_file", "list_repos", "delete_repo", "get_repo"],
                    "description": "GitHub action to perform.",
                },
                "repo_name": {"type": "string", "description": "Repository name."},
                "description": {"type": "string", "description": "Repo description."},
                "private": {"type": "boolean", "description": "Whether the repo is private.", "default": True},
                "file_path": {"type": "string", "description": "Path within the repo (for push_file)."},
                "content": {"type": "string", "description": "File content (for push_file)."},
                "commit_message": {"type": "string", "description": "Commit message.", "default": "Update via agent"},
                "branch": {"type": "string", "description": "Branch name.", "default": "main"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs["action"]
        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            try:
                if action == "create_repo":
                    resp = await client.post(
                        f"{GITHUB_API}/user/repos",
                        json={
                            "name": kwargs["repo_name"],
                            "description": kwargs.get("description", ""),
                            "private": kwargs.get("private", True),
                            "auto_init": True,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return {"success": True, "url": data["html_url"], "clone_url": data["clone_url"]}

                elif action == "push_file":
                    import base64
                    owner = config.github_owner
                    repo = kwargs["repo_name"]
                    path = kwargs["file_path"]
                    content_b64 = base64.b64encode(kwargs["content"].encode()).decode()

                    # Check if file exists (to get SHA for updates)
                    sha = None
                    check = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")
                    if check.status_code == 200:
                        sha = check.json().get("sha")

                    payload = {
                        "message": kwargs.get("commit_message", "Update via agent"),
                        "content": content_b64,
                        "branch": kwargs.get("branch", "main"),
                    }
                    if sha:
                        payload["sha"] = sha

                    resp = await client.put(
                        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
                        json=payload,
                    )
                    resp.raise_for_status()
                    return {"success": True, "message": f"Pushed {path} to {owner}/{repo}"}

                elif action == "list_repos":
                    resp = await client.get(f"{GITHUB_API}/user/repos", params={"per_page": 50, "sort": "updated"})
                    resp.raise_for_status()
                    repos = [{"name": r["name"], "url": r["html_url"], "private": r["private"]} for r in resp.json()]
                    return {"success": True, "repos": repos}

                elif action == "get_repo":
                    owner = config.github_owner
                    resp = await client.get(f"{GITHUB_API}/repos/{owner}/{kwargs['repo_name']}")
                    resp.raise_for_status()
                    return {"success": True, "repo": resp.json()}

                elif action == "delete_repo":
                    owner = config.github_owner
                    resp = await client.delete(f"{GITHUB_API}/repos/{owner}/{kwargs['repo_name']}")
                    resp.raise_for_status()
                    return {"success": True, "message": f"Deleted {owner}/{kwargs['repo_name']}"}

                else:
                    return {"success": False, "error": f"Unknown action: {action}"}

            except httpx.HTTPStatusError as e:
                return {"success": False, "error": f"GitHub API error: {e.response.status_code} {e.response.text[:500]}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
