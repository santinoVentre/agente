"""GitHub tool — create repos, push code, manage files via GitHub API."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_github")

GITHUB_API = "https://api.github.com"
_SKIP_DIRS = {
    ".git", "node_modules", ".next", "dist", "build", ".vercel", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "venv", ".venv",
}
_SKIP_FILES = {".DS_Store"}


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

    def _owner(self, kwargs: dict[str, Any]) -> str:
        return kwargs.get("owner") or config.github_owner

    async def _get_repo(self, client: httpx.AsyncClient, owner: str, repo_name: str) -> httpx.Response:
        return await client.get(f"{GITHUB_API}/repos/{owner}/{repo_name}")

    async def _ensure_repo(
        self,
        client: httpx.AsyncClient,
        repo_name: str,
        description: str,
        private: bool,
        owner: str,
    ) -> dict[str, Any]:
        existing = await self._get_repo(client, owner, repo_name)
        if existing.status_code == 200:
            data = existing.json()
            return {
                "success": True,
                "existing": True,
                "url": data.get("html_url"),
                "clone_url": data.get("clone_url"),
                "default_branch": data.get("default_branch", "main"),
            }

        resp = await client.post(
            f"{GITHUB_API}/user/repos",
            json={
                "name": repo_name,
                "description": description,
                "private": private,
                "auto_init": True,
            },
        )

        if resp.status_code == 422:
            existing = await self._get_repo(client, owner, repo_name)
            existing.raise_for_status()
            data = existing.json()
            return {
                "success": True,
                "existing": True,
                "url": data.get("html_url"),
                "clone_url": data.get("clone_url"),
                "default_branch": data.get("default_branch", "main"),
            }

        resp.raise_for_status()
        data = resp.json()
        return {
            "success": True,
            "existing": False,
            "url": data.get("html_url"),
            "clone_url": data.get("clone_url"),
            "default_branch": data.get("default_branch", "main"),
        }

    async def _resolve_branch(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo_name: str,
        requested_branch: str | None,
    ) -> str:
        if requested_branch:
            return requested_branch
        resp = await self._get_repo(client, owner, repo_name)
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
        return "main"

    def _should_skip(self, path: Path, source_dir: Path) -> bool:
        rel_parts = path.relative_to(source_dir).parts
        if any(part in _SKIP_DIRS for part in rel_parts[:-1]):
            return True
        return path.name in _SKIP_FILES

    async def _upsert_file(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo_name: str,
        repo_path: str,
        raw_content: bytes,
        branch: str,
        commit_message: str,
    ) -> dict[str, Any]:
        sha = None
        existing_resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo_name}/contents/{repo_path}",
            params={"ref": branch},
        )

        if existing_resp.status_code == 200:
            existing = existing_resp.json()
            sha = existing.get("sha")
            if existing.get("encoding") == "base64" and existing.get("content"):
                existing_bytes = base64.b64decode(existing["content"].encode())
                if existing_bytes == raw_content:
                    return {"success": True, "skipped": True, "path": repo_path}
        elif existing_resp.status_code not in (404,):
            existing_resp.raise_for_status()

        payload = {
            "message": commit_message,
            "content": base64.b64encode(raw_content).decode(),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        put_resp = await client.put(
            f"{GITHUB_API}/repos/{owner}/{repo_name}/contents/{repo_path}",
            json=payload,
        )
        put_resp.raise_for_status()
        return {"success": True, "skipped": False, "path": repo_path}

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["validate_auth", "create_repo", "push_file", "push_directory", "git_push", "list_repos", "delete_repo", "get_repo"],
                    "description": "GitHub action to perform. Prefer 'git_push' over 'push_directory' for speed.",
                },
                "owner": {"type": "string", "description": "GitHub owner/org. Defaults to configured owner."},
                "repo_name": {"type": "string", "description": "Repository name."},
                "description": {"type": "string", "description": "Repo description."},
                "private": {"type": "boolean", "description": "Whether the repo is private.", "default": True},
                "file_path": {"type": "string", "description": "Path within the repo (for push_file)."},
                "content": {"type": "string", "description": "File content (for push_file)."},
                "source_dir": {"type": "string", "description": "Absolute directory to push (for push_directory / git_push)."},
                "commit_message": {"type": "string", "description": "Commit message.", "default": "Update via agent"},
                "branch": {"type": "string", "description": "Branch name. Defaults to the repo default branch."},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action")
        if not action:
            return {"success": False, "error": "Missing required parameter(s): action.", "failure_kind": "invalid_args"}
        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            try:
                if action == "validate_auth":
                    resp = await client.get(f"{GITHUB_API}/user")
                    resp.raise_for_status()
                    data = resp.json()
                    return {
                        "success": True,
                        "login": data.get("login"),
                        "name": data.get("name"),
                        "configured_owner": config.github_owner,
                        "scopes": resp.headers.get("X-OAuth-Scopes", ""),
                    }

                if action == "create_repo":
                    owner = self._owner(kwargs)
                    ensured = await self._ensure_repo(
                        client=client,
                        repo_name=kwargs["repo_name"],
                        description=kwargs.get("description", ""),
                        private=kwargs.get("private", True),
                        owner=owner,
                    )
                    return {"success": True, "owner": owner, **ensured}

                elif action == "push_file":
                    owner = self._owner(kwargs)
                    repo = kwargs["repo_name"]
                    path = kwargs["file_path"]
                    branch = await self._resolve_branch(client, owner, repo, kwargs.get("branch"))
                    result = await self._upsert_file(
                        client=client,
                        owner=owner,
                        repo_name=repo,
                        repo_path=path,
                        raw_content=kwargs["content"].encode(),
                        branch=branch,
                        commit_message=kwargs.get("commit_message", "Update via agent"),
                    )
                    return {
                        "success": True,
                        "owner": owner,
                        "branch": branch,
                        "message": f"Pushed {path} to {owner}/{repo}",
                        **result,
                    }

                elif action == "push_directory":
                    owner = self._owner(kwargs)
                    repo = kwargs["repo_name"]
                    source_dir = Path(kwargs["source_dir"]).resolve()
                    if not source_dir.exists() or not source_dir.is_dir():
                        return {
                            "success": False,
                            "error": f"Source directory not found: {source_dir}",
                            "failure_kind": "not_found",
                        }

                    await self._ensure_repo(
                        client=client,
                        repo_name=repo,
                        description=kwargs.get("description", ""),
                        private=kwargs.get("private", True),
                        owner=owner,
                    )
                    branch = await self._resolve_branch(client, owner, repo, kwargs.get("branch"))

                    pushed = 0
                    skipped = 0
                    failures: list[dict[str, str]] = []
                    for file_path in sorted(source_dir.rglob("*")):
                        if not file_path.is_file() or self._should_skip(file_path, source_dir):
                            continue

                        repo_path = file_path.relative_to(source_dir).as_posix()
                        try:
                            result = await self._upsert_file(
                                client=client,
                                owner=owner,
                                repo_name=repo,
                                repo_path=repo_path,
                                raw_content=file_path.read_bytes(),
                                branch=branch,
                                commit_message=kwargs.get("commit_message", f"Sync project files for {repo}"),
                            )
                            if result.get("skipped"):
                                skipped += 1
                            else:
                                pushed += 1
                        except Exception as exc:
                            failures.append({"path": repo_path, "error": str(exc)[:300]})

                    return {
                        "success": len(failures) == 0,
                        "owner": owner,
                        "repo": repo,
                        "branch": branch,
                        "pushed": pushed,
                        "skipped": skipped,
                        "failed": len(failures),
                        "failures": failures[:20],
                        "message": (
                            f"Synced directory {source_dir} to {owner}/{repo} "
                            f"(pushed={pushed}, skipped={skipped}, failed={len(failures)})"
                        ),
                    }

                elif action == "git_push":
                    # Fast Git CLI push — single commit, much faster than REST API
                    owner = self._owner(kwargs)
                    repo = kwargs.get("repo_name")
                    source_dir_str = kwargs.get("source_dir")
                    if not repo or not source_dir_str:
                        return {"success": False, "error": "Missing required parameter(s): repo_name, source_dir.", "failure_kind": "invalid_args"}

                    source_dir = Path(source_dir_str).resolve()
                    if not source_dir.exists() or not source_dir.is_dir():
                        return {"success": False, "error": f"Source directory not found: {source_dir}", "failure_kind": "not_found"}

                    # Ensure repo exists via API first
                    await self._ensure_repo(
                        client=client,
                        repo_name=repo,
                        description=kwargs.get("description", ""),
                        private=kwargs.get("private", True),
                        owner=owner,
                    )
                    branch = kwargs.get("branch") or "main"
                    commit_msg = kwargs.get("commit_message", "Update via agent")
                    remote_url = f"https://x-access-token:{config.github_token}@github.com/{owner}/{repo}.git"

                    # Build a shell script to init, add, commit, push
                    # Use -C to avoid cd issues; configure git to avoid identity prompts
                    git_script = (
                        f'cd "{source_dir}" && '
                        f'git init -b {branch} && '
                        f'git config user.email "agent@automated.dev" && '
                        f'git config user.name "Agent" && '
                        f'git add -A && '
                        f'git commit -m "{commit_msg}" --allow-empty && '
                        f'git remote remove origin 2>/dev/null; '
                        f'git remote add origin "{remote_url}" && '
                        f'git push -u origin {branch} --force'
                    )

                    try:
                        import shutil
                        bash_path = shutil.which("bash")
                        if bash_path:
                            proc = await asyncio.create_subprocess_exec(
                                bash_path, "-lc", git_script,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                cwd=str(source_dir),
                            )
                        else:
                            proc = await asyncio.create_subprocess_shell(
                                git_script,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                cwd=str(source_dir),
                            )
                        stdout_bytes, stderr_bytes = await asyncio.wait_for(
                            proc.communicate(), timeout=120
                        )
                        stdout_text = stdout_bytes.decode("utf-8", errors="replace")[:5000]
                        stderr_text = stderr_bytes.decode("utf-8", errors="replace")[:5000]
                        # Sanitize: remove token from any error output
                        stderr_text = stderr_text.replace(config.github_token, "***")
                        stdout_text = stdout_text.replace(config.github_token, "***")

                        if proc.returncode != 0:
                            return {
                                "success": False,
                                "error": f"git push failed (exit {proc.returncode}): {stderr_text}",
                                "stdout": stdout_text,
                                "failure_kind": "runtime_error",
                            }

                        return {
                            "success": True,
                            "owner": owner,
                            "repo": repo,
                            "branch": branch,
                            "message": f"Git pushed {source_dir} to {owner}/{repo} ({branch})",
                            "stdout": stdout_text,
                        }
                    except asyncio.TimeoutError:
                        return {"success": False, "error": "git push timed out after 120s", "failure_kind": "timeout"}
                    except Exception as exc:
                        error_msg = str(exc).replace(config.github_token, "***")
                        return {"success": False, "error": error_msg, "failure_kind": "runtime_error"}

                elif action == "list_repos":
                    resp = await client.get(f"{GITHUB_API}/user/repos", params={"per_page": 50, "sort": "updated"})
                    resp.raise_for_status()
                    repos = [{"name": r["name"], "url": r["html_url"], "private": r["private"]} for r in resp.json()]
                    return {"success": True, "repos": repos}

                elif action == "get_repo":
                    owner = self._owner(kwargs)
                    resp = await client.get(f"{GITHUB_API}/repos/{owner}/{kwargs['repo_name']}")
                    resp.raise_for_status()
                    return {"success": True, "repo": resp.json()}

                elif action == "delete_repo":
                    owner = self._owner(kwargs)
                    resp = await client.delete(f"{GITHUB_API}/repos/{owner}/{kwargs['repo_name']}")
                    resp.raise_for_status()
                    return {"success": True, "message": f"Deleted {owner}/{kwargs['repo_name']}"}

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
                    "error": f"GitHub API error: {status} {e.response.text[:500]}",
                    "failure_kind": failure_kind,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "failure_kind": "runtime_error"}
