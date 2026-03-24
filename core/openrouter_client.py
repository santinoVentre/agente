"""Async OpenRouter API client with streaming, retry, and cost tracking."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from config import config
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("openrouter")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Thin async wrapper around the OpenRouter chat completions API."""

    def __init__(self):
        self._api_key = config.openrouter_api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/santinosagent/agent-infra",
                    "X-Title": "Personal Agent Infrastructure",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full response dict."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        resp = await self._request_with_retry(client, payload)
        await self._track_cost(model, resp, task_id)
        return resp

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Streaming chat completion. Yields text chunks."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    async def _request_with_retry(
        self, client: httpx.AsyncClient, payload: dict, max_retries: int = 3
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_err = e
                log.warning(f"OpenRouter request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter request failed after {max_retries} retries: {last_err}")

    async def _track_cost(self, model: str, response: dict, task_id: int | None = None):
        usage = response.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        # OpenRouter returns cost in the response header or we estimate
        cost = float(response.get("usage", {}).get("total_cost", 0) or 0)
        await cost_tracker.record(model, tokens_in, tokens_out, cost, task_id=task_id)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton
openrouter = OpenRouterClient()
