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
        actual_model = resp.get("_agent_model_used", model)
        await self._track_cost(actual_model, resp, task_id)
        return resp

    def _get_mid_fallback_model(self, model: str) -> str | None:
        fallback = (config.model_mid_fallback or "").strip()
        if not fallback:
            return None
        if model != config.model_mid:
            return None
        if fallback == model:
            return None
        return fallback

    def _should_try_mid_fallback(self, error: Exception) -> bool:
        if isinstance(error, json.JSONDecodeError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            return status in (400, 404, 422)
        if isinstance(error, httpx.RequestError):
            return True
        return False

    async def _post_and_parse(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            body_preview = resp.text[:1500]
            log.error(
                "OpenRouter HTTP %d for model %s — body: %s",
                resp.status_code,
                payload.get("model"),
                body_preview,
            )
        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            body_preview = resp.text[:1200]
            log.warning(
                "OpenRouter returned non-JSON body for model %s: %s",
                payload.get("model"),
                body_preview,
            )
            raise exc
        data["_agent_model_used"] = payload.get("model")
        return data

    async def _request_once(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._post_and_parse(client, payload)
        except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError) as primary_error:
            fallback_model = self._get_mid_fallback_model(str(payload.get("model", "")))
            if not fallback_model or not self._should_try_mid_fallback(primary_error):
                raise

            fallback_payload = dict(payload)
            fallback_payload["model"] = fallback_model
            log.warning(
                "OpenRouter MID fallback triggered: %s -> %s due to %s",
                payload.get("model"),
                fallback_model,
                primary_error,
            )
            data = await self._post_and_parse(client, fallback_payload)
            data["_agent_model_fallback_from"] = payload.get("model")
            return data

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
                return await self._request_once(client, payload)
            except httpx.HTTPStatusError as e:
                last_err = e
                status = e.response.status_code
                log.warning(f"OpenRouter request failed (attempt {attempt+1}/{max_retries}): HTTP {status}")
                # 4xx = client error (bad model name, invalid payload) — retry won't help
                if 400 <= status < 500:
                    raise RuntimeError(
                        f"OpenRouter client error {status} for model {payload.get('model')}: "
                        f"{e.response.text[:500]}"
                    )
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
            except (httpx.RequestError, json.JSONDecodeError) as e:
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
        # OpenRouter fields can vary by provider/model; fallback to estimator in cost_tracker.
        raw_cost = (
            usage.get("total_cost")
            or usage.get("total_cost_usd")
            or usage.get("cost")
            or response.get("total_cost")
            or response.get("cost")
            or 0
        )
        if not raw_cost:
            prompt_cost = usage.get("prompt_cost") or 0
            completion_cost = usage.get("completion_cost") or 0
            raw_cost = (prompt_cost or 0) + (completion_cost or 0)

        try:
            cost = float(raw_cost or 0)
        except (TypeError, ValueError):
            cost = 0.0

        await cost_tracker.record(model, tokens_in, tokens_out, cost, task_id=task_id)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton
openrouter = OpenRouterClient()
