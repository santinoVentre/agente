"""Base class for all tools in the agent infrastructure."""

from __future__ import annotations

import abc
import json
from typing import Any

from db.models import RiskLevel


class BaseTool(abc.ABC):
    """Every tool must subclass this and implement execute()."""

    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    version: int = 1
    source: str = "builtin"  # "builtin" or "generated"

    @abc.abstractmethod
    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the tool's parameters (OpenAI function-calling format)."""

    @abc.abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """Run the tool with the given parameters. Returns a result dict."""

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function definition for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def __repr__(self):
        return f"<Tool {self.name} v{self.version} risk={self.risk_level.value}>"
