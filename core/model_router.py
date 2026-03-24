"""Smart model routing — picks the optimal LLM based on task type."""

from __future__ import annotations

from enum import Enum

from config import config
from utils.logging import setup_logging

log = setup_logging("model_router")


class TaskType(str, Enum):
    ROUTING = "routing"            # Classify user intent → cheapest model
    SIMPLE_CHAT = "simple_chat"    # Quick answers, status checks
    CODE_GENERATION = "code_gen"   # Writing new code / tools
    CODE_REVIEW = "code_review"    # Reviewing / debugging code
    TOOL_EXECUTION = "tool_exec"   # Invoking existing tools via function calling
    COMPLEX_REASONING = "complex"  # Multi-step planning, architecture
    SUMMARIZATION = "summary"      # Summarizing conversations / docs
    WEB_DEV = "web_dev"            # Creating websites
    WEB_BROWSING = "web_browsing"  # Browsing, scraping, searching the web
    SYSTEM = "system"              # System administration, VPS management
    MEDIA = "media"                # Media processing instructions


# Maps task types to the config attribute holding the model name
_TASK_MODEL_MAP: dict[TaskType, str] = {
    TaskType.ROUTING: "model_cheap",
    TaskType.SIMPLE_CHAT: "model_cheap",
    TaskType.TOOL_EXECUTION: "model_cheap",
    TaskType.SUMMARIZATION: "model_cheap",
    TaskType.CODE_GENERATION: "model_mid",
    TaskType.CODE_REVIEW: "model_cheap",
    TaskType.COMPLEX_REASONING: "model_mid",
    TaskType.WEB_DEV: "model_cheap",
    TaskType.WEB_BROWSING: "model_cheap",
    TaskType.SYSTEM: "model_cheap",
    TaskType.MEDIA: "model_cheap",
}


def get_model_for_task(task_type: TaskType) -> str:
    attr = _TASK_MODEL_MAP.get(task_type, "model_mid")
    model = getattr(config, attr, config.model_mid)
    log.debug(f"Task type {task_type.value} → model {model}")
    return model


# Prompt used by the orchestrator to classify user intent
CLASSIFICATION_PROMPT = """You are a request classifier. Given the user's message, respond with EXACTLY ONE of these task types:
- routing: meta question about the system itself
- simple_chat: quick factual answer, greetings, status checks
- code_gen: writing new code, creating tools, scripts, or programs
- code_review: reviewing, debugging, or improving existing code
- tool_exec: executing an existing tool or capability
- complex: multi-step planning, architecture, detailed analysis
- summary: summarizing conversations, documents, or data
- web_dev: creating or modifying websites
- web_browsing: browsing websites, scraping content, searching online
- system: system administration, server management, VPS ops
- media: image/video processing, download, conversion

Respond with ONLY the task type string, nothing else."""
