"""Context compressor — keeps recent turns + a compressed summary to minimise token usage.

Called by BaseAgent every SUMMARY_INTERVAL steps.
Always uses the CHEAP model so compression itself costs almost nothing.
"""

from __future__ import annotations

from config import config
from core.openrouter_client import openrouter
from utils.logging import setup_logging

log = setup_logging("context_compressor")

_COMPRESS_SYSTEM = (
    "Compress the following conversation history into a concise summary of at most 150 words. "
    "Preserve: task objective, key decisions made, files created, errors encountered, current state. "
    "Output ONLY the summary — no headers, no bullet labels."
)

_MIN_MESSAGES_TO_COMPRESS = 3  # need at least 3 conv turns before it's worth compressing


def _find_safe_split(rest: list[dict], keep_recent: int = 2) -> int:
    """Find a safe split index so we never orphan tool_use/tool_result pairs.

    Returns the index in `rest` where the "keep" portion starts.
    The split point is moved earlier if cutting at -keep_recent would orphan
    a tool result (i.e. separate it from its preceding assistant+tool_calls).
    """
    if len(rest) <= keep_recent:
        return 0

    split = len(rest) - keep_recent

    # Walk backwards from split: if the message at `split` is a tool result,
    # its assistant (with tool_calls) must be somewhere before split.
    # We need to include the full tool-call block in the kept portion.
    while split > 0 and rest[split]["role"] == "tool":
        split -= 1

    # Also check: if the message at split is an assistant with tool_calls,
    # the tool results after it must all be in the kept portion too.
    if split > 0 and split < len(rest):
        msg = rest[split - 1]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # The assistant with tool_calls is in the "to_compress" section,
            # but its tool_results might start at split — include the assistant too
            split -= 1

    return max(split, 0)


async def compress_messages(
    messages: list[dict],
    task_id: int | None = None,
) -> list[dict]:
    """
    Given the current message list, compress older conversation turns
    into a single [CONTEXT SUMMARY] system message.

    System messages that were already summaries are included in what gets compressed.
    Always keeps the original system prompt (first message if role==system) intact.
    Never splits assistant(tool_calls) from their tool(result) messages.
    Uses the cheap model — typically < 0.001 USD per compression.
    """
    if not messages:
        return messages

    # Separate the original system prompt from the rest
    original_system: list[dict] = []
    rest: list[dict] = []
    _seen_first_system = False
    for m in messages:
        if m["role"] == "system" and not _seen_first_system:
            original_system.append(m)
            _seen_first_system = True
        else:
            rest.append(m)

    # Only non-system turns count towards compression threshold
    conv_turns = [m for m in rest if m["role"] in ("user", "assistant")]
    if len(conv_turns) < _MIN_MESSAGES_TO_COMPRESS:
        return messages

    # Find safe split point that doesn't orphan tool pairs
    split_idx = _find_safe_split(rest, keep_recent=2)

    to_compress = rest[:split_idx]
    keep_recent = rest[split_idx:]

    if not to_compress:
        return messages

    # Build the text to summarise (truncate each message to keep the request cheap)
    lines: list[str] = []
    for m in to_compress:
        role = m["role"].upper()
        content = str(m.get("content") or "")[:600]
        lines.append(f"{role}: {content}")
    text_to_compress = "\n".join(lines)

    try:
        response = await openrouter.chat(
            model=config.model_cheap,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": text_to_compress},
            ],
            temperature=0.0,
            max_tokens=250,
            task_id=task_id,
        )
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception as exc:
        log.warning(f"Context compression failed (task {task_id}): {exc}")
        summary = f"[{len(to_compress)} earlier messages — compression failed]"

    summary_msg = {"role": "system", "content": f"[CONTEXT SUMMARY]\n{summary}"}
    compressed = original_system + [summary_msg] + keep_recent
    log.debug(
        f"[task {task_id}] Compressed {len(to_compress)} messages → 1 summary "
        f"({len(messages)} → {len(compressed)} total)"
    )
    return compressed
