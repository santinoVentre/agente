"""Context compressor — keeps last 2 turns + a compressed summary to minimise token usage.

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


async def compress_messages(
    messages: list[dict],
    task_id: int | None = None,
) -> list[dict]:
    """
    Given the current message list, compress all conversation turns except the last 2
    into a single [CONTEXT SUMMARY] system message.

    System messages that were already summaries are included in what gets compressed.
    Always keeps the original system prompt (first message if role==system) intact.
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

    # Split: everything except the last 2 turns → compress; last 2 → keep
    to_compress = rest[:-2]
    keep_recent = rest[-2:]

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
