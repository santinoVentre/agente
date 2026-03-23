"""Memory module — short-term (conversation) and long-term (persistent facts)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select, update

from db.connection import async_session
from db.models import AgentMemory, Conversation
from utils.logging import setup_logging

log = setup_logging("memory")


class Memory:
    """Manages conversation history and long-term agent memory."""

    # ── Conversation history ─────────────────────────────────────────

    async def save_message(
        self,
        user_id: int,
        role: str,
        content: str,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
    ):
        async with async_session() as session:
            msg = Conversation(
                user_id=user_id,
                role=role,
                content=content,
                model_used=model,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cost_usd=cost,
            )
            session.add(msg)
            await session.commit()

    async def get_conversation_history(
        self, user_id: int, limit: int = 20
    ) -> list[dict[str, str]]:
        """Get recent conversation messages formatted for LLM context."""
        async with async_session() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            rows = list(result.scalars().all())
            rows.reverse()  # chronological order
            return [{"role": r.role, "content": r.content} for r in rows]

    # ── Long-term memory ─────────────────────────────────────────────

    async def remember(self, user_id: int, key: str, value: str, category: str = "general"):
        """Store or update a long-term memory fact."""
        async with async_session() as session:
            existing = await session.execute(
                select(AgentMemory).where(
                    AgentMemory.user_id == user_id,
                    AgentMemory.key == key,
                )
            )
            row = existing.scalar_one_or_none()
            if row:
                await session.execute(
                    update(AgentMemory)
                    .where(AgentMemory.id == row.id)
                    .values(value=value, category=category)
                )
            else:
                session.add(AgentMemory(
                    user_id=user_id, key=key, value=value, category=category
                ))
            await session.commit()
            log.info(f"Memory saved: [{category}] {key}")

    async def recall(self, user_id: int, key: str) -> str | None:
        """Retrieve a specific memory by key."""
        async with async_session() as session:
            result = await session.execute(
                select(AgentMemory).where(
                    AgentMemory.user_id == user_id,
                    AgentMemory.key == key,
                )
            )
            row = result.scalar_one_or_none()
            return row.value if row else None

    async def recall_by_category(self, user_id: int, category: str) -> list[dict[str, str]]:
        """Get all memories in a category."""
        async with async_session() as session:
            result = await session.execute(
                select(AgentMemory).where(
                    AgentMemory.user_id == user_id,
                    AgentMemory.category == category,
                )
            )
            return [{"key": r.key, "value": r.value} for r in result.scalars().all()]

    async def recall_all(self, user_id: int) -> list[dict[str, str]]:
        """Get all memories for context injection."""
        async with async_session() as session:
            result = await session.execute(
                select(AgentMemory).where(AgentMemory.user_id == user_id)
            )
            return [
                {"key": r.key, "value": r.value, "category": r.category}
                for r in result.scalars().all()
            ]

    async def forget(self, user_id: int, key: str):
        async with async_session() as session:
            await session.execute(
                delete(AgentMemory).where(
                    AgentMemory.user_id == user_id,
                    AgentMemory.key == key,
                )
            )
            await session.commit()

    def format_memories_for_prompt(self, memories: list[dict]) -> str:
        """Format memories as context for LLM prompts."""
        if not memories:
            return ""
        lines = ["## Known facts about the user:"]
        for m in memories:
            lines.append(f"- [{m.get('category', 'general')}] {m['key']}: {m['value']}")
        return "\n".join(lines)


# Singleton
memory = Memory()
