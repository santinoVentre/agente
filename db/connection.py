"""Async database connection pool using SQLAlchemy."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import config

engine = create_async_engine(
    config.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables (for initial setup / dev). Use Alembic in prod."""
    from db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()
