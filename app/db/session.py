"""Async engine and session management."""

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    url = settings.database_url
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite is for tests and zero-Docker local runs. An in-memory database exists per
        # connection, so share a single one.
        if ":memory:" in url or url.endswith("://"):
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        engine = create_async_engine(url, **kwargs)

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_conn, _record):
            # Off by default in SQLite; without it ON DELETE CASCADE would silently not run.
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_async_engine(url, **kwargs)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(
    maker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session; commit on success, roll back on error. For background jobs and scripts."""
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
