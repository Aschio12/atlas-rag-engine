"""FastAPI dependency providers (dependency injection entry points)."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings


def settings_dep(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(settings_dep)]


async def session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """One database session per request.

    Handlers commit explicitly (`await session.commit()`) before returning, so a failed commit
    surfaces as an error response instead of happening after the client was told "201 Created".
    Anything not committed is rolled back when the session closes.
    """
    async with request.app.state.sessionmaker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(session_dep)]
