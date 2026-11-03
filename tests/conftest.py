"""Shared fixtures: an app on in-memory SQLite, with schema created from the models."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.db.base import Base
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider="fake",
        embed_provider="fake",
        vector_store="memory",
        jwt_secret="t" * 40,
        log_level="WARNING",
    )


@pytest.fixture
async def app(settings):
    application = create_app(settings)
    async with application.state.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield application
    await application.state.engine.dispose()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
