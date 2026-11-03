"""Shared fixtures: an app on in-memory SQLite, with schema created from the models."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.db.base import Base
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider="fake",
        embed_provider="fake",
        vector_store="memory",
        jwt_secret="t" * 40,
        log_level="WARNING",
        storage_dir=str(tmp_path / "blobs"),
        worker_enabled=False,
        ocr_enabled=False,
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


@pytest.fixture
def signup(client):
    """Create a user and return Authorization headers for them."""

    async def _signup(email: str = "ada@example.com", password: str = "correct-horse-9", **extra):
        r = await client.post(
            "/v1/auth/register", json={"email": email, "password": password, **extra}
        )
        assert r.status_code == 201, r.text
        r = await client.post("/v1/auth/login", json={"email": email, "password": password})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _signup


DOCS_URL = "/v1/workspaces/{ws}/collections/{col}/documents"


@pytest.fixture
async def space(client, signup):
    """A signed-in owner with a workspace and one empty collection."""
    headers = await signup()
    ws = (await client.post("/v1/workspaces", json={"name": "Acme"}, headers=headers)).json()
    col = (
        await client.post(
            f"/v1/workspaces/{ws['id']}/collections", json={"name": "Manuals"}, headers=headers
        )
    ).json()
    return {
        "headers": headers,
        "ws": ws["id"],
        "col": col["id"],
        "url": DOCS_URL.format(ws=ws["id"], col=col["id"]),
    }
