from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from app.core.security import create_access_token
from app.db.models import User

GOOD = {"email": "ada@example.com", "password": "correct-horse-9", "display_name": "Ada"}


async def test_register_returns_user_without_secrets(client):
    r = await client.post("/v1/auth/register", json=GOOD)
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "ada@example.com" and body["display_name"] == "Ada"
    assert "password" not in r.text and "hash" not in r.text and "scrypt" not in r.text


async def test_register_normalises_email_and_rejects_case_variant_duplicates(client):
    r = await client.post("/v1/auth/register", json={**GOOD, "email": "  ADA@Example.COM "})
    assert r.json()["email"] == "ada@example.com"
    dup = await client.post("/v1/auth/register", json=GOOD)
    assert dup.status_code == 409
    assert dup.json()["type"].endswith("/conflict")


@pytest.mark.parametrize(
    "patch",
    [
        {"email": "not-an-email"},
        {"email": "a@b"},
        {"password": "short"},
        {"password": "x" * 129},
        {"display_name": "n" * 121},
    ],
)
async def test_register_validation(client, patch):
    r = await client.post("/v1/auth/register", json={**GOOD, **patch})
    assert r.status_code == 422
    assert "correct-horse-9" not in r.text


async def test_login_issues_a_working_token(client):
    await client.post("/v1/auth/register", json=GOOD)
    r = await client.post(
        "/v1/auth/login", json={"email": "ADA@example.com", "password": GOOD["password"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 60 * 24 * 60
    me = await client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert me.status_code == 200 and me.json()["email"] == "ada@example.com"


async def test_wrong_password_and_unknown_email_are_indistinguishable(client):
    await client.post("/v1/auth/register", json=GOOD)
    wrong = await client.post(
        "/v1/auth/login", json={"email": GOOD["email"], "password": "nope-nope-1"}
    )
    unknown = await client.post(
        "/v1/auth/login", json={"email": "who@example.com", "password": "nope-nope-1"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]
    assert wrong.headers["www-authenticate"] == "Bearer"


async def test_me_requires_a_token(client):
    r = await client.get("/v1/auth/me")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.headers["content-type"].startswith("application/problem+json")


@pytest.mark.parametrize("header", ["Bearer garbage", "Bearer ", "Basic abc", "token"])
async def test_me_rejects_bad_credentials(client, header):
    assert (await client.get("/v1/auth/me", headers={"Authorization": header})).status_code == 401


async def test_expired_token_is_rejected(client, settings, signup):
    headers = await signup()
    uid = (await client.get("/v1/auth/me", headers=headers)).json()["id"]
    token = create_access_token(uid, settings, now=datetime.now(UTC) - timedelta(days=3))
    r = await client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


async def test_token_for_deleted_or_deactivated_user_is_rejected(client, app, signup):
    headers = await signup()
    assert (await client.get("/v1/auth/me", headers=headers)).status_code == 200
    async with app.state.sessionmaker() as s:
        await s.execute(update(User).values(is_active=False))
        await s.commit()
    r = await client.get("/v1/auth/me", headers=headers)
    assert r.status_code == 401 and r.json()["detail"] == "Invalid or expired token"
    login = await client.post(
        "/v1/auth/login", json={"email": GOOD["email"], "password": "correct-horse-9"}
    )
    assert login.status_code == 401
