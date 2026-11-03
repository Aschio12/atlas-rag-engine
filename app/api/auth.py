"""Registration, login and the current user."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.auth import CurrentUser
from app.core.deps import SessionDep, SettingsDep
from app.core.errors import Conflict, Unauthorized
from app.core.security import (
    MAX_PASSWORD_LENGTH,
    create_access_token,
    hash_password,
    verify_against_dummy,
    verify_password,
)
from app.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalise_email(value: str) -> str:
    value = value.strip().lower()
    if len(value) > 320 or not _EMAIL.match(value):
        raise ValueError("must be a valid email address")
    return value


class RegisterIn(BaseModel):
    email: str = Field(examples=["ada@example.com"])
    password: str = Field(
        min_length=8, max_length=MAX_PASSWORD_LENGTH, examples=["correct-horse-9"]
    )
    display_name: str = Field(default="", max_length=120, examples=["Ada Lovelace"])

    _email = field_validator("email")(_normalise_email)


class LoginIn(BaseModel):
    email: str
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)

    _email = field_validator("email")(_normalise_email)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until the token expires")


@router.post("/register", status_code=201, response_model=UserOut)
async def register(body: RegisterIn, session: SessionDep) -> User:
    if (await session.execute(select(User.id).where(User.email == body.email))).first():
        raise Conflict("An account with this email already exists")
    # scrypt is CPU-bound (tens of ms): keep it off the event loop.
    pw_hash = await asyncio.to_thread(hash_password, body.password)
    user = User(email=body.email, password_hash=pw_hash, display_name=body.display_name.strip())
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:  # lost a race with a concurrent registration of the same email
        raise Conflict("An account with this email already exists") from None
    return user


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, session: SessionDep, settings: SettingsDep) -> TokenOut:
    user = (
        await session.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if user is None:
        await asyncio.to_thread(verify_against_dummy, body.password)
    ok = user is not None and await asyncio.to_thread(
        verify_password, body.password, user.password_hash
    )
    if not ok or not user.is_active:
        raise Unauthorized("Incorrect email or password")
    return TokenOut(
        access_token=create_access_token(str(user.id), settings),
        expires_in=settings.jwt_ttl_minutes * 60,
    )


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
