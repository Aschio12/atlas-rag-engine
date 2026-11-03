"""Password hashing and access tokens (Phase 1.6)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import unicodedata
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt

from app.core.config import Settings

# --- passwords --------------------------------------------------------------------------------
# scrypt from the standard library: memory-hard, no extra dependency. Parameters are stored in the
# hash string so they can be raised later without breaking existing hashes.
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32
_MAXMEM = 64 * 1024 * 1024
MAX_PASSWORD_LENGTH = 128


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    # NFKC so visually identical passwords typed on different systems hash the same.
    pw = unicodedata.normalize("NFKC", password).encode()
    return hashlib.scrypt(pw, salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=_MAXMEM)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _derive(password, salt, _N, _R, _P, _DKLEN)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Returns False (never raises) for malformed stored hashes."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        actual = _derive(password, salt, int(n), int(r), int(p), len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


@lru_cache
def _dummy_hash() -> str:
    return hash_password("atlas-timing-equaliser")


def verify_against_dummy(password: str) -> None:
    """Spend the same time as a real check, so unknown emails aren't distinguishable by latency."""
    verify_password(password, _dummy_hash())


# --- access tokens ----------------------------------------------------------------------------
ALGORITHM = "HS256"
TOKEN_TYPE = "access"


class TokenError(Exception):
    """The token is missing, malformed, expired, tampered with, or not an access token."""


def create_access_token(user_id: str, settings: Settings, *, now: datetime | None = None) -> str:
    issued = now or datetime.now(UTC)
    claims = {
        "sub": user_id,
        "typ": TOKEN_TYPE,
        "iat": issued,
        "exp": issued + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> str:
    """Return the user id (`sub`) or raise TokenError."""
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[ALGORITHM],  # pinned: rejects alg=none and algorithm-confusion tokens
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise TokenError(type(exc).__name__) from exc
    if claims.get("typ") != TOKEN_TYPE:
        raise TokenError("wrong token type")
    return claims["sub"]
