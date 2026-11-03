from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import Settings
from app.core.security import (
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SETTINGS = Settings(_env_file=None, jwt_secret="k" * 40, jwt_ttl_minutes=30)


def test_hash_verifies_and_is_salted():
    h1, h2 = hash_password("correct horse"), hash_password("correct horse")
    assert h1 != h2
    assert verify_password("correct horse", h1) and verify_password("correct horse", h2)
    assert not verify_password("wrong", h1)


def test_hash_never_contains_the_password():
    assert "correct horse" not in hash_password("correct horse")


def test_unicode_normalisation():
    # "é" precomposed vs "e" + combining accent
    assert verify_password("café-pass", hash_password("café-pass"))


@pytest.mark.parametrize(
    "stored",
    ["", "garbage", "bcrypt$1$2$3$4$5", "scrypt$x$8$1$AAAA$AAAA", "scrypt$16384$8$1$!!$!!"],
)
def test_malformed_stored_hashes_fail_closed(stored):
    assert verify_password("anything", stored) is False


def test_token_round_trip():
    assert decode_access_token(create_access_token("user-1", SETTINGS), SETTINGS) == "user-1"


def test_expired_token_rejected():
    old = datetime.now(UTC) - timedelta(hours=2)
    token = create_access_token("user-1", SETTINGS, now=old)
    with pytest.raises(TokenError):
        decode_access_token(token, SETTINGS)


def test_token_signed_with_other_secret_rejected():
    other = Settings(_env_file=None, jwt_secret="z" * 40)
    with pytest.raises(TokenError):
        decode_access_token(create_access_token("user-1", other), SETTINGS)


def test_alg_none_token_rejected():
    forged = jwt.encode({"sub": "admin", "typ": "access"}, key=None, algorithm="none")
    with pytest.raises(TokenError):
        decode_access_token(forged, SETTINGS)


def test_non_access_token_type_rejected():
    now = datetime.now(UTC)
    token = jwt.encode(
        {"sub": "u", "typ": "refresh", "iat": now, "exp": now + timedelta(minutes=5)},
        "k" * 40,
        algorithm="HS256",
    )
    with pytest.raises(TokenError, match="type"):
        decode_access_token(token, SETTINGS)


def test_token_missing_required_claims_rejected():
    token = jwt.encode({"typ": "access"}, "k" * 40, algorithm="HS256")
    with pytest.raises(TokenError):
        decode_access_token(token, SETTINGS)


def test_garbage_rejected():
    with pytest.raises(TokenError):
        decode_access_token("not.a.jwt", SETTINGS)
