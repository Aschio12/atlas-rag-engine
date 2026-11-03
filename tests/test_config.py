import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_work_without_any_env():
    s = make()
    assert s.llm_provider == "ollama"
    assert s.app_env == "dev"


def test_prod_rejects_default_jwt_secret():
    with pytest.raises(ValidationError):
        make(app_env="prod")


def test_prod_rejects_short_secret():
    with pytest.raises(ValidationError):
        make(app_env="prod", jwt_secret="short-but-not-default")


@pytest.mark.parametrize(
    "placeholder",
    [
        "change-me-generate-one-with-openssl-rand-hex-32",
        "dev-only-insecure-jwt-secret-do-not-use-in-production",
    ],
)
def test_prod_rejects_long_placeholder_secrets(placeholder):
    with pytest.raises(ValidationError):
        make(app_env="prod", jwt_secret=placeholder)


def test_dev_default_secret_is_long_enough_for_hs256():
    assert len(make().jwt_secret.get_secret_value()) >= 32


def test_prod_accepts_strong_secret():
    assert make(app_env="prod", jwt_secret="x" * 40).app_env == "prod"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_hosted_provider_requires_key(provider):
    with pytest.raises(ValidationError):
        make(llm_provider=provider)


def test_hosted_provider_with_key_ok():
    assert make(llm_provider="openai", openai_api_key="sk-test").llm_provider == "openai"


def test_secrets_are_not_leaked_in_repr():
    assert "sk-test" not in repr(make(llm_provider="openai", openai_api_key="sk-test"))
