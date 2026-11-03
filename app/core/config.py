"""Application settings, grouped by concern and loaded from the environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_JWT_SECRETS = {"", "change-me", "secret"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"

    # --- storage -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")

    # --- embeddings / reranking -------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- auth --------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("change-me")
    jwt_ttl_minutes: int = Field(default=60 * 24, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.app_env == "prod" and self.jwt_secret.get_secret_value() in INSECURE_JWT_SECRETS:
            raise ValueError("JWT_SECRET must be set to a strong value when APP_ENV=prod")
        if self.llm_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key.get_secret_value():
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
