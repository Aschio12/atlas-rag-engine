"""Application settings, grouped by concern and loaded from the environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_JWT_SECRETS = {"", "change-me", "secret"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"

    # --- logging -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- storage -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM ---------------------------------------------------------------
    # "fake" needs no model or key: handy for first runs, demos and CI.
    llm_provider: Literal["ollama", "openai", "anthropic", "fake"] = "ollama"
    llm_model: str = ""  # empty = the provider's default (see resolved_llm_model)
    ollama_url: str = "http://localhost:11434"
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible server
    anthropic_api_key: SecretStr = SecretStr("")

    # --- embeddings / reranking -------------------------------------------
    embed_provider: Literal["ollama", "openai", "fake"] = "ollama"
    embed_model: str = ""  # empty = the provider's default
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- vector store ------------------------------------------------------
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_api_key: SecretStr = SecretStr("")

    # --- auth --------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("change-me")
    jwt_ttl_minutes: int = Field(default=60 * 24, gt=0)

    @property
    def resolved_llm_model(self) -> str:
        defaults = {
            "ollama": "llama3.1:8b",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-5-5",
            "fake": "fake-llm",
        }
        return self.llm_model or defaults[self.llm_provider]

    @property
    def resolved_embed_model(self) -> str:
        defaults = {
            "ollama": "nomic-embed-text",
            "openai": "text-embedding-3-small",
            "fake": "fake-embed",
        }
        return self.embed_model or defaults[self.embed_provider]

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.app_env == "prod" and self.jwt_secret.get_secret_value() in INSECURE_JWT_SECRETS:
            raise ValueError("JWT_SECRET must be set to a strong value when APP_ENV=prod")
        if self.llm_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key.get_secret_value():
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.embed_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required when EMBED_PROVIDER=openai")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
