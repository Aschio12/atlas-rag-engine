"""Application settings, grouped by concern and loaded from the environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder secrets: fine for local development, rejected in production.
INSECURE_JWT_PREFIXES = ("change-me", "dev-only", "secret")
DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-do-not-use-in-production"


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

    # --- documents and ingestion (Phase 2) -----------------------------------
    storage_dir: str = "./data/blobs"  # originals and parsed text; one folder per workspace
    max_upload_mb: int = Field(default=25, gt=0, le=500)
    allow_private_urls: bool = False  # let URL ingestion reach localhost / private networks
    url_fetch_timeout_s: float = Field(default=15.0, gt=0)
    embed_batch_size: int = Field(default=32, ge=1, le=512)  # texts per embedding request
    embed_max_chars: int = Field(default=8000, ge=256)  # longer texts are truncated, and counted
    # --- connectors (Phase 2.7) --------------------------------------------
    github_token: SecretStr = SecretStr("")  # optional: private repos and a higher rate limit
    github_api_url: str = "https://api.github.com"
    github_raw_url: str = "https://raw.githubusercontent.com"
    source_max_per_collection: int = Field(default=20, ge=1)
    source_sync_concurrency: int = Field(default=2, ge=1, le=8)
    ocr_enabled: bool = True  # use Tesseract for scanned PDF pages when it is installed
    ocr_language: str = "eng"  # Tesseract language code(s), e.g. "eng+deu"
    worker_enabled: bool = True  # run the ingestion worker inside the API process
    worker_poll_s: float = Field(default=1.0, gt=0)
    job_max_attempts: int = Field(default=3, ge=1)
    sse_poll_s: float = Field(default=0.5, gt=0)  # how often a job event stream checks for change
    job_backoff_s: float = Field(default=2.0, ge=0)  # doubled after every failed attempt

    # --- retrieval (Phase 3) ------------------------------------------------
    # Searches are saved with their trace for the Inspector; the newest N are kept per
    # workspace and older ones are deleted as new ones arrive. 0 turns saving off.
    retrieval_history_limit: int = Field(default=500, ge=0)

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
    # "none" disables reranking: a search that asks for it gets a clear error.
    rerank_provider: Literal["none", "fake", "llm", "cohere", "local"] = "none"
    rerank_model: str = ""  # empty = the provider's default (see resolved_rerank_model)
    rerank_api_key: SecretStr = SecretStr("")
    # Cohere by default; Jina (https://api.jina.ai/v1) and servers copying the API also work.
    rerank_base_url: str = "https://api.cohere.com/v2"
    rerank_cache_entries: int = Field(default=20_000, ge=0)  # (query, passage) scores kept
    rerank_cache_ttl_s: float = Field(default=3600.0, gt=0)

    # --- vector store ------------------------------------------------------
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_api_key: SecretStr = SecretStr("")

    # --- auth --------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    jwt_ttl_minutes: int = Field(default=60 * 24, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

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

    @property
    def resolved_rerank_model(self) -> str:
        defaults = {
            "none": "",
            "fake": "fake-rerank",
            "llm": self.resolved_llm_model,
            "cohere": "rerank-v3.5",
            "local": "BAAI/bge-reranker-base",
        }
        return self.rerank_model or defaults[self.rerank_provider]

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        secret = self.jwt_secret.get_secret_value()
        if self.app_env == "prod" and (
            len(secret) < 32 or secret.lower().startswith(INSECURE_JWT_PREFIXES)
        ):
            raise ValueError(
                "JWT_SECRET must be a real secret of at least 32 characters when APP_ENV=prod "
                "(generate one with: openssl rand -hex 32)"
            )
        if self.llm_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key.get_secret_value():
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.embed_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required when EMBED_PROVIDER=openai")
        if self.rerank_provider == "cohere" and not self.rerank_api_key.get_secret_value():
            raise ValueError("RERANK_API_KEY is required when RERANK_PROVIDER=cohere")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
