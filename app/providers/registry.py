"""Lazily builds and caches providers for one app instance, and closes their HTTP clients."""

from app.core.config import Settings
from app.providers.base import LLM, Embedder, VectorStore
from app.providers.factory import build_embedder, build_llm, build_vector_store


class Providers:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm: LLM | None = None
        self._embedder: Embedder | None = None
        self._store: VectorStore | None = None

    @property
    def llm(self) -> LLM:
        self._llm = self._llm or build_llm(self._settings)
        return self._llm

    @property
    def embedder(self) -> Embedder:
        self._embedder = self._embedder or build_embedder(self._settings)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        self._store = self._store or build_vector_store(self._settings)
        return self._store

    async def aclose(self) -> None:
        for p in (self._llm, self._embedder, self._store):
            client = getattr(p, "_client", None)
            if client is not None:
                await client.aclose()
