"""Lazily builds and caches providers for one app instance, and closes their HTTP clients."""

from app.core.config import Settings
from app.providers.base import LLM, Embedder, Reranker, VectorStore
from app.providers.factory import build_embedder, build_llm, build_reranker, build_vector_store


class Providers:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm: LLM | None = None
        self._embedder: Embedder | None = None
        self._embedders: dict[str, Embedder] = {}
        self._store: VectorStore | None = None
        self._reranker: Reranker | None = None
        self._reranker_built = False
        self._rerank_cache = None

    @property
    def reranker(self) -> Reranker | None:
        """None when reranking is not configured (RERANK_PROVIDER=none)."""
        if not self._reranker_built:
            llm = self.llm if self._settings.rerank_provider == "llm" else None
            self._reranker = build_reranker(self._settings, llm)
            self._reranker_built = True
        return self._reranker

    @property
    def rerank_cache(self):
        """Scores of (query, passage) pairs already reranked, shared by every search."""
        if self._rerank_cache is None:
            from app.retrieve.rerank import RerankCache

            self._rerank_cache = RerankCache(
                self._settings.rerank_cache_entries, self._settings.rerank_cache_ttl_s
            )
        return self._rerank_cache

    @property
    def llm(self) -> LLM:
        self._llm = self._llm or build_llm(self._settings)
        return self._llm

    @property
    def embedder(self) -> Embedder:
        self._embedder = self._embedder or build_embedder(self._settings)
        return self._embedder

    def embedder_for(self, model: str | None) -> Embedder:
        """The embedder for a collection's model. Each collection fixes its embedding model at
        creation, so vectors in one index always come from one model."""
        if not model or model == self._settings.resolved_embed_model:
            return self.embedder
        if model not in self._embedders:
            self._embedders[model] = build_embedder(self._settings, model)
        return self._embedders[model]

    @property
    def vector_store(self) -> VectorStore:
        self._store = self._store or build_vector_store(self._settings)
        return self._store

    async def aclose(self) -> None:
        for p in (
            self._llm,
            self._embedder,
            self._store,
            self._reranker,
            *self._embedders.values(),
        ):
            client = getattr(p, "_client", None)
            if client is not None:
                await client.aclose()
