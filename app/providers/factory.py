"""Build providers from Settings. The only place that knows which concrete class is used."""

from app.core.config import Settings
from app.providers.anthropic import AnthropicLLM
from app.providers.base import LLM, Embedder, Reranker, VectorStore
from app.providers.fake import FakeEmbedder, FakeLLM, FakeReranker, InMemoryVectorStore
from app.providers.ollama import OllamaEmbedder, OllamaLLM
from app.providers.openai import OpenAIEmbedder, OpenAILLM
from app.providers.qdrant import QdrantStore
from app.providers.rerank import CrossEncoderReranker, HTTPReranker, LLMReranker


def build_llm(s: Settings) -> LLM:
    model = s.resolved_llm_model
    match s.llm_provider:
        case "ollama":
            return OllamaLLM(s.ollama_url, model)
        case "openai":
            return OpenAILLM(s.openai_api_key.get_secret_value(), model, s.openai_base_url)
        case "anthropic":
            return AnthropicLLM(s.anthropic_api_key.get_secret_value(), model)
        case "fake":
            return FakeLLM(model)


def build_embedder(s: Settings, model: str | None = None) -> Embedder:
    """The configured embedding provider, for `model` (default: the configured model)."""
    model = model or s.resolved_embed_model
    match s.embed_provider:
        case "ollama":
            return OllamaEmbedder(s.ollama_url, model)
        case "openai":
            return OpenAIEmbedder(s.openai_api_key.get_secret_value(), model, s.openai_base_url)
        case "fake":
            return FakeEmbedder(model=model)


def build_vector_store(s: Settings) -> VectorStore:
    if s.vector_store == "qdrant":
        return QdrantStore(s.qdrant_url, s.qdrant_api_key.get_secret_value())
    return InMemoryVectorStore()


def build_reranker(s: Settings, llm: LLM | None = None) -> Reranker | None:
    """The configured reranker, or None when reranking is switched off. `llm` is the chat model
    the `llm` reranker grades with."""
    match s.rerank_provider:
        case "none":
            return None
        case "fake":
            return FakeReranker()
        case "llm":
            return LLMReranker(llm or build_llm(s))
        case "cohere":
            return HTTPReranker(
                s.rerank_api_key.get_secret_value(), s.resolved_rerank_model, s.rerank_base_url
            )
        case "local":
            return CrossEncoderReranker(s.resolved_rerank_model)
