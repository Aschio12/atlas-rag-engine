import pytest

from app.core.config import Settings
from app.providers.anthropic import AnthropicLLM
from app.providers.factory import build_embedder, build_llm, build_vector_store
from app.providers.fake import FakeEmbedder, FakeLLM, InMemoryVectorStore
from app.providers.ollama import OllamaEmbedder, OllamaLLM
from app.providers.openai import OpenAIEmbedder, OpenAILLM
from app.providers.qdrant import QdrantStore


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize(
    ("kw", "cls", "model"),
    [
        ({}, OllamaLLM, "llama3.1:8b"),
        ({"llm_provider": "fake"}, FakeLLM, "fake-llm"),
        ({"llm_provider": "openai", "openai_api_key": "k"}, OpenAILLM, "gpt-4o-mini"),
        (
            {"llm_provider": "anthropic", "anthropic_api_key": "k"},
            AnthropicLLM,
            "claude-sonnet-5-5",
        ),
        ({"llm_provider": "fake", "llm_model": "custom"}, FakeLLM, "custom"),
    ],
)
def test_build_llm(kw, cls, model):
    llm = build_llm(make(**kw))
    assert isinstance(llm, cls) and llm.model == model


@pytest.mark.parametrize(
    ("kw", "cls", "model"),
    [
        ({}, OllamaEmbedder, "nomic-embed-text"),
        ({"embed_provider": "fake"}, FakeEmbedder, "fake-embed"),
        (
            {"embed_provider": "openai", "openai_api_key": "k"},
            OpenAIEmbedder,
            "text-embedding-3-small",
        ),
    ],
)
def test_build_embedder(kw, cls, model):
    e = build_embedder(make(**kw))
    assert isinstance(e, cls) and e.model == model


def test_build_vector_store():
    assert isinstance(build_vector_store(make()), QdrantStore)
    assert isinstance(build_vector_store(make(vector_store="memory")), InMemoryVectorStore)


def test_openai_embeddings_require_key():
    with pytest.raises(ValueError):
        make(embed_provider="openai")
