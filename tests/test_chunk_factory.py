import pytest

from app.core.config import Settings
from app.db.models import ChunkingStrategy
from app.ingest.chunking import ChunkingConfig, build_chunker
from app.ingest.chunking.contextual import ContextualChunker
from app.ingest.chunking.fixed import FixedChunker
from app.ingest.chunking.parent_child import ParentChildChunker
from app.ingest.chunking.recursive import RecursiveChunker
from app.ingest.chunking.semantic import SemanticChunker
from app.ingest.parsers import ParsedDocument
from app.providers.registry import Providers


@pytest.fixture
def providers():
    return Providers(
        Settings(_env_file=None, llm_provider="fake", embed_provider="fake", vector_store="memory")
    )


@pytest.mark.parametrize(
    ("strategy", "cls"),
    [
        (ChunkingStrategy.FIXED, FixedChunker),
        (ChunkingStrategy.RECURSIVE, RecursiveChunker),
        (ChunkingStrategy.PARENT_CHILD, ParentChildChunker),
        (ChunkingStrategy.SEMANTIC, SemanticChunker),
        (ChunkingStrategy.CONTEXTUAL, ContextualChunker),
    ],
)
def test_every_strategy_has_a_chunker(strategy, cls, providers):
    assert isinstance(build_chunker(strategy, providers), cls)


def test_structural_strategies_do_not_touch_model_providers():
    class Untouchable:
        @property
        def embedder(self):
            raise AssertionError("embedder must not be built")

        @property
        def llm(self):
            raise AssertionError("llm must not be built")

    for s in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE, ChunkingStrategy.PARENT_CHILD):
        build_chunker(s, Untouchable())


async def test_all_strategies_chunk_the_same_document_with_the_fake_providers(providers):
    text = "\n\n".join(
        f"Paragraph {i} is about topic {i % 3} and has some words." for i in range(60)
    )
    parsed = ParsedDocument(text=text)
    cfg = ChunkingConfig(
        target_tokens=64, overlap_tokens=8, max_tokens=128, parent_tokens=256, child_tokens=64
    )
    for strategy in ChunkingStrategy:
        drafts = await build_chunker(strategy, providers).chunk(parsed, cfg)
        assert drafts, strategy
        assert all(parsed.text[d.char_start : d.char_end] == d.text for d in drafts), strategy
