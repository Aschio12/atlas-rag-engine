"""Strategy name -> chunker."""

from __future__ import annotations

from app.db.models import ChunkingStrategy
from app.ingest.chunking.base import Chunker
from app.ingest.chunking.contextual import ContextualChunker
from app.ingest.chunking.fixed import FixedChunker
from app.ingest.chunking.parent_child import ParentChildChunker
from app.ingest.chunking.recursive import RecursiveChunker
from app.ingest.chunking.semantic import SemanticChunker
from app.providers.registry import Providers


def build_chunker(strategy: ChunkingStrategy, providers: Providers) -> Chunker:
    """Providers are only touched for the strategies that need them (embedder for semantic,
    LLM for contextual), so the structural strategies work with no model configured."""
    match strategy:
        case ChunkingStrategy.FIXED:
            return FixedChunker()
        case ChunkingStrategy.RECURSIVE:
            return RecursiveChunker()
        case ChunkingStrategy.PARENT_CHILD:
            return ParentChildChunker()
        case ChunkingStrategy.SEMANTIC:
            return SemanticChunker(providers.embedder)
        case ChunkingStrategy.CONTEXTUAL:
            return ContextualChunker(providers.llm)
    raise ValueError(f"unknown chunking strategy: {strategy!r}")
