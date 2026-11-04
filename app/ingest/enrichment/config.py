"""Per-collection enrichment settings. Local enrichers are on by default (they are free);
LLM enrichers cost one call per chunk, so they are opt-in and capped."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EnrichmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: bool = Field(default=True, description="Detect each chunk's language (offline)")
    keywords: bool = Field(default=True, description="Extract keywords (offline)")
    max_keywords: int = Field(default=8, ge=1, le=20)
    summaries: bool = Field(default=False, description="LLM one-sentence summary per chunk")
    questions: bool = Field(
        default=False,
        description="LLM hypothetical questions each chunk answers; indexed for "
        "question-to-chunk matching",
    )
    max_questions: int = Field(default=3, ge=1, le=8)
    max_chunks: int = Field(
        default=500, ge=1, le=10_000, description="Cost cap: LLM calls per document"
    )
    concurrency: int = Field(default=4, ge=1, le=16, description="Parallel LLM calls")

    @property
    def uses_llm(self) -> bool:
        return self.summaries or self.questions
