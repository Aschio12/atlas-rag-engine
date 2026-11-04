"""What a caller can ask of answering (Phase 4)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, model_validator

from app.generate.policy import InjectionMode, PiiMode
from app.retrieve.filters import MAX_CONDITIONS, FilterError, MetadataFilter, compile_filters
from app.retrieve.options import RetrievalOptions


class AnswerOptions(BaseModel):
    retrieval: RetrievalOptions | None = Field(
        None,
        description="How to search. Default: hybrid, top 8, reranked when a reranker is configured",
    )
    collection_ids: list[uuid.UUID] | None = Field(
        None, max_length=50, description="Answer from only these collections; default: all"
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        max_length=MAX_CONDITIONS,
        description="Metadata filters, all must hold",
    )
    temperature: float = Field(0.1, ge=0, le=1)
    max_tokens: int = Field(800, ge=64, le=4000, description="Longest answer to generate")
    prompt_version: int | None = Field(
        None, ge=1, description="Pin a version of the answer prompt; default: the latest"
    )
    abstain_below: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Say 'not found' below this confidence; default: the workspace's",
    )
    pii: PiiMode | None = Field(
        None, description="Redact personal data; can only tighten the policy"
    )
    injection: InjectionMode | None = Field(
        None, description="Treatment of suspicious passages; can only tighten the policy"
    )
    rewrite_follow_ups: bool = Field(
        True, description="Make a follow-up question stand alone before searching"
    )

    @model_validator(mode="after")
    def _filters_are_consistent(self) -> AnswerOptions:
        try:
            compile_filters(self.filters)
        except FilterError as e:
            raise ValueError(str(e)) from None
        return self
