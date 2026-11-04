"""Per-collection chunking settings, validated once and stored with the collection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Sizes are in estimated tokens (see `estimate_tokens`). Every strategy reads only the
    fields it needs, so one config shape serves all of them and can be tuned by evaluation."""

    model_config = ConfigDict(extra="forbid")

    target_tokens: int = Field(default=400, ge=32, le=4000, description="Aimed-for chunk size")
    overlap_tokens: int = Field(default=50, ge=0, le=1000, description="Shared with the neighbour")
    max_tokens: int = Field(default=800, ge=64, le=8000, description="Hard ceiling per chunk")
    # semantic
    breakpoint_percentile: float = Field(
        default=90.0, ge=50.0, le=99.9, description="Split where topic distance exceeds this"
    )
    # parent-child
    parent_tokens: int = Field(default=1200, ge=128, le=8000)
    child_tokens: int = Field(default=200, ge=32, le=2000)
    # contextual
    context_max_chunks: int = Field(
        default=500, ge=1, le=10_000, description="Cost cap: chunks beyond this get no context"
    )

    @model_validator(mode="after")
    def _consistent(self) -> ChunkingConfig:
        if self.overlap_tokens * 2 > self.target_tokens:
            raise ValueError("overlap_tokens must be at most half of target_tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be at least target_tokens")
        if self.child_tokens * 2 > self.parent_tokens:
            raise ValueError("parent_tokens must be at least twice child_tokens")
        return self

    @property
    def min_tokens(self) -> int:
        """Chunks smaller than this are merged into a neighbour when possible."""
        return max(self.target_tokens // 4, 16)
