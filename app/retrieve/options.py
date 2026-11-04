"""What a caller can ask of retrieval. These models are the request's option groups and the
pipeline's configuration at once, so the API and the library can never disagree."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.retrieve.transform import TransformOptions

Mode = Literal["dense", "sparse", "hybrid"]

MAX_TOP_K = 50
MIN_POOL = 10
MAX_POOL = 100


class FusionOptions(BaseModel):
    """How the dense and sparse rankings (and query variants) are merged (Phase 3.3)."""

    method: Literal["rrf", "weighted"] = "rrf"
    k: int = Field(60, ge=1, le=1000, description="RRF damping constant")
    variant_weight: float = Field(
        1.0, ge=0, le=10, description="Weight of the extra searches from query transforms"
    )
    dense_weight: float = Field(1.0, ge=0, le=10)
    sparse_weight: float = Field(1.0, ge=0, le=10)

    @model_validator(mode="after")
    def _some_weight(self) -> FusionOptions:
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one of dense_weight and sparse_weight must be above 0")
        return self

    def weights(self) -> dict[str, float]:
        return {"dense": self.dense_weight, "sparse": self.sparse_weight}


class RerankOptions(BaseModel):
    """Re-score the best fused candidates with a cross-encoder (Phase 3.4)."""

    enabled: bool = False
    candidates: int = Field(
        50, ge=1, le=MAX_POOL, description="How many fused candidates the reranker reads"
    )


class RetrievalOptions(BaseModel):
    mode: Mode = "hybrid"
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)
    pool: int = Field(
        50, ge=MIN_POOL, le=MAX_POOL, description="Candidates taken from each channel"
    )
    min_score: float | None = Field(
        None, ge=-1, le=1, description="Dense mode only: drop passages less similar than this"
    )
    fusion: FusionOptions = Field(default_factory=FusionOptions)
    rerank: RerankOptions = Field(default_factory=RerankOptions)
    transforms: TransformOptions = Field(default_factory=TransformOptions)

    @model_validator(mode="after")
    def _consistent(self) -> RetrievalOptions:
        if self.min_score is not None and self.mode != "dense":
            raise ValueError(
                "min_score is a cosine similarity, so it only applies when mode is 'dense'"
            )
        return self
