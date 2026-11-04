"""The ingestion pipeline: an ordered list of stages, each tied to a job state.

A stage receives a `StageContext`, does its work and either returns (success), raises
`PermanentError` (retrying cannot help, e.g. a corrupt file; the message is shown to the user)
or raises anything else (treated as transient and retried with backoff). Stages must be
idempotent because delivery is at-least-once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Collection, Document, JobState
from app.ingest.parsers import OcrEngine
from app.providers.registry import Providers
from app.storage import BlobStore


class PermanentError(Exception):
    """A failure that will not go away by retrying. `str(error)` is safe to show to the user."""


async def _no_progress(fraction: float) -> None:
    return None


@dataclass
class StageContext:
    session: AsyncSession
    settings: Settings
    blobs: BlobStore
    providers: Providers
    document: Document
    collection: Collection
    ocr: OcrEngine | None = None
    options: dict = field(default_factory=dict)  # per-job switches, e.g. {"reparse": False}
    # Report progress within the stage: 0.0 (just started) to 1.0 (finished).
    progress: Callable[[float], Awaitable[None]] = _no_progress


Stage = Callable[[StageContext], Awaitable[None]]
StageList = list[tuple[JobState, Stage]]


def default_stages() -> StageList:
    """Stages are appended here as the phases that implement them land."""
    from app.ingest.stages import chunk_stage, parse_stage  # stages import this module

    return [(JobState.PARSING, parse_stage), (JobState.CHUNKING, chunk_stage)]
