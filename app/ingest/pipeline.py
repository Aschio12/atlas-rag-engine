"""The ingestion pipeline: an ordered list of stages, each tied to a job state.

A stage receives a `StageContext`, does its work and either returns (success), raises
`PermanentError` (retrying cannot help, e.g. a corrupt file; the message is shown to the user)
or raises anything else (treated as transient and retried with backoff). Stages must be
idempotent because delivery is at-least-once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Collection, Document, JobState
from app.providers.registry import Providers
from app.storage import BlobStore


class PermanentError(Exception):
    """A failure that will not go away by retrying. `str(error)` is safe to show to the user."""


@dataclass
class StageContext:
    session: AsyncSession
    settings: Settings
    blobs: BlobStore
    providers: Providers
    document: Document
    collection: Collection


Stage = Callable[[StageContext], Awaitable[None]]
StageList = list[tuple[JobState, Stage]]


def default_stages() -> StageList:
    """Stages are appended here as the phases that implement them land."""
    return []
