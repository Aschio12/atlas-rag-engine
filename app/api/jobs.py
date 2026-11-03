"""Ingestion jobs: status, listing, manual retry and a live event stream (Phase 2.2)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import SessionDep
from app.core.errors import Conflict, NotFound
from app.db.models import Collection, Document, DocumentStatus, IngestJob, JobState
from app.ingest import queue

router = APIRouter(prefix="/workspaces/{workspace_id}/jobs", tags=["jobs"])

HEARTBEAT_S = 15.0  # SSE comment line so proxies do not close an idle stream
MAX_STREAM_S = 15 * 60  # clients reconnect; keeps abandoned streams from living forever


class JobOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    state: JobState
    progress: int
    attempts: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobList(BaseModel):
    items: list[JobOut]
    total: int


def _scoped(workspace_id: uuid.UUID):
    """Jobs of one workspace, with their document's title and status."""
    return (
        select(IngestJob, Document.title, Document.status)
        .join(Document, Document.id == IngestJob.document_id)
        .join(Collection, Collection.id == Document.collection_id)
        .where(Collection.workspace_id == workspace_id)
    )


def _to_out(job: IngestJob, title: str) -> JobOut:
    return JobOut(
        id=job.id,
        document_id=job.document_id,
        document_title=title,
        state=job.state,
        progress=job.progress,
        attempts=job.attempts,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


async def _get(session: AsyncSession, workspace_id: uuid.UUID, job_id: uuid.UUID):
    row = (await session.execute(_scoped(workspace_id).where(IngestJob.id == job_id))).first()
    if row is None:
        raise NotFound("Job not found")
    return row


@router.get("", response_model=JobList)
async def list_jobs(
    access: ViewerAccess,
    session: SessionDep,
    state: JobState | None = None,
    document_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    filters = []
    if state is not None:
        filters.append(IngestJob.state == state)
    if document_id is not None:
        filters.append(IngestJob.document_id == document_id)
    base = _scoped(access.workspace.id).where(*filters)
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(
            base.order_by(IngestJob.created_at.desc(), IngestJob.id).limit(limit).offset(offset)
        )
    ).all()
    return JobList(items=[_to_out(j, title) for j, title, _ in rows], total=total)


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, access: ViewerAccess, session: SessionDep):
    job, title, _ = await _get(session, access.workspace.id, job_id)
    return _to_out(job, title)


@router.post("/{job_id}/retry", response_model=JobOut)
async def retry_job(job_id: uuid.UUID, access: MemberAccess, request: Request, session: SessionDep):
    """Re-run a failed job with a fresh attempt budget."""
    job, title, _ = await _get(session, access.workspace.id, job_id)
    if job.state != JobState.FAILED:
        raise Conflict(f"Only failed jobs can be retried (this one is {job.state.value})")
    if not await queue.reset_for_retry(session, job.id):
        raise Conflict("The job changed state; reload and try again")
    await session.execute(
        update(Document).where(Document.id == job.document_id).values(status=DocumentStatus.PENDING)
    )
    await session.commit()
    request.app.state.worker.wake()
    job, title, _ = await _get(session, access.workspace.id, job_id)
    return _to_out(job, title)


def _snapshot(job: IngestJob, document_status: DocumentStatus) -> dict:
    return {
        "id": str(job.id),
        "document_id": str(job.document_id),
        "state": job.state.value,
        "progress": job.progress,
        "attempts": job.attempts,
        "error": job.error,
        "document_status": document_status.value,
    }


async def _read(
    sessionmaker: async_sessionmaker[AsyncSession], workspace_id: uuid.UUID, job_id: uuid.UUID
) -> dict | None:
    async with sessionmaker() as session:
        row = (await session.execute(_scoped(workspace_id).where(IngestJob.id == job_id))).first()
    return None if row is None else _snapshot(row[0], row[2])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.get("/{job_id}/events", response_class=StreamingResponse)
async def job_events(job_id: uuid.UUID, access: ViewerAccess, request: Request):
    """Server-sent events: one `job` event now and another whenever state, progress, attempts or
    error change, ending after `done`/`failed`. Browsers' `EventSource` cannot send an
    Authorization header, so use `fetch()` streaming (the Studio does) or any SSE client that can.
    """
    sessionmaker = request.app.state.sessionmaker
    poll = request.app.state.settings.sse_poll_s
    workspace_id = access.workspace.id

    first = await _read(sessionmaker, workspace_id, job_id)
    if first is None:
        raise NotFound("Job not found")

    async def stream() -> AsyncIterator[str]:
        snapshot, last, quiet = first, None, 0.0
        started = time.monotonic()
        while True:
            if snapshot is None:
                yield _sse("gone", {"id": str(job_id)})  # the document was deleted meanwhile
                return
            if snapshot != last:
                yield _sse("job", snapshot)
                last, quiet = snapshot, 0.0
                if snapshot["state"] in (JobState.DONE.value, JobState.FAILED.value):
                    return
            if time.monotonic() - started > MAX_STREAM_S:
                yield _sse("timeout", {"id": str(job_id)})
                return
            await asyncio.sleep(poll)
            quiet += poll
            if quiet >= HEARTBEAT_S:
                yield ": ping\n\n"
                quiet = 0.0
            snapshot = await _read(sessionmaker, workspace_id, job_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
