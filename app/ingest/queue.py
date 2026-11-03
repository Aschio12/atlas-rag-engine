"""A small durable job queue on top of the database (Phase 2.2).

Why not Redis/Celery? Ingestion volume is modest, the data is already in Postgres, and fewer
moving parts means a laptop with a few GB of RAM can run the whole stack. The queue is
designed so it could be swapped later: callers only use `enqueue`, `claim`, `advance`,
`complete`, `fail` and `requeue`.

Guarantees
* **At-least-once.** A claimed job holds a lease (`locked_until`). If the worker dies, the lease
  expires and another claim picks the job up again; stages are therefore written to be idempotent.
* **Safe with several workers.** `claim` is an optimistic compare-and-set UPDATE, so two workers
  can never both win the same job, on SQLite or Postgres.
* **Bounded retries with exponential backoff** (`run_after`), then a terminal `failed` state.
* **Transitions never resurrect deleted data.** They are plain UPDATEs; if the document (and thus
  the job) was deleted meanwhile they affect zero rows and the caller stops.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import IngestJob, JobState

RUNNING_STATES = (JobState.PARSING, JobState.CHUNKING, JobState.EMBEDDING, JobState.INDEXING)
ACTIVE_STATES = (JobState.QUEUED, *RUNNING_STATES)
TERMINAL_STATES = (JobState.DONE, JobState.FAILED)


def utcnow() -> datetime:
    return datetime.now(UTC)


def backoff_seconds(attempts: int, base: float) -> float:
    """Delay before retry number `attempts` (1-based): base, 2*base, 4*base ... capped at 1 h."""
    return min(base * (2 ** max(attempts - 1, 0)), 3600.0)


@dataclass(frozen=True)
class Claim:
    job_id: uuid.UUID
    document_id: uuid.UUID
    attempts: int  # including this one


async def enqueue(session: AsyncSession, document_id: uuid.UUID) -> IngestJob:
    """Add a job for a document. The caller commits, so the document and its job appear together.

    A document never has two active jobs: if one is already queued or running it is returned.
    """
    existing = (
        await session.execute(
            select(IngestJob).where(
                IngestJob.document_id == document_id, IngestJob.state.in_(ACTIVE_STATES)
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    job = IngestJob(id=uuid.uuid4(), document_id=document_id, state=JobState.QUEUED)
    session.add(job)
    return job


def _mine(c: Claim):
    # Matching on attempts too means a worker whose lease expired (and whose job was re-claimed)
    # can no longer change the job.
    return and_(IngestJob.id == c.job_id, IngestJob.attempts == c.attempts)


async def claim(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    lease_s: float,
    max_attempts: int,
    now: datetime | None = None,
) -> Claim | None:
    """Take the oldest runnable job, or return None.

    Runnable means queued and past its `run_after`, or running with an expired lease (its worker
    died) and attempts left. Claiming moves the job to PARSING and starts a new lease.
    """
    now = now or utcnow()
    runnable = or_(
        and_(
            IngestJob.state == JobState.QUEUED,
            or_(IngestJob.run_after.is_(None), IngestJob.run_after <= now),
        ),
        and_(
            IngestJob.state.in_(RUNNING_STATES),
            IngestJob.locked_until < now,
            IngestJob.attempts < max_attempts,
        ),
    )
    async with sessionmaker() as session:
        for _ in range(5):  # a lost race just moves on to the next candidate
            row = (
                await session.execute(
                    select(IngestJob.id, IngestJob.document_id, IngestJob.state, IngestJob.attempts)
                    .where(runnable)
                    .order_by(IngestJob.created_at, IngestJob.id)
                    .limit(1)
                )
            ).first()
            if row is None:
                return None
            won = await session.execute(
                update(IngestJob)
                .where(
                    IngestJob.id == row.id,
                    IngestJob.state == row.state,
                    IngestJob.attempts == row.attempts,
                )
                .values(
                    state=JobState.PARSING,
                    progress=0,
                    attempts=row.attempts + 1,
                    locked_until=now + timedelta(seconds=lease_s),
                    run_after=None,
                    error=None,
                    started_at=now,
                    finished_at=None,
                )
            )
            await session.commit()
            if won.rowcount:
                return Claim(row.id, row.document_id, row.attempts + 1)
    return None


async def reap_exhausted(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    max_attempts: int,
    now: datetime | None = None,
) -> list[uuid.UUID]:
    """Fail running jobs whose lease expired after their last attempt (they keep killing their
    worker). Returns the affected document ids so the caller can mark those documents failed."""
    now = now or utcnow()
    async with sessionmaker() as session:
        rows = (
            await session.execute(
                select(IngestJob.id, IngestJob.document_id, IngestJob.attempts).where(
                    IngestJob.state.in_(RUNNING_STATES),
                    IngestJob.locked_until < now,
                    IngestJob.attempts >= max_attempts,
                )
            )
        ).all()
        reaped: list[uuid.UUID] = []
        for row in rows:
            res = await session.execute(
                update(IngestJob)
                .where(
                    IngestJob.id == row.id,
                    IngestJob.attempts == row.attempts,
                    IngestJob.state.in_(RUNNING_STATES),
                )
                .values(
                    state=JobState.FAILED,
                    error=f"Gave up after {row.attempts} attempts: the worker stopped mid-job",
                    finished_at=now,
                    locked_until=None,
                )
            )
            if res.rowcount:
                reaped.append(row.document_id)
        await session.commit()
    return reaped


async def advance(
    session: AsyncSession, c: Claim, state: JobState, progress: int, *, lease_s: float
) -> bool:
    """Move a running job to its next stage and extend the lease. False means the job is gone
    or was taken over: stop working on it. The caller commits."""
    res = await session.execute(
        update(IngestJob)
        .where(_mine(c), IngestJob.state.in_(RUNNING_STATES))
        .values(
            state=state,
            progress=max(0, min(progress, 99)),
            locked_until=utcnow() + timedelta(seconds=lease_s),
        )
    )
    return bool(res.rowcount)


async def extend_lease(session: AsyncSession, c: Claim, *, lease_s: float) -> bool:
    res = await session.execute(
        update(IngestJob)
        .where(_mine(c), IngestJob.state.in_(RUNNING_STATES))
        .values(locked_until=utcnow() + timedelta(seconds=lease_s))
    )
    return bool(res.rowcount)


async def complete(session: AsyncSession, c: Claim) -> bool:
    res = await session.execute(
        update(IngestJob)
        .where(_mine(c), IngestJob.state.in_(RUNNING_STATES))
        .values(state=JobState.DONE, progress=100, finished_at=utcnow(), locked_until=None)
    )
    return bool(res.rowcount)


async def fail(session: AsyncSession, c: Claim, error: str) -> bool:
    res = await session.execute(
        update(IngestJob)
        .where(_mine(c), IngestJob.state.in_(RUNNING_STATES))
        .values(state=JobState.FAILED, error=error, finished_at=utcnow(), locked_until=None)
    )
    return bool(res.rowcount)


async def retry_later(session: AsyncSession, c: Claim, error: str, *, delay_s: float) -> bool:
    """Put a failed attempt back in the queue, not to run before `delay_s` from now."""
    res = await session.execute(
        update(IngestJob)
        .where(_mine(c), IngestJob.state.in_(RUNNING_STATES))
        .values(
            state=JobState.QUEUED,
            error=error,
            run_after=utcnow() + timedelta(seconds=delay_s),
            locked_until=None,
        )
    )
    return bool(res.rowcount)


async def reset_for_retry(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Manual retry of a failed job: fresh attempt budget, runnable immediately."""
    res = await session.execute(
        update(IngestJob)
        .where(IngestJob.id == job_id, IngestJob.state == JobState.FAILED)
        .values(
            state=JobState.QUEUED,
            progress=0,
            attempts=0,
            error=None,
            run_after=None,
            locked_until=None,
            started_at=None,
            finished_at=None,
        )
    )
    return bool(res.rowcount)
