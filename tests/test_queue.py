import itertools
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import Collection, Document, IngestJob, JobState, User, Workspace
from app.ingest import queue
from app.ingest.queue import utcnow

LEASE = 60.0
MAX = 3


@pytest.fixture
def sm(app):
    return app.state.sessionmaker


@pytest.fixture
async def make_doc(sm):
    """Insert a document (with its parents) and return its id."""
    async with sm() as s:
        user = User(email=f"{uuid.uuid4().hex[:8]}@example.com", password_hash="x")
        s.add(user)
        await s.flush()
        ws = Workspace(name="w", slug=uuid.uuid4().hex[:12], owner_id=user.id)
        s.add(ws)
        await s.flush()
        col = Collection(workspace_id=ws.id, name="c")
        s.add(col)
        await s.commit()
        col_id = col.id

    async def _make() -> uuid.UUID:
        async with sm() as s:
            doc = Document(
                collection_id=col_id,
                title="t",
                source_type="text",
                content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
            )
            s.add(doc)
            await s.commit()
            return doc.id

    return _make


_tick = itertools.count()


async def _enqueue(sm, doc_id):
    async with sm() as s:
        job = await queue.enqueue(s, doc_id)
        # SQLite's now() has one-second resolution; space jobs out so "oldest first" is testable.
        job.created_at = utcnow() - timedelta(hours=1) + timedelta(milliseconds=next(_tick))
        await s.commit()
        return job.id


async def _job(sm, job_id) -> IngestJob:
    async with sm() as s:
        return (await s.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one()


async def _claim(sm, **kw):
    return await queue.claim(sm, lease_s=LEASE, max_attempts=MAX, **kw)


async def test_enqueue_creates_a_queued_job(sm, make_doc):
    job = await _job(sm, await _enqueue(sm, await make_doc()))
    assert job.state == JobState.QUEUED
    assert job.attempts == 0
    assert job.progress == 0


async def test_one_active_job_per_document(sm, make_doc):
    doc = await make_doc()
    first = await _enqueue(sm, doc)
    assert await _enqueue(sm, doc) == first


async def test_a_finished_job_allows_a_new_one(sm, make_doc):
    doc = await make_doc()
    first = await _enqueue(sm, doc)
    c = await _claim(sm)
    async with sm() as s:
        await queue.complete(s, c)
        await s.commit()
    assert await _enqueue(sm, doc) != first


async def test_claim_returns_none_when_empty(sm):
    assert await _claim(sm) is None


async def test_claim_takes_the_oldest_and_marks_it_running(sm, make_doc):
    d1, d2 = await make_doc(), await make_doc()
    j1 = await _enqueue(sm, d1)
    await _enqueue(sm, d2)
    c = await _claim(sm)
    assert (c.job_id, c.document_id, c.attempts) == (j1, d1, 1)
    job = await _job(sm, j1)
    assert job.state == JobState.PARSING
    assert job.attempts == 1
    assert job.started_at is not None
    assert job.locked_until is not None


async def test_a_claimed_job_cannot_be_claimed_again(sm, make_doc):
    await _enqueue(sm, await make_doc())
    assert await _claim(sm) is not None
    assert await _claim(sm) is None


async def test_each_job_is_claimed_exactly_once_in_order(sm, make_doc):
    ids = [await _enqueue(sm, await make_doc()) for _ in range(4)]
    got = [(await _claim(sm)).job_id for _ in range(4)]
    assert got == ids
    assert await _claim(sm) is None


async def test_jobs_wait_for_run_after(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    async with sm() as s:
        await queue.retry_later(s, c, "boom", delay_s=30)
        await s.commit()
    assert await _claim(sm) is None  # still backing off
    later = utcnow() + timedelta(seconds=31)
    again = await _claim(sm, now=later)
    assert again is not None
    assert again.attempts == 2


async def test_expired_lease_lets_another_worker_take_over(sm, make_doc):
    await _enqueue(sm, await make_doc())
    first = await _claim(sm)
    assert await _claim(sm) is None  # lease still valid
    later = utcnow() + timedelta(seconds=LEASE + 1)
    second = await _claim(sm, now=later)
    assert second is not None
    assert second.job_id == first.job_id
    assert second.attempts == 2


async def test_the_old_owner_loses_control_after_takeover(sm, make_doc):
    await _enqueue(sm, await make_doc())
    first = await _claim(sm)
    await _claim(sm, now=utcnow() + timedelta(seconds=LEASE + 1))  # someone else took it
    async with sm() as s:
        assert not await queue.advance(s, first, JobState.CHUNKING, 40, lease_s=LEASE)
        assert not await queue.complete(s, first)
        assert not await queue.fail(s, first, "late")
        assert not await queue.retry_later(s, first, "late", delay_s=1)
        await s.commit()


async def test_advance_updates_state_progress_and_lease(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    before = (await _job(sm, c.job_id)).locked_until
    async with sm() as s:
        assert await queue.advance(s, c, JobState.CHUNKING, 40, lease_s=LEASE * 2)
        await s.commit()
    job = await _job(sm, c.job_id)
    assert (job.state, job.progress) == (JobState.CHUNKING, 40)
    assert job.locked_until > before


async def test_advance_caps_progress_below_100(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    async with sm() as s:
        await queue.advance(s, c, JobState.INDEXING, 500, lease_s=LEASE)
        await s.commit()
    assert (await _job(sm, c.job_id)).progress == 99


async def test_complete_and_fail_set_terminal_state(sm, make_doc):
    await _enqueue(sm, await make_doc())
    await _enqueue(sm, await make_doc())
    a, b = await _claim(sm), await _claim(sm)
    async with sm() as s:
        assert await queue.complete(s, a)
        assert await queue.fail(s, b, "bad file")
        await s.commit()
    done, failed = await _job(sm, a.job_id), await _job(sm, b.job_id)
    assert (done.state, done.progress, done.finished_at is not None) == (JobState.DONE, 100, True)
    assert (failed.state, failed.error) == (JobState.FAILED, "bad file")
    assert done.locked_until is None and failed.locked_until is None


async def test_terminal_jobs_ignore_further_transitions(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    async with sm() as s:
        await queue.complete(s, c)
        await s.commit()
    async with sm() as s:
        assert not await queue.fail(s, c, "too late")
        assert not await queue.advance(s, c, JobState.PARSING, 1, lease_s=LEASE)
        await s.commit()
    assert (await _job(sm, c.job_id)).state == JobState.DONE


async def test_exhausted_jobs_are_reaped_not_retried(sm, make_doc):
    doc = await make_doc()
    await _enqueue(sm, doc)
    now = utcnow()
    for i in range(MAX):  # every attempt's worker dies without reporting
        now += timedelta(seconds=LEASE + 1)
        assert await _claim(sm, now=now) is not None, i
    now += timedelta(seconds=LEASE + 1)
    assert await _claim(sm, now=now) is None  # no attempts left
    assert await queue.reap_exhausted(sm, max_attempts=MAX, now=now) == [doc]
    assert await queue.reap_exhausted(sm, max_attempts=MAX, now=now) == []
    job = await _job_for_doc(sm, doc)
    assert job.state == JobState.FAILED
    assert "Gave up" in job.error


async def _job_for_doc(sm, doc_id):
    async with sm() as s:
        return (
            await s.execute(select(IngestJob).where(IngestJob.document_id == doc_id))
        ).scalar_one()


async def test_manual_retry_resets_a_failed_job(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    async with sm() as s:
        await queue.fail(s, c, "nope")
        await s.commit()
    async with sm() as s:
        assert await queue.reset_for_retry(s, c.job_id)
        await s.commit()
    job = await _job(sm, c.job_id)
    assert (job.state, job.attempts, job.error) == (JobState.QUEUED, 0, None)
    assert await _claim(sm) is not None


async def test_manual_retry_only_applies_to_failed_jobs(sm, make_doc):
    job_id = await _enqueue(sm, await make_doc())
    async with sm() as s:
        assert not await queue.reset_for_retry(s, job_id)
        await s.commit()


async def test_extend_lease(sm, make_doc):
    await _enqueue(sm, await make_doc())
    c = await _claim(sm)
    before = (await _job(sm, c.job_id)).locked_until
    async with sm() as s:
        assert await queue.extend_lease(s, c, lease_s=LEASE * 10)
        await s.commit()
    assert (await _job(sm, c.job_id)).locked_until > before


def test_backoff_doubles_and_is_capped():
    assert [queue.backoff_seconds(n, 2.0) for n in (1, 2, 3, 4)] == [2.0, 4.0, 8.0, 16.0]
    assert queue.backoff_seconds(40, 2.0) == 3600.0
    assert queue.backoff_seconds(0, 2.0) == 2.0
