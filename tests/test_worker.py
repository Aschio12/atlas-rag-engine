import asyncio
import io
import uuid

import pytest
from sqlalchemy import delete, select

from app.db.models import Document, DocumentStatus, IngestJob, JobState
from app.ingest.pipeline import PermanentError
from app.ingest.worker import Worker


def _file(data=b"Refunds are issued within 14 days."):
    return {"file": ("notes.txt", io.BytesIO(data), "text/plain")}


@pytest.fixture
async def uploaded(client, space):
    r = await client.post(space["url"], files=_file(), headers=space["headers"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job"]["state"] == "queued"
    return body


@pytest.fixture
def make_worker(app, settings):
    def _make(stages=None, **kw):
        return Worker(
            app.state.sessionmaker, settings, app.state.blobs, app.state.providers, stages, **kw
        )

    return _make


async def _state(app, doc_id):
    doc_id = uuid.UUID(str(doc_id))
    async with app.state.sessionmaker() as s:
        doc = (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
        job = (
            await s.execute(select(IngestJob).where(IngestJob.document_id == doc_id))
        ).scalar_one_or_none()
        return doc, job


async def test_with_no_stages_the_job_completes_and_the_document_is_ready(
    app, uploaded, make_worker
):
    assert await make_worker().run_once() is True
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, job.progress) == (JobState.DONE, 100)
    assert doc.status == DocumentStatus.READY
    assert await make_worker().run_once() is False  # nothing left


async def test_stages_run_in_order_and_report_progress(app, uploaded, make_worker):
    seen = []

    def stage(name):
        async def run(ctx):
            _, job = await _state(app, ctx.document.id)
            seen.append((name, job.state, job.progress, ctx.document.status))

        return run

    worker = make_worker(
        [
            (JobState.PARSING, stage("a")),
            (JobState.CHUNKING, stage("b")),
            (JobState.INDEXING, stage("c")),
        ]
    )
    await worker.run_once()
    assert seen == [
        ("a", JobState.PARSING, 0, DocumentStatus.PROCESSING),
        ("b", JobState.CHUNKING, 33, DocumentStatus.PROCESSING),
        ("c", JobState.INDEXING, 66, DocumentStatus.PROCESSING),
    ]
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, doc.status) == (JobState.DONE, DocumentStatus.READY)


async def test_stage_context_exposes_document_and_collection(app, uploaded, space, make_worker):
    got = {}

    async def stage(ctx):
        got["doc"] = str(ctx.document.id)
        got["col"] = str(ctx.collection.id)

    await make_worker([(JobState.PARSING, stage)]).run_once()
    assert got == {"doc": uploaded["id"], "col": space["col"]}


async def test_permanent_errors_fail_immediately_with_the_message(app, uploaded, make_worker):
    async def stage(ctx):
        raise PermanentError("This PDF is password protected")

    await make_worker([(JobState.PARSING, stage)]).run_once()
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, job.error, job.attempts) == (
        JobState.FAILED,
        "This PDF is password protected",
        1,
    )
    assert doc.status == DocumentStatus.FAILED


async def test_transient_errors_are_retried_with_backoff(app, settings, uploaded, make_worker):
    calls = []

    async def flaky(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("db hiccup")

    worker = make_worker([(JobState.PARSING, flaky)])
    await worker.run_once()
    doc, job = await _state(app, uploaded["id"])
    assert job.state == JobState.QUEUED
    assert job.run_after is not None
    assert doc.status == DocumentStatus.PENDING  # waiting for the retry
    assert await worker.run_once() is False  # backing off (2 s default)

    settings.job_backoff_s = 0  # next retry would be immediate; make the current one due now
    async with app.state.sessionmaker() as s:
        await s.execute(IngestJob.__table__.update().values(run_after=None))
        await s.commit()
    await worker.run_once()
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, job.attempts, job.error) == (JobState.DONE, 2, None)
    assert doc.status == DocumentStatus.READY
    assert len(calls) == 2


async def test_internal_error_details_are_never_stored(app, settings, uploaded, make_worker):
    settings.job_max_attempts = 1

    async def leaky(ctx):
        raise RuntimeError("postgres://admin:hunter2@db/prod unreachable")

    await make_worker([(JobState.PARSING, leaky)]).run_once()
    _, job = await _state(app, uploaded["id"])
    assert job.error == "Unexpected error (RuntimeError)"
    assert "hunter2" not in job.error


async def test_gives_up_after_max_attempts(app, settings, uploaded, make_worker):
    settings.job_max_attempts = 2
    settings.job_backoff_s = 0

    async def broken(ctx):
        raise RuntimeError("always")

    worker = make_worker([(JobState.PARSING, broken)])
    await worker.run_once()
    await worker.run_once()
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, job.attempts) == (JobState.FAILED, 2)
    assert doc.status == DocumentStatus.FAILED
    assert await worker.run_once() is False


async def test_a_document_deleted_while_queued_is_skipped(
    client, app, space, uploaded, make_worker
):
    r = await client.delete(f"{space['url']}/{uploaded['id']}", headers=space["headers"])
    assert r.status_code == 204
    assert await make_worker().run_once() is False  # the job went with the document


async def test_a_document_deleted_mid_job_does_not_crash_the_worker(app, uploaded, make_worker):
    async def delete_self(ctx):
        await ctx.session.execute(delete(Document).where(Document.id == ctx.document.id))
        await ctx.session.commit()

    worker = make_worker([(JobState.PARSING, delete_self), (JobState.CHUNKING, delete_self)])
    assert await worker.run_once() is True
    doc, job = await _state(app, uploaded["id"])
    assert doc is None and job is None


async def test_dead_workers_are_recovered_through_lease_expiry(
    app, settings, uploaded, make_worker
):
    """A job stuck 'running' with an expired lease is picked up again."""
    from datetime import timedelta

    from app.ingest.queue import utcnow

    # Simulate a crash: claimed, never finished, lease in the past.
    async with app.state.sessionmaker() as s:
        await s.execute(
            IngestJob.__table__.update().values(
                state=JobState.CHUNKING, attempts=1, locked_until=utcnow() - timedelta(seconds=5)
            )
        )
        await s.commit()
    assert await make_worker().run_once() is True
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, job.attempts, doc.status) == (JobState.DONE, 2, DocumentStatus.READY)


async def test_exhausted_dead_jobs_mark_the_document_failed(app, settings, uploaded, make_worker):
    from datetime import timedelta

    from app.ingest.queue import utcnow

    async with app.state.sessionmaker() as s:
        await s.execute(
            IngestJob.__table__.update().values(
                state=JobState.EMBEDDING,
                attempts=settings.job_max_attempts,
                locked_until=utcnow() - timedelta(seconds=5),
            )
        )
        await s.commit()
    assert await make_worker().run_once() is False
    doc, job = await _state(app, uploaded["id"])
    assert (job.state, doc.status) == (JobState.FAILED, DocumentStatus.FAILED)


async def test_background_loop_processes_uploads_and_stops_cleanly(
    client, app, settings, space, make_worker
):
    settings.worker_poll_s = 0.02
    worker = make_worker()
    worker.start()
    try:
        r = await client.post(space["url"], files=_file(b"loop test"), headers=space["headers"])
        worker.wake()
        doc_id = r.json()["id"]
        for _ in range(200):
            doc, _job = await _state(app, doc_id)
            if doc.status == DocumentStatus.READY:
                break
            await asyncio.sleep(0.02)
        assert doc.status == DocumentStatus.READY
    finally:
        await worker.stop()
    assert worker._task is None


async def test_loop_survives_unexpected_errors(app, settings, uploaded, make_worker, monkeypatch):
    settings.worker_poll_s = 0.01
    worker = make_worker()
    real = worker.run_once
    calls = {"n": 0}

    async def flaky_run_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("database restarting")
        return await real()

    monkeypatch.setattr(worker, "run_once", flaky_run_once)
    worker.start()
    try:
        for _ in range(300):
            doc, _ = await _state(app, uploaded["id"])
            if doc.status == DocumentStatus.READY:
                break
            await asyncio.sleep(0.02)
        assert doc.status == DocumentStatus.READY
    finally:
        await worker.stop()
