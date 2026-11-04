"""The ingestion worker: claims jobs from the queue and runs them through the pipeline.

It runs inside the API process by default (one asyncio task, one job at a time, which keeps
memory predictable on small machines). Because the queue is durable and lease-based, the same
class can be run as a separate process, or many of them, without code changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Collection, Document, DocumentStatus
from app.ingest import queue
from app.ingest.parsers import OcrEngine, default_ocr
from app.ingest.pipeline import PermanentError, StageContext, StageList, default_stages
from app.providers.errors import ProviderError
from app.providers.registry import Providers
from app.storage import BlobStore

log = logging.getLogger("atlas.worker")

LEASE_S = 120.0  # a job is presumed lost if its worker is silent this long


class Worker:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        settings: Settings,
        blobs: BlobStore,
        providers: Providers,
        stages: StageList | None = None,
        lease_s: float = LEASE_S,
        ocr: OcrEngine | None = None,
    ):
        self.sessionmaker = sessionmaker
        self.settings = settings
        self.blobs = blobs
        self.providers = providers
        self.stages = default_stages() if stages is None else stages
        self.lease_s = lease_s
        self.ocr = ocr if ocr is not None else self._detect_ocr()
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._stopping = False

    def _detect_ocr(self) -> OcrEngine | None:
        if not self.settings.ocr_enabled:
            return None
        engine = default_ocr(self.settings.ocr_language)
        log.info("OCR for scanned PDFs: %s", "Tesseract" if engine else "not available")
        return engine

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._loop(), name="atlas-ingest-worker")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def wake(self) -> None:
        """Tell the worker there is new work so it does not wait for the next poll."""
        self._wake.set()

    async def _loop(self) -> None:
        log.info("ingestion worker started")
        while not self._stopping:
            try:
                if await self.run_once():
                    continue  # more work may be waiting
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker loop error; backing off")
                await asyncio.sleep(self.settings.worker_poll_s * 5)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.settings.worker_poll_s)
            self._wake.clear()

    # --- one unit of work --------------------------------------------------------------------

    async def run_once(self) -> bool:
        """Reap hopeless jobs, then claim and run one job. True if a job was processed."""
        for doc_id in await queue.reap_exhausted(
            self.sessionmaker, max_attempts=self.settings.job_max_attempts
        ):
            await self._set_document(doc_id, DocumentStatus.FAILED)
        claim = await queue.claim(
            self.sessionmaker,
            lease_s=self.lease_s,
            max_attempts=self.settings.job_max_attempts,
        )
        if claim is None:
            return False
        await self._process(claim)
        return True

    async def _process(self, claim: queue.Claim) -> None:
        started = time.monotonic()
        log.info(
            "job started",
            extra={"job_id": str(claim.job_id), "attempt": claim.attempts},
        )
        heartbeat = asyncio.create_task(self._heartbeat(claim))
        try:
            await self._set_document(claim.document_id, DocumentStatus.PROCESSING)
            if await self._run_stages(claim):
                async with self.sessionmaker() as s:
                    done = await queue.complete(s, claim)
                    if done:
                        await self._set_document(claim.document_id, DocumentStatus.READY, session=s)
                    await s.commit()
                if done:
                    log.info(
                        "job done",
                        extra={
                            "job_id": str(claim.job_id),
                            "duration_ms": round((time.monotonic() - started) * 1000),
                        },
                    )
        except asyncio.CancelledError:
            raise  # shutdown: the lease will expire and the job is retried later
        except PermanentError as e:
            await self._fail(claim, str(e), retry=False)
        except ProviderError as e:
            # A model or vector-store outage: transient. Name the provider, never its response.
            log.warning("job hit a provider error", extra={"job_id": str(claim.job_id)})
            message = f"The '{e.provider}' provider failed; will retry"
            await self._fail(claim, message, retry=True)
        except Exception as e:
            log.exception("job failed", extra={"job_id": str(claim.job_id)})
            await self._fail(claim, f"Unexpected error ({type(e).__name__})", retry=True)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _run_stages(self, claim: queue.Claim) -> bool:
        """Run every stage. Returns False if the job or its document vanished or was taken over."""
        total = max(len(self.stages), 1)
        for index, (state, stage) in enumerate(self.stages):
            async with self.sessionmaker() as s:
                moved = await queue.advance(
                    s, claim, state, 100 * index // total, lease_s=self.lease_s
                )
                await s.commit()
            if not moved:
                log.warning("job no longer ours; stopping", extra={"job_id": str(claim.job_id)})
                return False
            async with self.sessionmaker() as s:
                document = await s.get(Document, claim.document_id)
                if document is None:
                    return False  # deleted while queued/running
                collection = await s.get(Collection, document.collection_id)
                low, high = 100 * index // total, 100 * (index + 1) // total

                async def report(fraction: float, low=low, high=high) -> None:
                    value = low + int((high - low) * max(0.0, min(fraction, 1.0)))
                    async with self.sessionmaker() as ps:
                        await queue.set_progress(ps, claim, value)
                        await ps.commit()

                await stage(
                    StageContext(
                        session=s,
                        settings=self.settings,
                        blobs=self.blobs,
                        providers=self.providers,
                        document=document,
                        collection=collection,
                        ocr=self.ocr,
                        options=claim.options,
                        progress=report,
                    )
                )
                await s.commit()
        return True

    async def _heartbeat(self, claim: queue.Claim) -> None:
        """Keep the lease alive while a long stage runs."""
        while True:
            await asyncio.sleep(self.lease_s / 3)
            try:
                async with self.sessionmaker() as s:
                    await queue.extend_lease(s, claim, lease_s=self.lease_s)
                    await s.commit()
            except Exception:
                log.warning("could not extend job lease", exc_info=True)

    async def _fail(self, claim: queue.Claim, message: str, *, retry: bool) -> None:
        retries_left = retry and claim.attempts < self.settings.job_max_attempts
        async with self.sessionmaker() as s:
            if retries_left:
                delay = queue.backoff_seconds(claim.attempts, self.settings.job_backoff_s)
                changed = await queue.retry_later(s, claim, message, delay_s=delay)
                status = DocumentStatus.PENDING  # waiting for its next attempt
            else:
                changed = await queue.fail(s, claim, message)
                status = DocumentStatus.FAILED
            if changed:
                await self._set_document(claim.document_id, status, session=s)
            await s.commit()
        log.warning(
            "job %s",
            "will retry" if retries_left else "failed",
            extra={"job_id": str(claim.job_id), "attempt": claim.attempts},
        )

    async def _set_document(
        self, document_id: uuid.UUID, status: DocumentStatus, *, session: AsyncSession | None = None
    ) -> None:
        stmt = update(Document).where(Document.id == document_id).values(status=status)
        if session is not None:
            await session.execute(stmt)
            return
        async with self.sessionmaker() as s:
            await s.execute(stmt)
            await s.commit()
