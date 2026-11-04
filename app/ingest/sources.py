"""Running source syncs (Phase 2.7).

A sync discovers a source's items through its connector and adds, updates or removes the
matching documents; each added or changed document then flows through the normal ingestion
pipeline (parse, chunk, enrich, embed, index) via its own durable job.

The sync itself runs as an in-process task: its state lives in the `sources` row (status,
counters, errors), so progress is visible and a restart simply marks an interrupted sync failed
(`recover`) for the user to run again; nothing is half-applied because every item is committed on
its own and the pipeline jobs are durable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from app.core.errors import Conflict, UnsupportedMedia
from app.db.models import Collection, Document, Source, SourceStatus, SourceType
from app.ingest.connectors.base import Connector, ItemFailure, ItemUnchanged, SourceItem
from app.ingest.connectors.github import GithubConnector, GithubSourceConfig
from app.ingest.connectors.web import SyncAborted, WebConnector, WebSourceConfig
from app.ingest.purge import delete_documents
from app.ingest.service import create_document, find_by_source_uri, replace_content

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger("atlas.sources")

MAX_RECORDED_ERRORS = 10
SAVE_EVERY = 10  # persist counters after this many items so progress is visible


def new_stats() -> dict[str, Any]:
    return {
        "discovered": 0, "created": 0, "updated": 0, "unchanged": 0, "skipped": 0,
        "failed": 0, "removed": 0, "errors": [], "warnings": [],
    }  # fmt: skip


def now() -> datetime:
    return datetime.now(UTC)


def build_connector(app: FastAPI, source: Source, known: dict[str, str]) -> Connector:
    settings = app.state.settings
    if source.type == SourceType.WEB:

        async def fetch(url: str, *, max_bytes: int):
            return await app.state.fetch_url(
                url,
                max_bytes=max_bytes,
                timeout_s=settings.url_fetch_timeout_s,
                allow_private=settings.allow_private_urls,
            )

        return WebConnector(
            WebSourceConfig(**source.config), fetch, max_bytes=settings.max_upload_bytes
        )
    return GithubConnector(
        GithubSourceConfig(**source.config),
        api_url=settings.github_api_url,
        raw_url=settings.github_raw_url,
        token=settings.github_token.get_secret_value(),
        known=known,
        client=getattr(app.state, "github_client", None),
    )


class SyncRunner:
    def __init__(self, app: FastAPI):
        self.app = app
        self._tasks: dict[uuid.UUID, asyncio.Task] = {}
        self._slots = asyncio.Semaphore(app.state.settings.source_sync_concurrency)

    # --- control ----------------------------------------------------------------------------

    def is_running(self, source_id: uuid.UUID) -> bool:
        task = self._tasks.get(source_id)
        return task is not None and not task.done()

    def start(self, source_id: uuid.UUID) -> None:
        if self.is_running(source_id):
            raise RuntimeError("a sync is already running for this source")
        self._tasks[source_id] = asyncio.create_task(self._run(source_id), name=f"sync-{source_id}")

    async def wait(self, source_id: uuid.UUID | None = None) -> None:
        """Wait for one (or all) running syncs: used by tests and graceful shutdown."""
        tasks = [self._tasks[source_id]] if source_id else list(self._tasks.values())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel(self, source_id: uuid.UUID) -> None:
        task = self._tasks.get(source_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def shutdown(self) -> None:
        for source_id in list(self._tasks):
            await self.cancel(source_id)

    async def recover(self) -> int:
        """At startup: syncs that were running when the process died cannot still be running."""
        async with self.app.state.sessionmaker() as s:
            res = await s.execute(
                update(Source)
                .where(Source.status == SourceStatus.RUNNING)
                .values(
                    status=SourceStatus.FAILED,
                    error="Interrupted by a server restart; run the sync again",
                    finished_at=now(),
                )
            )
            await s.commit()
        return res.rowcount or 0

    # --- the sync ---------------------------------------------------------------------------

    async def _run(self, source_id: uuid.UUID) -> None:
        async with self._slots:
            try:
                await self._sync(source_id)
            except asyncio.CancelledError:
                await self._finish(source_id, SourceStatus.FAILED, error="The sync was cancelled")
                raise
            except SyncAborted as e:
                await self._finish(source_id, SourceStatus.FAILED, error=str(e))
            except Exception as e:
                log.exception("source sync failed", extra={"source_id": str(source_id)})
                await self._finish(
                    source_id, SourceStatus.FAILED, error=f"Unexpected error ({type(e).__name__})"
                )

    async def _finish(self, source_id: uuid.UUID, status: SourceStatus, **values: Any) -> None:
        async with self.app.state.sessionmaker() as s:
            await s.execute(
                update(Source)
                .where(Source.id == source_id)
                .values(status=status, finished_at=now(), **values)
            )
            await s.commit()

    async def _save_stats(self, source_id: uuid.UUID, stats: dict[str, Any]) -> None:
        async with self.app.state.sessionmaker() as s:
            await s.execute(update(Source).where(Source.id == source_id).values(stats=dict(stats)))
            await s.commit()

    async def _sync(self, source_id: uuid.UUID) -> None:
        app = self.app
        sm = app.state.sessionmaker
        stats = new_stats()
        async with sm() as s:
            source = await s.get(Source, source_id)
            collection = await s.get(Collection, source.collection_id) if source else None
            if source is None or collection is None:
                return  # deleted while queued
            source.status, source.error = SourceStatus.RUNNING, None
            source.started_at, source.finished_at, source.stats = now(), None, dict(stats)
            await s.commit()
            known = await self._known_hashes(s, source.id)
            config = dict(source.config)
            prune = bool(config.get("prune", True))
            connector = build_connector(app, source, known)
            collection_id = collection.id

        seen = 0
        try:
            async for item in connector.items():
                await self._apply(app, collection_id, source_id, item, stats)
                seen += 1
                if seen % SAVE_EVERY == 0:
                    await self._save_stats(source_id, stats)
                    app.state.worker.wake()
        finally:
            aclose = getattr(connector, "aclose", None)
            if aclose:
                await aclose()

        plan = connector.plan
        stats["discovered"] = plan.discovered
        stats["warnings"] = plan.warnings[:MAX_RECORDED_ERRORS]
        if prune and plan.complete and stats["failed"] == 0:
            stats["removed"] = await self._prune(app, collection_id, source_id, plan.uris)
        elif prune and not plan.complete:
            stats["warnings"].append("Not pruning deleted items: the listing was incomplete")

        handled = stats["created"] + stats["updated"] + stats["unchanged"] + stats["skipped"]
        async with sm() as s:
            if stats["failed"] and not handled:
                status, error = SourceStatus.FAILED, f"All {stats['failed']} items failed"
            else:
                status, error = SourceStatus.DONE, None
            await s.execute(
                update(Source)
                .where(Source.id == source_id)
                .values(
                    status=status,
                    error=error,
                    stats=stats,
                    finished_at=now(),
                    last_synced_at=now() if status == SourceStatus.DONE else Source.last_synced_at,
                )
            )
            await s.commit()
        app.state.worker.wake()
        log.info(
            "source synced",
            extra={
                "source_id": str(source_id),
                **{k: v for k, v in stats.items() if isinstance(v, int)},
            },
        )

    @staticmethod
    async def _known_hashes(session, source_id: uuid.UUID) -> dict[str, str]:
        rows = (
            await session.execute(
                select(Document.source_uri, Document.meta).where(Document.source_id == source_id)
            )
        ).all()
        return {
            uri: (meta or {}).get("blob_sha") for uri, meta in rows if (meta or {}).get("blob_sha")
        }

    @staticmethod
    def _record_error(stats: dict[str, Any], uri: str, message: str) -> None:
        stats["failed"] += 1
        if len(stats["errors"]) < MAX_RECORDED_ERRORS:
            stats["errors"].append({"item": uri, "error": message})

    async def _apply(
        self,
        app: FastAPI,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
        item: SourceItem | ItemFailure | ItemUnchanged,
        stats: dict[str, Any],
    ) -> None:
        if isinstance(item, ItemFailure):
            self._record_error(stats, item.uri, item.error)
            return
        if isinstance(item, ItemUnchanged):
            stats["unchanged"] += 1
            return
        provisional = {"title_provisional": True} if item.title is None else {}
        meta = {**item.meta, **provisional}
        async with app.state.sessionmaker() as s:
            collection = await s.get(Collection, collection_id)
            try:
                existing = await find_by_source_uri(s, collection_id, item.uri)
                if existing is not None:
                    job = await replace_content(
                        s, app.state.blobs, collection, existing, data=item.data,
                        filename=item.filename, content_type=item.content_type, extra_meta=meta,
                    )  # fmt: skip
                    stats["updated" if job is not None else "unchanged"] += 1
                else:
                    await create_document(
                        s, app.state.blobs, collection, data=item.data, filename=item.filename,
                        content_type=item.content_type, source_type=source_kind(item),
                        source_uri=item.uri, title=item.title, extra_meta=meta, source_id=source_id,
                    )  # fmt: skip
                    stats["created"] += 1
            except Conflict:
                stats["unchanged"] += 1  # identical content already in the collection
            except UnsupportedMedia as e:
                stats["skipped"] += 1
                log.info("skipped unsupported item", extra={"uri": item.uri, "why": str(e)})
            except Exception:
                log.exception("could not store a synced item", extra={"uri": item.uri})
                await s.rollback()
                self._record_error(stats, item.uri, "Could not be stored")

    @staticmethod
    async def _prune(app: FastAPI, collection_id, source_id, listed: set[str]) -> int:
        async with app.state.sessionmaker() as s:
            collection = await s.get(Collection, collection_id)
            rows = (
                await s.execute(
                    select(Document.id, Document.source_uri).where(Document.source_id == source_id)
                )
            ).all()
            gone = [doc_id for doc_id, uri in rows if uri not in listed]
            if not gone:
                return 0
            return await delete_documents(s, app.state.providers, app.state.blobs, collection, gone)


def source_kind(item: SourceItem) -> str:
    return "github" if item.meta.get("repo") else "url"
