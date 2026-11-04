import uuid

import pytest
from sqlalchemy import select

from app.core.errors import Conflict, UnsupportedMedia
from app.db.models import (
    Collection,
    Document,
    DocumentStatus,
    IngestJob,
    JobState,
    Source,
    SourceType,
    User,
    Workspace,
)
from app.ingest import keys
from app.ingest.service import create_document, find_by_source_uri, replace_content
from app.storage import LocalBlobStore


@pytest.fixture
async def env(app, tmp_path):
    blobs = LocalBlobStore(tmp_path / "svc")
    async with app.state.sessionmaker() as s:
        user = User(email="svc@example.com", password_hash="x")
        s.add(user)
        await s.flush()
        ws = Workspace(name="w", slug="w-svc", owner_id=user.id)
        s.add(ws)
        await s.flush()
        col = Collection(workspace_id=ws.id, name="c")
        s.add(col)
        await s.flush()
        src = Source(collection_id=col.id, type=SourceType.WEB, name="site", config={})
        s.add(src)
        await s.commit()
        yield app.state.sessionmaker, blobs, col.id, src.id


async def add(env, text=b"hello world", **kw):
    sm, blobs, col_id, _ = env
    async with sm() as s:
        col = await s.get(Collection, col_id)
        return await create_document(
            s,
            blobs,
            col,
            data=text,
            filename=kw.pop("filename", "page.txt"),
            content_type="text/plain",
            source_type=kw.pop("source_type", "url"),
            **kw,
        )


async def test_documents_can_be_linked_to_their_source_and_found_by_address(env):
    sm, blobs, col_id, src_id = env
    doc, job = await add(env, source_uri="https://example.com/a", source_id=src_id)
    assert doc.source_id == src_id and job.state == JobState.QUEUED
    async with sm() as s:
        found = await find_by_source_uri(s, col_id, "https://example.com/a")
        assert found.id == doc.id
        assert await find_by_source_uri(s, col_id, "https://example.com/other") is None
        assert await find_by_source_uri(s, uuid.uuid4(), "https://example.com/a") is None


async def test_replace_content_updates_the_document_and_queues_a_full_reparse(env):
    sm, blobs, col_id, src_id = env
    doc, job = await add(env, source_uri="https://example.com/a", source_id=src_id)
    # finish the first job so a new one can be queued
    async with sm() as s:
        await s.execute(IngestJob.__table__.update().values(state=JobState.DONE))
        await s.execute(Document.__table__.update().values(status=DocumentStatus.READY))
        await s.commit()
    async with sm() as s:
        col = await s.get(Collection, col_id)
        d = await s.get(Document, doc.id)
        new_job = await replace_content(
            s, blobs, col, d, data=b"changed text", filename="page.txt",
            content_type="text/plain", extra_meta={"etag": "v2"},
        )  # fmt: skip
        assert new_job is not None and new_job.options == {"reparse": True}
        assert (d.status, d.size_bytes, d.meta["etag"]) == (DocumentStatus.PENDING, 12, "v2")
    assert await blobs.get(keys.original_key(*(await _ids(sm, col_id)), doc.id)) == b"changed text"


async def _ids(sm, col_id):
    async with sm() as s:
        col = await s.get(Collection, col_id)
        return col.workspace_id, col.id


async def test_replace_content_with_identical_bytes_does_nothing(env):
    sm, blobs, col_id, _ = env
    doc, _ = await add(env, b"same")
    async with sm() as s:
        col = await s.get(Collection, col_id)
        d = await s.get(Document, doc.id)
        assert await replace_content(
            s, blobs, col, d, data=b"same", filename="p.txt", content_type="text/plain"
        ) is None  # fmt: skip


async def test_replace_content_cannot_collide_with_another_document(env):
    sm, blobs, col_id, _ = env
    first, _ = await add(env, b"first body", filename="one.txt")
    second, _ = await add(env, b"second body", filename="two.txt")
    async with sm() as s:
        col = await s.get(Collection, col_id)
        d = await s.get(Document, second.id)
        with pytest.raises(Conflict) as exc:
            await replace_content(
                s, blobs, col, d, data=b"first body", filename="two.txt", content_type="text/plain"
            )
        assert exc.value.extra["existing_document_id"] == str(first.id)


async def test_replace_content_rejects_unsupported_files(env):
    sm, blobs, col_id, _ = env
    doc, _ = await add(env, b"fine")
    async with sm() as s:
        col = await s.get(Collection, col_id)
        d = await s.get(Document, doc.id)
        with pytest.raises(UnsupportedMedia):
            await replace_content(
                s, blobs, col, d, data=b"\x00\x01\x02binary", filename="x.bin", content_type=None
            )


async def test_deleting_a_source_keeps_its_documents(env):
    sm, blobs, col_id, src_id = env
    doc, _ = await add(env, source_id=src_id, source_uri="https://example.com/a")
    async with sm() as s:
        await s.delete(await s.get(Source, src_id))
        await s.commit()
    async with sm() as s:
        kept = (await s.execute(select(Document).where(Document.id == doc.id))).scalar_one()
        assert kept.source_id is None


async def test_source_names_are_unique_per_collection(env):
    from sqlalchemy.exc import IntegrityError

    sm, _, col_id, _ = env
    async with sm() as s:
        s.add(Source(collection_id=col_id, type=SourceType.GITHUB, name="site", config={}))
        with pytest.raises(IntegrityError):
            await s.commit()
