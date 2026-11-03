import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.base import Base
from app.db.models import (
    Chunk,
    Collection,
    Conversation,
    Document,
    Message,
    MessageRole,
    User,
    Workspace,
)
from app.db.session import make_sessionmaker


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with make_sessionmaker(engine)() as s:
        yield s
    await engine.dispose()


async def make_workspace(s):
    user = User(email="a@b.co", password_hash="x")
    s.add(user)
    await s.flush()
    ws = Workspace(name="Demo", slug="demo", owner_id=user.id)
    s.add(ws)
    await s.flush()
    return user, ws


async def test_defaults_and_ids(session):
    user, ws = await make_workspace(session)
    assert user.id and ws.id and ws.plan == "free"


async def test_duplicate_content_hash_in_collection_rejected(session):
    _, ws = await make_workspace(session)
    col = Collection(workspace_id=ws.id, name="docs")
    session.add(col)
    await session.flush()
    h = hashlib.sha256(b"same").hexdigest()
    session.add(Document(collection_id=col.id, title="a", source_type="text", content_hash=h))
    await session.flush()
    session.add(Document(collection_id=col.id, title="b", source_type="text", content_hash=h))
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_deleting_document_cascades_to_chunks(session):
    _, ws = await make_workspace(session)
    col = Collection(workspace_id=ws.id, name="docs")
    session.add(col)
    await session.flush()
    doc = Document(collection_id=col.id, title="a", source_type="text", content_hash="h")
    session.add(doc)
    await session.flush()
    session.add(Chunk(document_id=doc.id, ordinal=0, text="hi", char_start=0, char_end=2))
    await session.flush()
    await session.refresh(doc, ["chunks"])
    await session.delete(doc)
    await session.flush()
    assert (await session.execute(select(Chunk))).scalars().all() == []


async def test_message_stores_citations_and_trace(session):
    user, ws = await make_workspace(session)
    conv = Conversation(workspace_id=ws.id, user_id=user.id)
    session.add(conv)
    await session.flush()
    msg = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="answer [1]",
        citations=[{"chunk_id": "c1", "n": 1}],
        retrieval_trace={"fused": [{"id": "c1", "rrf": 0.03}]},
    )
    session.add(msg)
    await session.flush()
    await session.refresh(msg)
    assert msg.citations[0]["n"] == 1
    assert msg.retrieval_trace["fused"][0]["rrf"] == 0.03
