"""Which collections a request may search."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFound
from app.db.models import Collection


async def resolve_collections(
    session: AsyncSession, workspace_id: uuid.UUID, wanted: list[uuid.UUID] | None
) -> list[Collection]:
    """All of the workspace's collections, or the named ones. An id from another workspace is
    reported exactly like one that does not exist."""
    query = (
        select(Collection).where(Collection.workspace_id == workspace_id).order_by(Collection.name)
    )
    rows = list((await session.execute(query)).scalars())
    if wanted is None:
        return rows
    by_id = {c.id: c for c in rows}
    chosen = list(dict.fromkeys(wanted))
    if missing := [i for i in chosen if i not in by_id]:
        raise NotFound(f"Collection not found: {missing[0]}")
    return [by_id[i] for i in chosen]
