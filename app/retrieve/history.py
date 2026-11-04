"""Saving searches with their trace (Phase 3.8).

A saved retrieval is what the Retrieval Inspector shows later: the question, the options, every
stage with its candidates, scores and reasons, and the hits. Passage text is left out (it is in
the chunks table and can change), so a record stays small and never becomes a second copy of the
corpus. The chat in Phase 4 attaches the same record to the answer it produced.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Retrieval


def strip_text(response: dict[str, Any]) -> dict[str, Any]:
    """A copy of a search response without passage text (hits and context blocks)."""
    record = dict(response)
    record["hits"] = [{k: v for k, v in h.items() if k != "text"} for h in response.get("hits", [])]
    if response.get("context"):
        context = dict(response["context"])
        context["blocks"] = [
            {k: v for k, v in b.items() if k != "text"} for b in context.get("blocks", [])
        ]
        record["context"] = context
    return record


async def save_retrieval(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID | None,
    response: dict[str, Any],
    keep: int,
) -> Retrieval | None:
    """Store a search response and drop the workspace's oldest records beyond `keep`."""
    if keep <= 0:
        return None
    row = Retrieval(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        user_id=user_id,
        query=response["query"],
        mode=response["mode"],
        hit_count=len(response.get("hits", [])),
        took_ms=float(response.get("trace", {}).get("total_ms", 0.0)),
        record=strip_text(response),
    )
    session.add(row)
    await session.flush()
    stale = (
        select(Retrieval.id)
        .where(Retrieval.workspace_id == workspace_id)
        .order_by(Retrieval.created_at.desc(), Retrieval.id)
        .offset(keep)
    )
    old = list((await session.execute(stale)).scalars())
    if old:
        await session.execute(delete(Retrieval).where(Retrieval.id.in_(old)))
    await session.commit()
    return row
