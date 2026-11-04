"""Saved searches (Phase 3.8): the history behind the Retrieval Inspector."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.auth import ROLE_RANK, ViewerAccess
from app.core.deps import SessionDep
from app.core.errors import NotFound
from app.db.models import Retrieval, Role

router = APIRouter(prefix="/workspaces/{workspace_id}/retrievals", tags=["retrievals"])


class RetrievalBrief(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None
    query: str
    mode: str
    hit_count: int
    took_ms: float
    created_at: datetime


class RetrievalList(BaseModel):
    items: list[RetrievalBrief]
    total: int


class RetrievalDetail(RetrievalBrief):
    record: dict[str, Any]


def _visible(access):
    """Members see their own searches; admins and owners see the workspace's."""
    where = [Retrieval.workspace_id == access.workspace.id]
    if ROLE_RANK[access.role] < ROLE_RANK[Role.ADMIN]:
        where.append(Retrieval.user_id == access.user.id)
    return where


async def _get(session, access, retrieval_id: uuid.UUID) -> Retrieval:
    row = (
        await session.execute(
            select(Retrieval).where(*_visible(access), Retrieval.id == retrieval_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise NotFound("Search not found")  # also for other people's searches: no probing
    return row


@router.get("", response_model=RetrievalList)
async def list_retrievals(
    access: ViewerAccess,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Recent searches, newest first (yours; admins see everyone's)."""
    where = _visible(access)
    total = (
        await session.execute(select(func.count()).select_from(Retrieval).where(*where))
    ).scalar_one()
    rows = (
        (
            await session.execute(
                select(Retrieval)
                .where(*where)
                .order_by(Retrieval.created_at.desc(), Retrieval.id)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return RetrievalList(
        items=[RetrievalBrief.model_validate(r, from_attributes=True) for r in rows], total=total
    )


@router.get("/{retrieval_id}", response_model=RetrievalDetail)
async def get_retrieval(retrieval_id: uuid.UUID, access: ViewerAccess, session: SessionDep):
    """One saved search with its full trace: every stage, its candidates, scores and reasons.
    Passage text is not stored; fetch chunks by id for it."""
    row = await _get(session, access, retrieval_id)
    return RetrievalDetail(
        **RetrievalBrief.model_validate(row, from_attributes=True).model_dump(), record=row.record
    )


@router.delete("/{retrieval_id}", status_code=204)
async def delete_retrieval(retrieval_id: uuid.UUID, access: ViewerAccess, session: SessionDep):
    row = await _get(session, access, retrieval_id)
    await session.delete(row)
    await session.commit()
    return Response(status_code=204)
