"""Long-term facts the assistant is told in every conversation of a workspace (Phase 4.4)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import SessionDep, SettingsDep
from app.core.errors import Conflict, NotFound
from app.db.models import MemoryFact

router = APIRouter(prefix="/workspaces/{workspace_id}/memory", tags=["memory"])

FactText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class FactCreate(BaseModel):
    text: FactText = Field(examples=["Answer in British English."])


class FactUpdate(BaseModel):
    text: FactText | None = None
    enabled: bool | None = None


class FactOut(BaseModel):
    id: uuid.UUID
    text: str
    enabled: bool
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


async def load_facts(session: AsyncSession, workspace_id: uuid.UUID) -> list[MemoryFact]:
    rows = await session.execute(
        select(MemoryFact)
        .where(MemoryFact.workspace_id == workspace_id)
        .order_by(MemoryFact.created_at, MemoryFact.id)
    )
    return list(rows.scalars())


async def _get(session: AsyncSession, workspace_id: uuid.UUID, fact_id: uuid.UUID) -> MemoryFact:
    fact = (
        await session.execute(
            select(MemoryFact).where(
                MemoryFact.id == fact_id, MemoryFact.workspace_id == workspace_id
            )
        )
    ).scalar_one_or_none()
    if fact is None:
        raise NotFound("Fact not found")
    return fact


@router.get("", response_model=list[FactOut])
async def list_facts(access: ViewerAccess, session: SessionDep):
    """Everything the assistant is told in this workspace, enabled or not."""
    return [
        FactOut.model_validate(f, from_attributes=True)
        for f in await load_facts(session, access.workspace.id)
    ]


@router.post("", status_code=201, response_model=FactOut)
async def add_fact(
    body: FactCreate, access: MemberAccess, session: SessionDep, settings: SettingsDep
):
    total = (
        await session.execute(
            select(func.count())
            .select_from(MemoryFact)
            .where(MemoryFact.workspace_id == access.workspace.id)
        )
    ).scalar_one()
    if total >= settings.memory_max_facts:
        raise Conflict(f"A workspace can hold at most {settings.memory_max_facts} facts")
    fact = MemoryFact(
        id=uuid.uuid4(),
        workspace_id=access.workspace.id,
        text=body.text,
        created_by=access.user.id,
        created_at=datetime.now(UTC),  # set here: the database's clock ties within a second
    )
    session.add(fact)
    await session.commit()
    await session.refresh(fact)
    return FactOut.model_validate(fact, from_attributes=True)


@router.patch("/{fact_id}", response_model=FactOut)
async def update_fact(
    fact_id: uuid.UUID, body: FactUpdate, access: MemberAccess, session: SessionDep
):
    """Edit a fact, or switch it off without losing it (`enabled: false`)."""
    fact = await _get(session, access.workspace.id, fact_id)
    if body.text is not None:
        fact.text = body.text
    if body.enabled is not None:
        fact.enabled = body.enabled
    await session.commit()
    await session.refresh(fact)
    return FactOut.model_validate(fact, from_attributes=True)


@router.delete("/{fact_id}", status_code=204)
async def delete_fact(fact_id: uuid.UUID, access: MemberAccess, session: SessionDep):
    fact = await _get(session, access.workspace.id, fact_id)
    await session.delete(fact)
    await session.commit()
    return Response(status_code=204)
