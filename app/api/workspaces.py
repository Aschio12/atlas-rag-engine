"""Workspaces: the tenant boundary. Every other resource hangs off one."""

from __future__ import annotations

import re
import secrets
import unicodedata
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AdminAccess, CurrentUser, OwnerAccess, ViewerAccess
from app.core.deps import BlobsDep, SessionDep
from app.core.errors import Conflict
from app.db.models import Collection, Role, Workspace, WorkspaceMember
from app.ingest import keys
from app.ingest.purge import purge_collection

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


def slugify(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:60].strip("-")
    return slug or "workspace"


def _clean_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120, examples=["Acme Support"])
    slug: str | None = Field(
        default=None, min_length=2, max_length=60, pattern=SLUG_PATTERN, examples=["acme-support"]
    )

    _name = field_validator("name")(_clean_name)


class WorkspaceUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    _name = field_validator("name")(_clean_name)


class WorkspaceOut(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: str
    role: Role = Field(description="The caller's role in this workspace")
    created_at: datetime


class WorkspaceList(BaseModel):
    items: list[WorkspaceOut]
    total: int


def to_out(workspace: Workspace, role: Role) -> WorkspaceOut:
    return WorkspaceOut(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        plan=workspace.plan,
        role=role,
        created_at=workspace.created_at,
    )


async def _slug_taken(session: AsyncSession, slug: str) -> bool:
    return (
        await session.execute(select(Workspace.id).where(Workspace.slug == slug))
    ).first() is not None


async def _unique_slug(session: AsyncSession, name: str) -> str:
    base = slugify(name)
    if not await _slug_taken(session, base):
        return base
    for _ in range(5):
        candidate = f"{base[:55].rstrip('-')}-{secrets.token_hex(2)}"
        if not await _slug_taken(session, candidate):
            return candidate
    raise Conflict("Could not generate a unique slug; please provide one")


@router.post("", status_code=201, response_model=WorkspaceOut)
async def create_workspace(body: WorkspaceCreate, user: CurrentUser, session: SessionDep):
    """Create a workspace; the caller becomes its owner."""
    if body.slug is not None:
        if await _slug_taken(session, body.slug):
            raise Conflict(f"The slug '{body.slug}' is already taken")
        slug = body.slug
    else:
        slug = await _unique_slug(session, body.name)
    workspace = Workspace(name=body.name, slug=slug, owner_id=user.id)
    workspace.members.append(WorkspaceMember(user_id=user.id, role=Role.OWNER))
    session.add(workspace)
    try:
        await session.commit()
    except IntegrityError:
        raise Conflict("That slug was just taken; please try again") from None
    return to_out(workspace, Role.OWNER)


@router.get("", response_model=WorkspaceList)
async def list_workspaces(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Workspaces the caller belongs to."""
    base = (
        select(Workspace, WorkspaceMember.role)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .where(WorkspaceMember.user_id == user.id)
    )
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(
            base.order_by(Workspace.created_at.desc(), Workspace.id).limit(limit).offset(offset)
        )
    ).all()
    return WorkspaceList(items=[to_out(w, r) for w, r in rows], total=total)


@router.get("/{workspace_id}", response_model=WorkspaceOut)
async def get_workspace(access: ViewerAccess):
    return to_out(access.workspace, access.role)


@router.patch("/{workspace_id}", response_model=WorkspaceOut)
async def update_workspace(body: WorkspaceUpdate, access: AdminAccess, session: SessionDep):
    """Rename a workspace (admin or owner). The slug is permanent."""
    access.workspace.name = body.name
    await session.commit()
    return to_out(access.workspace, access.role)


@router.delete("/{workspace_id}", status_code=204)
async def delete_workspace(
    request: Request, access: OwnerAccess, session: SessionDep, blobs: BlobsDep
) -> Response:
    """Delete a workspace and everything in it, including vectors and files (owner only)."""
    prefix = keys.workspace_prefix(access.workspace.id)
    collection_ids = (
        (
            await session.execute(
                select(Collection.id).where(Collection.workspace_id == access.workspace.id)
            )
        )
        .scalars()
        .all()
    )
    for cid in collection_ids:
        await purge_collection(request.app.state.providers, cid)
    await session.delete(access.workspace)
    await session.commit()
    await blobs.delete_prefix(prefix)
    return Response(status_code=204)
