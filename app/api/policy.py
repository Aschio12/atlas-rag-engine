"""A workspace's content rules for answers (Phase 4.6)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.answer import policy_of
from app.core.auth import AdminAccess, ViewerAccess
from app.core.deps import SessionDep
from app.generate.policy import WorkspacePolicy

router = APIRouter(tags=["policy"])


@router.get("/workspaces/{workspace_id}/policy", response_model=WorkspacePolicy)
async def get_policy(access: ViewerAccess):
    """The rules answers in this workspace follow (defaults for a workspace that set none)."""
    return policy_of(access.workspace)


@router.put("/workspaces/{workspace_id}/policy", response_model=WorkspacePolicy)
async def set_policy(body: WorkspacePolicy, access: AdminAccess, session: SessionDep):
    """Replace the workspace's content rules: house instructions, blocked terms, a disclaimer and
    the defaults for PII redaction and prompt-injection handling. Applies to the next answer."""
    access.workspace.policy = body.model_dump()
    await session.commit()
    return body
