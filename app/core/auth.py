"""Authentication and workspace-scoped authorization dependencies (Phase 1.6)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.core.deps import SessionDep, SettingsDep
from app.core.errors import Forbidden, NotFound, Unauthorized
from app.core.security import TokenError, decode_access_token
from app.db.models import Role, User, Workspace, WorkspaceMember

# auto_error=False so a missing header becomes our Problem+JSON 401, not FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False, description="Paste the access_token from /v1/auth/login")

ROLE_RANK = {Role.VIEWER: 0, Role.MEMBER: 1, Role.ADMIN: 2, Role.OWNER: 3}


async def current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    if creds is None:
        raise Unauthorized("Missing bearer token")
    try:
        user_id = uuid.UUID(decode_access_token(creds.credentials, settings))
    except (TokenError, ValueError):
        raise Unauthorized("Invalid or expired token") from None
    user = await session.get(User, user_id)
    # Same message for deleted/deactivated users: a valid token must not reveal account state.
    if user is None or not user.is_active:
        raise Unauthorized("Invalid or expired token")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


@dataclass(frozen=True)
class WorkspaceAccess:
    """The caller's verified relationship to one workspace."""

    user: User
    workspace: Workspace
    role: Role


def require_role(minimum: Role):
    """Dependency factory: the caller must belong to `{workspace_id}` with at least `minimum`.

    Non-members get 404 (not 403) so workspace ids can't be probed for existence.
    """

    async def dependency(
        workspace_id: uuid.UUID, user: CurrentUser, session: SessionDep
    ) -> WorkspaceAccess:
        row = (
            await session.execute(
                select(Workspace, WorkspaceMember.role)
                .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
                .where(Workspace.id == workspace_id, WorkspaceMember.user_id == user.id)
            )
        ).one_or_none()
        if row is None:
            raise NotFound("Workspace not found")
        workspace, role = row
        if ROLE_RANK[role] < ROLE_RANK[minimum]:
            raise Forbidden(f"This action requires the '{minimum.value}' role or higher")
        return WorkspaceAccess(user, workspace, role)

    return dependency


def access(minimum: Role):
    return Annotated[WorkspaceAccess, Depends(require_role(minimum))]


ViewerAccess = access(Role.VIEWER)
MemberAccess = access(Role.MEMBER)
AdminAccess = access(Role.ADMIN)
OwnerAccess = access(Role.OWNER)
