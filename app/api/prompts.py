"""The prompts the assistant runs on, readable by any signed-in user (Phase 4.1)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.auth import CurrentUser
from app.generate import prompts

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptOut(BaseModel):
    name: str
    version: int
    description: str
    variables: list[str]
    sha: str
    ref: str
    text: str


@router.get("", response_model=list[PromptOut])
async def list_prompts(user: CurrentUser):
    """Every version of every prompt, with its text and content hash. An answer's `prompt` field
    (`answer@1#3f2a9c1d`) names the exact one that produced it."""
    out = []
    for name, versions in prompts.available().items():
        for version in versions:
            p = prompts.get(name, version)
            out.append(
                PromptOut(
                    name=p.name, version=p.version, description=p.description,
                    variables=list(p.variables), sha=p.sha, ref=p.ref, text=p.text,
                )
            )  # fmt: skip
    return out
