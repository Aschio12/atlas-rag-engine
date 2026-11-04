"""Answering questions from a workspace's documents (Phase 4): one-shot or streamed."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from app.api.memory import load_facts
from app.api.search import PassageOut
from app.core.auth import ViewerAccess
from app.core.deps import SessionDep, SettingsDep
from app.core.errors import BadRequest
from app.db.models import Workspace
from app.generate.memory import facts_for_prompt
from app.generate.options import AnswerOptions
from app.generate.policy import WorkspacePolicy
from app.generate.service import AnswerService
from app.providers.errors import ProviderError
from app.retrieve.filters import MetadataFilter
from app.retrieve.transform import Turn

router = APIRouter(tags=["answers"])

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class AnswerRequest(AnswerOptions):
    question: Question
    stream: bool = Field(False, description="Reply as server-sent events instead of one JSON body")
    history: list[Turn] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier turns, oldest first, for a follow-up (conversations keep their own)",
    )


class CitationOut(BaseModel):
    label: str
    document_id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    page: int | None
    heading_path: list[str]
    char_start: int
    char_end: int
    chunk_ids: list[uuid.UUID]
    passages: list[PassageOut]
    mentions: int
    support: float | None
    supported: bool | None


class SourceOut(BaseModel):
    label: str
    title: str
    page: int | None
    heading_path: list[str]
    document_id: uuid.UUID
    collection_id: uuid.UUID
    char_start: int
    char_end: int
    chunk_ids: list[uuid.UUID]
    passages: list[PassageOut]
    score: float
    tokens: int
    text: str
    flagged: str | None = None


class ConfidenceOut(BaseModel):
    score: float
    level: str = Field(description="high, medium, low or none")
    signals: dict[str, float | None]
    reasons: list[str]


class AnswerOut(BaseModel):
    answer: str
    abstained: bool = Field(description="True when the answer is 'I couldn't find this'")
    refused: bool = Field(description="True when a workspace rule or guardrail stopped the answer")
    reason: str | None = Field(description="Why it abstained or refused, if it did")
    citations: list[CitationOut]
    grounding: dict[str, Any] | None
    confidence: ConfidenceOut
    sources: list[SourceOut]
    related: list[dict[str, Any]] = Field(description="Closest sources, when abstaining")
    inferred_filters: list[MetadataFilter]
    guardrails: dict[str, Any]
    prompt: str = Field(description="The prompt that produced it: name@version#hash")
    model: str
    usage: dict[str, Any]
    warnings: list[str]
    memory: dict[str, Any] = Field(description="What the prompt carried: facts, summary, turns")
    retrieval_trace: dict[str, Any] | None


def policy_of(workspace: Workspace) -> WorkspacePolicy:
    return WorkspacePolicy.model_validate(workspace.policy or {})


def to_exception(data: dict[str, Any]) -> Exception:
    if data.get("code") == "provider_error":
        return ProviderError(data.get("provider", "provider"), data["message"], data.get("status"))
    return BadRequest(data.get("message", "The question could not be answered"))


@router.post(
    "/workspaces/{workspace_id}/answer",
    response_model=AnswerOut,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "JSON, or SSE if stream"}
    },
)
async def answer(
    body: AnswerRequest,
    access: ViewerAccess,
    session: SessionDep,
    request: Request,
    settings: SettingsDep,
):
    """Answer a question using only the workspace's documents, with citations.

    The reply says what it is based on: `citations` map every `[S1]` in the text to a document,
    page and character span; `confidence` is a badge with its reasons; when the documents do not
    contain the answer, `abstained` is true and the closest sources are listed in `related`.

    With `"stream": true` the reply is server-sent events: `retrieval` (sources found, before any
    text), `token`, `citation` (first time a source is cited), then `done` with the final checked
    answer (use its `answer`, which can differ from the streamed text where invalid citations were
    removed) or `error`.
    """
    policy = policy_of(access.workspace)
    service = AnswerService(request.app.state.providers, request.app.state.blobs, settings)
    workspace_id = access.workspace.id
    options = AnswerOptions(**body.model_dump(include=set(AnswerOptions.model_fields)))
    facts = facts_for_prompt(await load_facts(session, workspace_id), settings.memory_max_facts)
    prepared = await service.prepare(
        session, workspace_id, options, policy, has_history=bool(body.history)
    )
    if not body.stream:
        done = None
        async for event in service.run(
            session,
            workspace_id,
            body.question,
            prepared,
            policy,
            history=body.history,
            facts=facts,
        ):
            if event.name == "error":
                raise to_exception(event.data)
            if event.name == "done":
                done = event.data
        assert done is not None  # run() always ends with done or error
        return done

    sessionmaker = request.app.state.sessionmaker

    async def stream() -> AsyncIterator[str]:
        yield ": stream open\n\n"
        async with sessionmaker() as own:  # the request's session may be closed by now
            ready = await service.prepare(
                own, workspace_id, options, policy, has_history=bool(body.history)
            )
            async for event in service.run(
                own, workspace_id, body.question, ready, policy, history=body.history, facts=facts
            ):
                yield event.sse()

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)
