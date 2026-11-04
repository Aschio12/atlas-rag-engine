"""Conversations (Phase 4.4): chat with the documents, with memory.

A conversation belongs to the person who started it; nobody else in the workspace can read it, not
even admins. Sending a message runs the same answering pipeline as `POST .../answer`, with the
conversation's memory added: its latest messages verbatim, a rolling summary of older ones and the
workspace's facts. A turn is saved only when it completes, so a failed or abandoned request leaves
no half conversation behind; the answer's citations, confidence, guardrail findings and retrieval
trace are stored with the assistant message, which is what the Inspector shows later.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.answer import SSE_HEADERS, AnswerOut, policy_of, to_exception
from app.api.memory import load_facts
from app.core.auth import ViewerAccess, WorkspaceAccess
from app.core.deps import SessionDep, SettingsDep
from app.core.errors import NotFound
from app.db.models import Conversation, Message, MessageRole
from app.generate import memory as memory_logic
from app.generate.events import Event
from app.generate.options import AnswerOptions
from app.generate.service import AnswerService

log = logging.getLogger("atlas.conversations")

router = APIRouter(prefix="/workspaces/{workspace_id}/conversations", tags=["conversations"])

DEFAULT_TITLE = "New conversation"
TITLE_CHARS = 60
Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ConversationCreate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    metadata: dict[str, Any] | None
    usage: dict[str, Any] | None
    feedback: int | None
    created_at: datetime
    retrieval_trace: dict[str, Any] | None = Field(
        None, description="Only with include_trace=true: it is large"
    )


class ConversationDetail(ConversationOut):
    summary: str = Field(description="The rolling summary of the oldest messages")
    summarized_count: int
    messages: list[MessageOut]


class SendMessage(AnswerOptions):
    content: Content
    stream: bool = Field(False, description="Reply as server-sent events instead of one JSON body")


class Exchange(BaseModel):
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    message_id: uuid.UUID = Field(description="The assistant's message")
    answer: AnswerOut


class Feedback(BaseModel):
    value: Literal[-1, 0, 1] = Field(description="-1 unhelpful, 0 none, +1 helpful")


# ---- helpers ----------------------------------------------------------------------------------


async def owned(
    session: AsyncSession, access: WorkspaceAccess, conversation_id: uuid.UUID
) -> Conversation:
    conversation = (
        await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.workspace_id == access.workspace.id,
                Conversation.user_id == access.user.id,
            )
        )
    ).scalar_one_or_none()
    if conversation is None:
        raise NotFound("Conversation not found")  # also for other people's conversations
    return conversation


async def messages_of(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    rows = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(rows.scalars())


def message_out(m: Message, trace: bool = False) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role.value,
        content=m.content,
        citations=m.citations or [],
        metadata=m.meta,
        usage=m.usage,
        feedback=m.feedback,
        created_at=m.created_at,
        retrieval_trace=m.retrieval_trace if trace else None,
    )


def stored_meta(done: dict[str, Any]) -> dict[str, Any]:
    """What is kept of an answer besides its text, citations, usage and trace. Passage text is
    left out: it is in the documents, and a source panel can be rebuilt from the spans."""
    sources = [{k: v for k, v in s.items() if k != "text"} for s in done.get("sources", [])]
    keep = (
        "abstained", "refused", "reason", "confidence", "grounding", "guardrails", "prompt",
        "model", "warnings", "memory", "related", "inferred_filters",
    )  # fmt: skip
    return {**{k: done.get(k) for k in keep}, "sources": sources}


def _conversation_out(c: Conversation, count: int) -> ConversationOut:
    return ConversationOut(
        id=c.id,
        title=c.title,
        message_count=count,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


# ---- conversations ----------------------------------------------------------------------------


@router.post("", status_code=201, response_model=ConversationOut)
async def create_conversation(body: ConversationCreate, access: ViewerAccess, session: SessionDep):
    now = datetime.now(UTC)
    conversation = Conversation(
        id=uuid.uuid4(),
        workspace_id=access.workspace.id,
        user_id=access.user.id,
        title=(body.title or DEFAULT_TITLE).strip() or DEFAULT_TITLE,
        created_at=now,
        updated_at=now,
    )
    session.add(conversation)
    await session.commit()
    return _conversation_out(conversation, 0)


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    access: ViewerAccess,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Your conversations in this workspace, most recently active first."""
    counts = (
        select(Message.conversation_id, func.count().label("n"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = await session.execute(
        select(Conversation, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .where(
            Conversation.workspace_id == access.workspace.id, Conversation.user_id == access.user.id
        )
        .order_by(Conversation.updated_at.desc(), Conversation.id)
        .limit(limit)
        .offset(offset)
    )
    return [_conversation_out(c, n) for c, n in rows.all()]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID,
    access: ViewerAccess,
    session: SessionDep,
    include_trace: bool = False,
):
    c = await owned(session, access, conversation_id)
    messages = await messages_of(session, c.id)
    base = _conversation_out(c, len(messages))
    return ConversationDetail(
        **base.model_dump(),
        summary=c.summary,
        summarized_count=c.summarized_count,
        messages=[message_out(m, include_trace) for m in messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def rename_conversation(
    conversation_id: uuid.UUID, body: ConversationUpdate, access: ViewerAccess, session: SessionDep
):
    c = await owned(session, access, conversation_id)
    c.title = body.title.strip() or c.title
    c.updated_at = datetime.now(UTC)  # set here: the database's onupdate would need a reload
    await session.commit()
    count = len(await messages_of(session, c.id))
    return _conversation_out(c, count)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID, access: ViewerAccess, session: SessionDep
):
    await session.delete(await owned(session, access, conversation_id))
    await session.commit()
    return Response(status_code=204)


# ---- a turn -----------------------------------------------------------------------------------


async def run_turn(
    session: AsyncSession,
    request: Request,
    access: WorkspaceAccess,
    conversation_id: uuid.UUID,
    body: SendMessage,
) -> AsyncIterator[Event]:
    """Answer one message with the conversation's memory, save the turn when it completes and
    yield the answer's events (the final `done` also carries the saved message ids)."""
    settings = request.app.state.settings
    conversation = await owned(session, access, conversation_id)
    messages = await messages_of(session, conversation.id)
    history = memory_logic.window(
        messages,
        conversation.summarized_count,
        settings.answer_history_messages,
        settings.answer_history_tokens,
    )
    facts = memory_logic.facts_for_prompt(
        await load_facts(session, access.workspace.id), settings.memory_max_facts
    )
    policy = policy_of(access.workspace)
    service = AnswerService(request.app.state.providers, request.app.state.blobs, settings)
    options = AnswerOptions(**body.model_dump(include=set(AnswerOptions.model_fields)))
    prepared = await service.prepare(
        session, access.workspace.id, options, policy, has_history=bool(history)
    )
    async for event in service.run(
        session,
        access.workspace.id,
        body.content,
        prepared,
        policy,
        history=history,
        summary_text=conversation.summary,
        facts=facts,
    ):
        if event.name != "done":
            yield event
            continue
        done = event.data
        now = datetime.now(UTC)
        user = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=body.content,
            citations=[],
            created_at=now,
        )
        assistant = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=done["answer"],
            citations=done["citations"],
            retrieval_trace=done.get("retrieval_trace"),
            usage=done.get("usage"),
            meta=stored_meta(done),
            created_at=now + timedelta(microseconds=1),
        )
        session.add_all([user, assistant])
        if conversation.title == DEFAULT_TITLE:
            conversation.title = " ".join(body.content.split())[:TITLE_CHARS]
        conversation.updated_at = now
        await session.commit()
        yield Event(
            "done",
            {
                **done,
                "conversation_id": str(conversation.id),
                "user_message_id": str(user.id),
                "message_id": str(assistant.id),
            },
        )
        await _remember(request, session, conversation, [*messages, user, assistant])


async def _remember(
    request: Request, session: AsyncSession, conversation: Conversation, messages: list[Message]
) -> None:
    """Fold old messages into the summary when enough have piled up. Never fails the turn."""
    settings = request.app.state.settings
    try:
        changed = await memory_logic.maybe_summarise(
            request.app.state.providers.llm,
            conversation,
            messages,
            max_messages=settings.answer_history_messages,
            after=settings.answer_summarize_after,
        )
        if changed:
            await session.commit()
    except Exception:  # a summary is a convenience; the saved turn must not depend on it
        log.exception("could not update the conversation summary")
        await session.rollback()


@router.post(
    "/{conversation_id}/messages",
    response_model=Exchange,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "JSON, or SSE if stream"}
    },
)
async def send_message(
    conversation_id: uuid.UUID,
    body: SendMessage,
    access: ViewerAccess,
    session: SessionDep,
    request: Request,
    settings: SettingsDep,
):
    """Ask a question in a conversation. Same answer as `POST .../answer`, with the conversation's
    memory, saved when it completes. With `"stream": true` the events are those of `.../answer`;
    the final `done` event also carries `conversation_id`, `user_message_id` and `message_id`."""
    conversation = await owned(session, access, conversation_id)
    if not body.stream:
        done: dict[str, Any] | None = None
        async for event in run_turn(session, request, access, conversation_id, body):
            if event.name == "error":
                raise to_exception(event.data)
            if event.name == "done":
                done = event.data
        assert done is not None
        return {
            "conversation_id": conversation.id,
            "user_message_id": done["user_message_id"],
            "message_id": done["message_id"],
            "answer": done,
        }

    # Validate before the stream starts, so mistakes are ordinary error responses.
    history = bool(
        memory_logic.recent(
            await messages_of(session, conversation.id),
            conversation.summarized_count,
            settings.answer_history_messages,
        )
    )
    await AnswerService(request.app.state.providers, request.app.state.blobs, settings).prepare(
        session,
        access.workspace.id,
        AnswerOptions(**body.model_dump(include=set(AnswerOptions.model_fields))),
        policy_of(access.workspace),
        has_history=history,
    )
    sessionmaker = request.app.state.sessionmaker
    user_id, workspace_id = access.user.id, access.workspace.id

    async def stream() -> AsyncIterator[str]:
        yield ": stream open\n\n"
        async with sessionmaker() as own:  # the request's session may be closed by now
            fresh = await _reload(own, access, workspace_id, user_id)
            async for event in run_turn(own, request, fresh, conversation_id, body):
                yield event.sse()

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


async def _reload(
    session: AsyncSession, access: WorkspaceAccess, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> WorkspaceAccess:
    """The same access, with workspace and user loaded in `session`."""
    from app.db.models import User, Workspace

    workspace = await session.get(Workspace, workspace_id)
    user = await session.get(User, user_id)
    return WorkspaceAccess(user, workspace, access.role)


@router.post("/{conversation_id}/messages/{message_id}/feedback", response_model=MessageOut)
async def give_feedback(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    body: Feedback,
    access: ViewerAccess,
    session: SessionDep,
):
    """Mark an answer helpful (+1), unhelpful (-1) or clear the mark (0)."""
    conversation = await owned(session, access, conversation_id)
    message = (
        await session.execute(
            select(Message).where(
                Message.id == message_id,
                Message.conversation_id == conversation.id,
                Message.role == MessageRole.ASSISTANT,
            )
        )
    ).scalar_one_or_none()
    if message is None:
        raise NotFound("Answer not found")
    message.feedback = body.value or None
    await session.commit()
    return message_out(message)
