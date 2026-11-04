"""ORM models. Portable types (Uuid, JSON, String enums) so tests can run on SQLite."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin


def _enum(e: type[enum.Enum]) -> Enum:
    # Stored as VARCHAR + CHECK, so adding a member never needs an ALTER TYPE.
    return Enum(e, native_enum=False, length=32, values_callable=lambda x: [m.value for m in x])


class Role(enum.StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class User(IdMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    is_active: Mapped[bool] = mapped_column(default=True)


class Workspace(IdMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    plan: Mapped[str] = mapped_column(String(32), default="free")
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    # Content rules for answers (Phase 4.6): a validated WorkspacePolicy as JSON.
    policy: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))

    members: Mapped[list["WorkspaceMember"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    collections: Mapped[list["Collection"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    retrievals: Mapped[list["Retrieval"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    memory_facts: Mapped[list["MemoryFact"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class WorkspaceMember(IdMixin, TimestampMixin, Base):
    __tablename__ = "workspace_members"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[Role] = mapped_column(_enum(Role), default=Role.MEMBER)

    workspace: Mapped[Workspace] = relationship(back_populates="members")


class ChunkingStrategy(enum.StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    CONTEXTUAL = "contextual"


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SourceType(enum.StrEnum):
    WEB = "web"
    GITHUB = "github"


class SourceStatus(enum.StrEnum):
    IDLE = "idle"  # created or finished with nothing running
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Collection(IdMixin, TimestampMixin, Base):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("workspace_id", "name"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    chunking_strategy: Mapped[ChunkingStrategy] = mapped_column(
        _enum(ChunkingStrategy), default=ChunkingStrategy.RECURSIVE
    )
    embed_model: Mapped[str] = mapped_column(String(200), default="BAAI/bge-small-en-v1.5")
    # Validated ChunkingConfig as JSON; fixed at creation, like the strategy (changing either
    # means re-indexing, Phase 2.8). Missing keys fall back to the ChunkingConfig defaults.
    chunking_config: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    # Validated EnrichmentConfig as JSON (Phase 2.5); missing keys fall back to its defaults.
    enrichment_config: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))

    workspace: Mapped[Workspace] = relationship(back_populates="collections")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )
    sources: Mapped[list["Source"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )


class Source(IdMixin, TimestampMixin, Base):
    """An external place documents are pulled from (a website, a GitHub repository). A sync
    discovers its items and adds or updates the matching documents (Phase 2.7)."""

    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("collection_id", "name"),)

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[SourceType] = mapped_column(_enum(SourceType))
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[SourceStatus] = mapped_column(_enum(SourceStatus), default=SourceStatus.IDLE)
    # discovered / created / updated / unchanged / failed counts and the first few errors
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    collection: Mapped[Collection] = relationship(back_populates="sources")


class Document(IdMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("collection_id", "content_hash"),)

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(32))  # file | url | text | github
    source_uri: Mapped[str | None] = mapped_column(String(2000))
    content_hash: Mapped[str] = mapped_column(String(64))  # sha256, dedup per collection
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus), default=DocumentStatus.PENDING
    )
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    # User-defined attributes (year, author, tags, ...), searchable as filters (Phase 3.7).
    attributes: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    # The source that produced this document, if any (kept when the source is deleted).
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL"), index=True
    )

    collection: Mapped[Collection] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(IdMixin, Base):
    """A retrievable passage. Its id is also the point id in the vector store."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "ordinal"),)

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Character offsets into the parsed document text, used for citation highlighting.
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Retrieval(IdMixin, Base):
    """A saved search: the question, how it went, and the whole trace for the Inspector
    (Phase 3.8). Passage text is not kept, only what identifies and explains each hit; the
    chunks themselves are fetched from their own tables."""

    __tablename__ = "retrievals"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    query: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16))
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    took_ms: Mapped[float] = mapped_column(Float, default=0.0)
    record: Mapped[dict] = mapped_column(JSON, default=dict)  # hits, context, warnings, trace
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


class JobState(enum.StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    ENRICHING = "enriching"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


class IngestJob(IdMixin, TimestampMixin, Base):
    __tablename__ = "ingest_jobs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[JobState] = mapped_column(_enum(JobState), default=JobState.QUEUED, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    # Per-run switches, e.g. {"reparse": false} to rebuild chunks and vectors from stored text.
    options: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Queue bookkeeping: a queued job is not picked up before `run_after` (retry backoff), and a
    # running job whose `locked_until` lease has expired is presumed orphaned by a dead worker.
    run_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(IdMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    # Rolling summary of the oldest messages (Phase 4.4): `summarized_count` messages, counted from
    # the start of the conversation, are folded into `summary`; newer ones are sent verbatim.
    summary: Mapped[str] = mapped_column(Text, default="", server_default=text("''"))
    summarized_count: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(IdMixin, TimestampMixin, Base):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    # Per-stage scores and reasons, rendered by the Retrieval Inspector (Phase 3.8).
    retrieval_trace: Mapped[dict | None] = mapped_column(JSON)
    # Tokens, latency per stage and cost (Phase 6.8).
    usage: Mapped[dict | None] = mapped_column(JSON)
    feedback: Mapped[int | None] = mapped_column(Integer)  # -1, 0, +1
    # Everything else worth knowing about an answer (Phase 4): confidence, grounding, guardrail
    # findings, the prompt version, warnings, whether it abstained.
    meta: Mapped[dict | None] = mapped_column("metadata", JSON)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class MemoryFact(IdMixin, TimestampMixin, Base):
    """A long-term fact the assistant is told in every conversation of a workspace (Phase 4.4):
    "answer in British English", "our fiscal year starts in April". Written by people, so they
    can read, correct and switch off what the assistant 'remembers'."""

    __tablename__ = "memory_facts"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(String(500))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )


class EvalStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRun(IdMixin, TimestampMixin, Base):
    __tablename__ = "eval_runs"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    suite: Mapped[str] = mapped_column(String(120), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[EvalStatus] = mapped_column(_enum(EvalStatus), default=EvalStatus.RUNNING)
