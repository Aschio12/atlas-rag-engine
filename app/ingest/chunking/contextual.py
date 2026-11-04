"""Contextual chunking (2.4.5): each chunk gets an LLM-written sentence that situates it.

A chunk like "Revenue grew 3% over the previous quarter." is ambiguous on its own: whose revenue,
which quarter? Following Anthropic's "contextual retrieval" idea, an LLM reads the surrounding
document and writes a short context ("From ACME Corp's Q2 2025 report, regional results
section."), which is embedded and searched together with the chunk. Published results show a
large drop in failed retrievals; our evaluation harness (Phase 6) will measure it on your data.

Cost control, because this is one LLM call per chunk:
* the document goes in a **cached prefix** shared by all chunks of the same window, so after the
  first call the provider bills it at a fraction (Anthropic prompt caching; OpenAI-compatible
  servers cache automatically). Calls are sequential to keep the cache warm;
* long documents are cut into fixed windows so the cached prefix stays the same;
* `context_max_chunks` caps the number of calls per document;
* a failed call just means that chunk has no context; if the provider is clearly down the error
  is raised so the job retries instead of silently storing context-less chunks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.ingest.chunking.base import ChunkDraft, Chunker
from app.ingest.chunking.config import ChunkingConfig
from app.ingest.chunking.recursive import RecursiveChunker
from app.ingest.parsers import ParsedDocument
from app.providers.base import LLM, ChatMessage
from app.providers.errors import ProviderError

log = logging.getLogger("atlas.chunking.contextual")

WINDOW_CHARS = 48_000  # document text placed in the cached prefix (about 12k tokens)
WINDOW_SLACK = 8_000  # so chunks that start near a window's end still fit inside it
MAX_CONTEXT_CHARS = 400
MAX_FAILURES_BEFORE_ABORT = 3

SYSTEM_PREFIX = (
    "You help improve search over documents. Here is the document (or the part of it that "
    "contains the passage you will be asked about).\n<document>\n{document}\n</document>"
)
USER_PROMPT = (
    "Here is a passage taken from that document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Write one or two short sentences that situate this passage within the overall document, so "
    "that someone searching can find it: name the subject, the section and anything the passage "
    "assumes but does not say. Answer with the context only, no preamble."
)


@dataclass
class ContextReport:
    """What the contextual pass did, stored on the document for transparency and cost review."""

    calls: int = 0
    failed: int = 0
    skipped: int = 0  # chunks beyond the context_max_chunks cap
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def clean_context(raw: str) -> str | None:
    text = re.sub(r"\s+", " ", raw).strip().strip("\"'“”")
    text = re.sub(r"^(context|situating context)\s*[:\-–]\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return None
    return text if len(text) <= MAX_CONTEXT_CHARS else text[: MAX_CONTEXT_CHARS - 1].rstrip() + "…"


class ContextualChunker:
    def __init__(self, llm: LLM, base: Chunker | None = None):
        self.llm = llm
        self.base = base or RecursiveChunker()
        self.report = ContextReport()

    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        drafts = await self.base.chunk(parsed, cfg)
        self.report = ContextReport()
        text = parsed.text
        prefixes: dict[int, ChatMessage] = {}
        successes = 0
        for index, draft in enumerate(drafts):
            if index >= cfg.context_max_chunks:
                self.report.skipped = len(drafts) - index
                break
            window = draft.char_start // WINDOW_CHARS
            if window not in prefixes:
                start = window * WINDOW_CHARS
                excerpt = text[start : start + WINDOW_CHARS + WINDOW_SLACK]
                title = f"Title: {parsed.title}\n\n" if parsed.title and window == 0 else ""
                prefixes[window] = ChatMessage(
                    "system", SYSTEM_PREFIX.format(document=title + excerpt), cache=True
                )
            self.report.calls += 1
            try:
                reply = await self.llm.complete(
                    [prefixes[window], ChatMessage("user", USER_PROMPT.format(chunk=draft.text))],
                    temperature=0.0,
                    max_tokens=160,
                )
            except ProviderError as e:
                self.report.failed += 1
                log.warning("context generation failed for a chunk: %s", e)
                if (
                    self.report.failed >= MAX_FAILURES_BEFORE_ABORT
                    and self.report.failed > successes
                ):
                    raise
                continue
            successes += 1
            self.report.prompt_tokens += reply.usage.prompt_tokens
            self.report.completion_tokens += reply.usage.completion_tokens
            self.report.cached_tokens += reply.usage.cached_tokens
            draft.context = clean_context(reply.text)
        return drafts
