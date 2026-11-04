"""Query transformation (Phase 3.5): turn the user's words into better searches.

Each technique is a separate switch, because each has a cost (one LLM call) and a failure mode:

* **rewrite**: resolve references to the conversation ("what about its price?") so the search
  sees a standalone question. It replaces the query. Needs `history`.
* **multi_query**: a few differently worded queries; their rankings are fused, which helps when
  the user's vocabulary differs from the documents'.
* **hyde**: embed a *hypothetical answer* instead of the question; answers look more like
  passages than questions do. Dense search only.
* **step_back**: also search for a broader question that retrieves background.
* **decompose**: split a multi-part question into sub-questions and search each, so every part
  of the question gets passages (the fused result is then diverse by construction).

The model's output is used only as search text. It cannot change filters, scopes or anything
else, so a question crafted to hijack it (prompt injection) can at worst make the search worse;
the question is also fenced in `<question>` tags and the replies are validated strictly. A
failed or unusable reply is skipped with a warning and the search carries on with what it has.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.providers.base import LLM, ChatMessage, Usage
from app.providers.errors import ProviderError

MAX_QUERY_CHARS = 500
MIN_QUERY_CHARS = 3
MAX_PASSAGE_CHARS = 1500
MAX_VARIANTS = 12
HISTORY_TURNS = 6
HISTORY_TURN_CHARS = 1000


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class TransformOptions(BaseModel):
    rewrite: bool = False
    multi_query: bool = False
    multi_query_count: int = Field(3, ge=1, le=5)
    hyde: bool = False
    step_back: bool = False
    decompose: bool = False
    max_sub_questions: int = Field(4, ge=1, le=6)

    @property
    def any(self) -> bool:
        return any((self.rewrite, self.multi_query, self.hyde, self.step_back, self.decompose))


@dataclass(frozen=True)
class Variant:
    label: str  # "multi_query:1", "hyde", "step_back", "sub:2"
    text: str
    kinds: tuple[str, ...]  # channels that search with this text: "dense", "sparse"


@dataclass
class TransformResult:
    primary: str  # the query the main channels use (the rewrite, if there was one)
    variants: list[Variant] = field(default_factory=list)  # extra searches
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    took_ms: dict[str, float] = field(default_factory=dict)


REWRITE = """Rewrite the user's latest question so it can be understood without the conversation.
Replace pronouns and references ("it", "that", "the second one") with what they refer to. Keep the
meaning, the language and the level of detail. Do not answer it. If it already stands alone,
return it unchanged.

Conversation:
{history}

<question>{query}</question>

Reply with JSON only: {{"query": "..."}}"""

MULTI_QUERY = """Write {n} different search queries that would find passages useful for answering
the question. Vary the vocabulary and the angle (synonyms, related terms, a more specific or a
more general phrasing). Do not repeat the question itself.

<question>{query}</question>

Reply with JSON only: {{"queries": ["...", "..."]}}"""

HYDE = """Write a short passage (two to four sentences) that would plausibly answer the question,
in the style of a documentation page. It does not need to be correct: it is only used to find
real passages that look like it.

<question>{query}</question>

Reply with JSON only: {{"passage": "..."}}"""

STEP_BACK = """Write one broader, more general question whose answer gives the background needed
to answer the specific question below.

<question>{query}</question>

Reply with JSON only: {{"query": "..."}}"""

DECOMPOSE = """If the question has several parts, or needs several separate facts, split it into at
most {n} self-contained sub-questions (each must make sense on its own). If it is a single simple
question, return an empty list.

<question>{query}</question>

Reply with JSON only: {{"questions": ["...", "..."]}}"""


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the reply")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def _clean(value: object, limit: int = MAX_QUERY_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not MIN_QUERY_CHARS <= len(text) <= limit or not any(ch.isalnum() for ch in text):
        return None  # too short or long, or only punctuation such as the "..." of an echoed prompt
    return text


def _clean_list(values: object, limit: int, exclude: str) -> list[str]:
    """Valid, distinct texts (not `exclude` itself), at most `limit`."""
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen = {exclude.casefold()}
    for v in values:
        if (text := _clean(v)) and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out[:limit]


def format_history(history: list[Turn]) -> str:
    turns = history[-HISTORY_TURNS:]
    return "\n".join(f"{t.role}: {t.content[:HISTORY_TURN_CHARS]}" for t in turns)


class QueryTransformer:
    """One method per technique; each returns validated text and records its LLM usage."""

    def __init__(self, llm: LLM):
        self._llm = llm
        self.calls = 0
        self.usage = Usage()

    async def _ask(self, prompt: str, *, temperature: float, max_tokens: int = 300) -> dict:
        resp = await self._llm.complete(
            [ChatMessage("user", prompt)], temperature=temperature, max_tokens=max_tokens
        )
        self.calls += 1
        self.usage = Usage(
            self.usage.prompt_tokens + resp.usage.prompt_tokens,
            self.usage.completion_tokens + resp.usage.completion_tokens,
            self.usage.cached_tokens + resp.usage.cached_tokens,
        )
        try:
            return _json_object(resp.text)
        except (ValueError, TypeError) as exc:
            raise ProviderError("llm", f"unusable reply: {exc}") from exc

    async def rewrite(self, query: str, history: list[Turn]) -> str:
        data = await self._ask(
            REWRITE.format(history=format_history(history), query=query), temperature=0.0
        )
        if (text := _clean(data.get("query"))) is None:
            raise ProviderError("llm", "the rewrite was empty or unusable")
        return text

    async def multi_query(self, query: str, n: int) -> list[str]:
        data = await self._ask(MULTI_QUERY.format(n=n, query=query), temperature=0.7)
        return _clean_list(data.get("queries"), n, query)

    async def hyde(self, query: str) -> str:
        data = await self._ask(HYDE.format(query=query), temperature=0.4, max_tokens=400)
        if (text := _clean(data.get("passage"), MAX_PASSAGE_CHARS)) is None:
            raise ProviderError("llm", "the hypothetical passage was empty or unusable")
        return text

    async def step_back(self, query: str) -> str:
        data = await self._ask(STEP_BACK.format(query=query), temperature=0.2)
        if (text := _clean(data.get("query"))) is None:
            raise ProviderError("llm", "the broader question was empty or unusable")
        return text

    async def decompose(self, query: str, n: int) -> list[str]:
        data = await self._ask(DECOMPOSE.format(n=n, query=query), temperature=0.0)
        return _clean_list(data.get("questions"), n, query)


async def transform_query(
    llm: LLM,
    query: str,
    history: list[Turn],
    options: TransformOptions,
    *,
    mode: str,
) -> TransformResult:
    """Run the enabled transformations. `mode` decides which channels a variant can use."""
    result = TransformResult(primary=query)
    tf = QueryTransformer(llm)
    both = (
        ("sparse",) if mode == "sparse" else ("dense",) if mode == "dense" else ("dense", "sparse")
    )

    async def timed(name: str, coro):
        started = time.perf_counter()
        try:
            return await coro
        except ProviderError as exc:
            result.warnings.append(f"Query transform '{name}' failed ({exc}); skipped")
            return None
        finally:
            result.took_ms[name] = round((time.perf_counter() - started) * 1000, 2)

    if options.rewrite:
        if not history:
            result.skipped["rewrite"] = "no conversation history was sent"
        elif (rewritten := await timed("rewrite", tf.rewrite(query, history))) is not None:
            result.primary = rewritten

    primary = result.primary
    jobs: dict[str, object] = {}
    if options.multi_query:
        jobs["multi_query"] = timed(
            "multi_query", tf.multi_query(primary, options.multi_query_count)
        )
    if options.hyde:
        if mode == "sparse":
            result.skipped["hyde"] = "a hypothetical answer needs embeddings (dense or hybrid mode)"
        else:
            jobs["hyde"] = timed("hyde", tf.hyde(primary))
    if options.step_back:
        jobs["step_back"] = timed("step_back", tf.step_back(primary))
    if options.decompose:
        jobs["decompose"] = timed("decompose", tf.decompose(primary, options.max_sub_questions))
    outputs = dict(zip(jobs, await asyncio.gather(*jobs.values()), strict=True))

    seen = {primary.casefold()}

    def add(label: str, text: str, kinds: tuple[str, ...]) -> None:
        if text.casefold() in seen or len(result.variants) >= MAX_VARIANTS:
            return
        seen.add(text.casefold())
        result.variants.append(Variant(label, text, kinds))

    for i, text in enumerate(outputs.get("multi_query") or [], start=1):
        add(f"multi_query:{i}", text, both)
    if outputs.get("step_back"):
        add("step_back", outputs["step_back"], both)
    if "decompose" in outputs:
        subs = outputs["decompose"] or []
        if not subs and outputs["decompose"] is not None:
            result.skipped["decompose"] = "the question is a single question"
        for i, text in enumerate(subs, start=1):
            add(f"sub:{i}", text, both)
    if outputs.get("hyde"):
        add("hyde", outputs["hyde"], ("dense",))  # the passage is matched by meaning, not words
    result.llm_calls, result.usage = tf.calls, tf.usage
    return result
