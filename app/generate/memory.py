"""Conversation memory (Phase 4.4): what the assistant remembers within and across conversations.

**Short-term**: a conversation sends the model its most recent messages verbatim (a window limited
by message count and by tokens) plus a *rolling summary* of everything older. Once enough old
messages pile up they are folded into the summary by the model and not sent again, so a long
conversation costs a roughly constant amount per turn instead of growing without bound, and
early decisions ("we are only talking about the EU contract") survive in the summary.

**Long-term**: workspace *facts* that people write and can edit (`MemoryFact`): "answer in British
English", "our fiscal year starts in April". They are added to the system prompt of every
conversation in the workspace. Nothing is learned silently: the assistant remembers only what is
listed there, which anyone with write access can read, correct or switch off.

Old assistant messages are shown to the model without their `[S1]` markers: those ids belonged to
the context of that earlier turn, and keeping them invites the model to reuse them for the new
sources.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.db.models import Conversation, MemoryFact, Message, MessageRole
from app.generate import prompts
from app.generate.citations import normalise
from app.ingest.chunking.base import estimate_tokens
from app.providers.base import LLM, ChatMessage
from app.providers.errors import ProviderError
from app.retrieve.transform import Turn

log = logging.getLogger("atlas.memory")

TURN_CHARS = 2000  # a Turn holds at most this much
SUMMARY_WORDS = 150
SUMMARY_CHARS = 2000
FOLD_CHARS = 700  # how much of each folded message the summariser reads
FACTS_CHARS = 4000  # facts beyond this many characters are left out of the prompt


def visible_text(message: Message) -> str:
    """What the model sees of an earlier message: assistant answers lose their markers."""
    text = message.content
    if message.role == MessageRole.ASSISTANT:
        text = " ".join(normalise(text, set())[0].split())  # no valid labels: all markers go
    return text[:TURN_CHARS]


def recent(messages: Sequence[Message], summarized: int, max_messages: int) -> list[Message]:
    """The messages that are sent verbatim: after the summarised ones, at most `max_messages`."""
    live = [m for m in messages[summarized:] if m.role in (MessageRole.USER, MessageRole.ASSISTANT)]
    return live[-max_messages:] if max_messages > 0 else []


def window(
    messages: Sequence[Message], summarized: int, max_messages: int, max_tokens: int
) -> list[Turn]:
    """The recent turns for the prompt, oldest first, within both limits. A window never starts
    with an assistant message (an answer without its question confuses more than it helps)."""
    chosen = recent(messages, summarized, max_messages)
    turns = [(m, visible_text(m)) for m in chosen]
    total = sum(estimate_tokens(t) for _, t in turns)
    while turns and (total > max_tokens or turns[0][0].role == MessageRole.ASSISTANT):
        total -= estimate_tokens(turns[0][1])
        turns.pop(0)
    return [Turn(role=m.role.value, content=text) for m, text in turns if text.strip()]


def foldable(messages: Sequence[Message], summarized: int, max_messages: int) -> list[Message]:
    """Messages old enough to be folded into the summary: not yet summarised, not in the window."""
    live = [m for m in messages[summarized:] if m.role in (MessageRole.USER, MessageRole.ASSISTANT)]
    if max_messages <= 0:
        return live
    return live[:-max_messages] if len(live) > max_messages else []


async def fold(llm: LLM, conversation: Conversation, old: Sequence[Message]) -> str | None:
    """The updated summary after reading `old`, or None if the model could not produce one."""
    prompt = prompts.get("summarize")
    lines = [f"{m.role.value}: {visible_text(m)[:FOLD_CHARS]}" for m in old]
    text = prompt.render(
        max_words=str(SUMMARY_WORDS),
        summary=conversation.summary or "(none yet)",
        messages="\n".join(lines),
    )
    try:
        response = await llm.complete([ChatMessage("user", text)], temperature=0.0, max_tokens=400)
    except ProviderError as exc:
        log.warning("could not summarise a conversation: %s", exc)
        return None
    summary = re.sub(r"\n{3,}", "\n\n", response.text.strip())
    return summary[:SUMMARY_CHARS] if summary else None


async def maybe_summarise(
    llm: LLM,
    conversation: Conversation,
    messages: Sequence[Message],
    *,
    max_messages: int,
    after: int,
) -> bool:
    """Fold old messages into the conversation's summary once `after` of them have piled up.
    Returns whether the summary changed. Failures leave things as they were: the next turn
    tries again."""
    old = foldable(messages, conversation.summarized_count, max_messages)
    if len(old) < after:
        return False
    summary = await fold(llm, conversation, old)
    if summary is None:
        return False
    conversation.summary = summary
    conversation.summarized_count += len(old)
    return True


def facts_for_prompt(facts: Sequence[MemoryFact], limit: int) -> list[str]:
    """Enabled facts, oldest first, as many as fit the limits."""
    chosen, used = [], 0
    for fact in sorted((f for f in facts if f.enabled), key=lambda f: f.created_at)[:limit]:
        if used + len(fact.text) > FACTS_CHARS:
            break
        chosen.append(fact.text)
        used += len(fact.text)
    return chosen
