import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.db.models import Conversation, MemoryFact, Message, MessageRole
from app.generate.memory import (
    FACTS_CHARS,
    facts_for_prompt,
    fold,
    foldable,
    maybe_summarise,
    recent,
    visible_text,
    window,
)
from app.providers.errors import ProviderError
from app.providers.fake import FakeLLM


def msg(role, content):
    return Message(id=uuid.uuid4(), role=MessageRole(role), content=content)


def chat(n):
    """n alternating messages: user 0, assistant 1, user 2, ..."""
    return [msg("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(n)]


def contents(turns):
    return [t.content for t in turns]


# ---- the verbatim window ----------------------------------------------------------------------


def test_the_window_is_the_latest_messages_in_order():
    turns = window(chat(10), 0, max_messages=4, max_tokens=1000)
    assert contents(turns) == ["message 6", "message 7", "message 8", "message 9"]
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]


def test_summarised_messages_are_not_sent_again():
    turns = window(chat(10), 8, max_messages=6, max_tokens=1000)
    assert contents(turns) == ["message 8", "message 9"]


def test_a_window_never_starts_with_an_answer_whose_question_was_cut():
    turns = window(chat(10), 0, max_messages=5, max_tokens=1000)  # would start at message 5
    assert contents(turns)[0] == "message 6" and turns[0].role == "user"


def test_the_token_limit_drops_the_oldest_first():
    long = [msg("user", "word " * 400), msg("assistant", "ok"), msg("user", "next question")]
    turns = window(long, 0, max_messages=6, max_tokens=50)
    assert contents(turns) == ["next question"]
    assert window(long, 0, max_messages=6, max_tokens=0) == []


def test_zero_messages_means_no_history():
    assert window(chat(6), 0, max_messages=0, max_tokens=1000) == []
    assert recent(chat(6), 0, 0) == []


def test_system_messages_are_never_sent_as_turns():
    messages = [msg("system", "note"), *chat(2)]
    assert contents(window(messages, 0, 6, 1000)) == ["message 0", "message 1"]


def test_answers_lose_their_citation_markers_but_questions_keep_brackets():
    answer = msg("assistant", "Refunds take 14 days [S1][S2]. Orders ship fast [S3].")
    assert visible_text(answer) == "Refunds take 14 days. Orders ship fast."
    assert visible_text(msg("user", "What is [S1] in the report?")) == "What is [S1] in the report?"


def test_very_long_messages_are_clipped():
    assert len(visible_text(msg("user", "x" * 5000))) == 2000


# ---- folding into the summary -----------------------------------------------------------------


def test_only_old_unsummarised_messages_are_foldable():
    messages = chat(12)
    assert [m.content for m in foldable(messages, 0, max_messages=4)] == [
        f"message {i}" for i in range(8)
    ]
    assert [m.content for m in foldable(messages, 6, max_messages=4)] == ["message 6", "message 7"]
    assert foldable(messages, 0, max_messages=20) == []
    assert len(foldable(messages, 0, max_messages=0)) == 12


class Summariser(FakeLLM):
    def __init__(self, reply):
        super().__init__(reply=reply)


def conversation(summary="", summarized=0):
    return Conversation(id=uuid.uuid4(), summary=summary, summarized_count=summarized)


async def test_folding_asks_the_model_with_the_old_summary_and_the_new_messages():
    llm = Summariser("User asks about refunds; 14 days.")
    c = conversation(summary="Earlier: shipping.")
    out = await fold(llm, c, chat(2))
    assert out == "User asks about refunds; 14 days."
    prompt = llm.calls[0][0].content
    assert "Earlier: shipping." in prompt and "user: message 0" in prompt
    assert "assistant: message 1" in prompt and "at most 150 words" in prompt
    fresh = Summariser("x")
    await fold(fresh, conversation(), chat(2))
    assert "(none yet)" in fresh.calls[0][0].content


async def test_a_summary_is_tidied_and_bounded():
    llm = Summariser("line one\n\n\n\nline two " + "y" * 3000)
    out = await fold(llm, conversation(), chat(2))
    assert "\n\n\n" not in out and len(out) == 2000
    assert await fold(Summariser("   "), conversation(), chat(2)) is None


async def test_summarising_waits_until_enough_old_messages_have_piled_up():
    llm = Summariser("summary")
    c = conversation()
    assert await maybe_summarise(llm, c, chat(8), max_messages=6, after=4) is False  # 2 old
    assert llm.calls == [] and c.summary == "" and c.summarized_count == 0
    assert await maybe_summarise(llm, c, chat(12), max_messages=6, after=4) is True  # 6 old
    assert c.summary == "summary" and c.summarized_count == 6


async def test_the_next_fold_continues_from_where_the_last_one_stopped():
    llm = Summariser("second summary")
    c = conversation(summary="first summary", summarized=6)
    assert await maybe_summarise(llm, c, chat(16), max_messages=6, after=4) is True
    assert c.summarized_count == 10 and c.summary == "second summary"
    prompt = llm.calls[0][0].content
    assert "first summary" in prompt and "message 6" in prompt and "message 5" not in prompt


class Down(FakeLLM):
    async def complete(self, messages, **kw):
        raise ProviderError("down", "HTTP 503")


async def test_a_failed_summary_changes_nothing_so_the_next_turn_retries():
    c = conversation(summary="keep me", summarized=2)
    assert await maybe_summarise(Down(), c, chat(14), max_messages=6, after=4) is False
    assert c.summary == "keep me" and c.summarized_count == 2


# ---- long-term facts --------------------------------------------------------------------------


def fact(text, minutes=0, enabled=True):
    return MemoryFact(
        id=uuid.uuid4(), text=text, enabled=enabled,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
    )  # fmt: skip


def test_only_enabled_facts_are_used_oldest_first():
    facts = [fact("b", 2), fact("a", 1), fact("off", 3, enabled=False)]
    assert facts_for_prompt(facts, 10) == ["a", "b"]


def test_the_number_and_total_size_of_facts_are_bounded():
    facts = [fact(f"fact {i}", i) for i in range(5)]
    assert facts_for_prompt(facts, 2) == ["fact 0", "fact 1"]
    big = [fact("x" * 3000, 1), fact("y" * 3000, 2), fact("z", 3)]
    assert facts_for_prompt(big, 10) == ["x" * 3000]  # the second would exceed the budget
    assert FACTS_CHARS == 4000 and facts_for_prompt([], 5) == []


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_conversations_have_nothing_to_fold(n):
    assert foldable(chat(n), 0, 6) == []
