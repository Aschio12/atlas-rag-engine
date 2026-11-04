"""Conversations: CRUD, memory, rolling summary, streaming, privacy and feedback."""

import json
import uuid

import pytest
from answer_helpers import (
    REFUNDS,
    SpyLLM,
    add_doc,
    add_member,
    sse_events,
    use_llm,
)
from sqlalchemy import func, select

from app.db.models import Conversation, Message, Role
from app.providers.base import LLM
from app.providers.errors import ProviderError


def curl(space, cid=""):
    return f"/v1/workspaces/{space['ws']}/conversations" + (f"/{cid}" if cid else "")


async def new(client, space, title=None):
    body = {"title": title} if title else {}
    r = await client.post(curl(space), json=body, headers=space["headers"])
    assert r.status_code == 201, r.text
    return r.json()


async def say(client, space, cid, content, expect=200, **body):
    r = await client.post(
        curl(space, cid) + "/messages", json={"content": content, **body}, headers=space["headers"]
    )
    assert r.status_code == expect, r.text
    return r.json()


async def detail(client, space, cid, **params):
    r = await client.get(curl(space, cid), params=params, headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
async def ready(client, app, space):
    await add_doc(client, app, space)
    return space


# ---- the conversation itself ------------------------------------------------------------------


async def test_conversations_are_created_listed_renamed_and_deleted(client, space):
    a = await new(client, space)
    b = await new(client, space, "Refunds research")
    assert (
        a["title"] == "New conversation"
        and a["message_count"] == 0
        and b["title"] == "Refunds research"
    )
    listed = (await client.get(curl(space), headers=space["headers"])).json()
    assert [c["id"] for c in listed] == [b["id"], a["id"]]  # most recent first
    r = await client.patch(
        curl(space, a["id"]), json={"title": "Renamed"}, headers=space["headers"]
    )
    assert r.status_code == 200 and r.json()["title"] == "Renamed"
    assert (await client.delete(curl(space, a["id"]), headers=space["headers"])).status_code == 204
    assert [c["id"] for c in (await client.get(curl(space), headers=space["headers"])).json()] == [
        b["id"]
    ]
    assert (await client.get(curl(space, a["id"]), headers=space["headers"])).status_code == 404


@pytest.mark.parametrize("body", [{"title": ""}, {"title": "x" * 201}])
async def test_bad_titles_are_422(client, space, body):
    assert (await client.post(curl(space), json=body, headers=space["headers"])).status_code == 422
    c = await new(client, space)
    assert (
        await client.patch(curl(space, c["id"]), json=body, headers=space["headers"])
    ).status_code == 422


# ---- a turn -----------------------------------------------------------------------------------


async def test_a_message_gets_a_cited_answer_and_both_are_saved(client, ready):
    c = await new(client, ready)
    out = await say(client, ready, c["id"], "How long do refunds take?")
    answer = out["answer"]
    assert (
        answer["answer"].startswith("Refunds are issued within 14 days")
        and "[S1]" in answer["answer"]
    )
    assert out["conversation_id"] == c["id"] and out["message_id"] and out["user_message_id"]
    saved = await detail(client, ready, c["id"])
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]
    user, assistant = saved["messages"]
    assert user["content"] == "How long do refunds take?" and user["id"] == out["user_message_id"]
    assert assistant["id"] == out["message_id"] and assistant["content"] == answer["answer"]
    assert [c_["label"] for c_ in assistant["citations"]] == ["S1"]
    meta = assistant["metadata"]
    assert meta["abstained"] is False and meta["prompt"].startswith("answer@1#")
    assert meta["confidence"]["level"] in ("high", "medium", "low")
    assert meta["sources"][0]["label"] == "S1" and "text" not in meta["sources"][0]
    assert assistant["usage"]["prompt_tokens"] > 0 and assistant["retrieval_trace"] is None


async def test_the_trace_is_stored_and_returned_on_request(client, ready):
    c = await new(client, ready)
    await say(client, ready, c["id"], "How long do refunds take?")
    plain = await detail(client, ready, c["id"])
    full = await detail(client, ready, c["id"], include_trace=True)
    assert plain["messages"][1]["retrieval_trace"] is None
    trace = full["messages"][1]["retrieval_trace"]
    assert trace["schema_version"] == 1 and trace["stages"][0]["name"] == "scope"
    assert full["messages"][0]["retrieval_trace"] is None  # the question has none


async def test_the_first_question_names_an_untitled_conversation(client, ready):
    c = await new(client, ready)
    await say(client, ready, c["id"], "How long do refunds take?")
    assert (await detail(client, ready, c["id"]))["title"] == "How long do refunds take?"
    named = await new(client, ready, "Mine")
    await say(client, ready, named["id"], "How long do refunds take?")
    assert (await detail(client, ready, named["id"]))["title"] == "Mine"
    long = await new(client, ready)
    await say(client, ready, long["id"], "refunds " * 40)
    assert len((await detail(client, ready, long["id"]))["title"]) == 60


async def test_an_abstained_turn_is_saved_like_any_other(client, ready):
    c = await new(client, ready)
    out = await say(client, ready, c["id"], "Explain quantum chromodynamics and gluon confinement")
    assert out["answer"]["abstained"] is True
    saved = await detail(client, ready, c["id"])
    assert saved["messages"][1]["metadata"]["abstained"] is True
    assert saved["messages"][1]["metadata"]["related"]


async def test_a_failed_answer_saves_nothing(client, app, ready):
    class Down(LLM):
        name, model = "down", "d"

        async def complete(self, messages, **kw):
            raise ProviderError("down", "HTTP 503")

    use_llm(app, Down())
    c = await new(client, ready)
    await say(client, ready, c["id"], "How long do refunds take?", expect=502)
    saved = await detail(client, ready, c["id"])
    assert saved["messages"] == [] and saved["title"] == "New conversation"


@pytest.mark.parametrize("body", [{"content": ""}, {"content": " "}, {"content": "x" * 4001}, {}])
async def test_invalid_messages_are_422(client, ready, body):
    c = await new(client, ready)
    r = await client.post(curl(ready, c["id"]) + "/messages", json=body, headers=ready["headers"])
    assert r.status_code == 422


# ---- memory -----------------------------------------------------------------------------------


async def test_a_follow_up_is_sent_with_the_earlier_turns_and_rewritten(client, app, ready):
    llm = use_llm(app, SpyLLM())
    c = await new(client, ready)
    await say(client, ready, c["id"], "How long do refunds take?")
    out = await say(client, ready, c["id"], "And what is the refund payment method?")
    second = llm.answer_calls()[-1]
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert second[1].content == "How long do refunds take?"
    assert "[S1]" not in second[2].content  # earlier markers are not shown to the model again
    assert second[2].content.startswith("Refunds are issued within 14 days")
    stages = [s["name"] for s in out["answer"]["retrieval_trace"]["stages"]]
    assert "rewrite" in stages and out["answer"]["memory"]["turns"] == 2
    first = llm.answer_calls()[0]
    assert [m.role for m in first] == ["system", "user"]


async def test_old_messages_are_folded_into_a_rolling_summary(client, app, ready):
    app.state.settings.answer_history_messages = 2
    app.state.settings.answer_summarize_after = 2
    llm = use_llm(app, SpyLLM())
    c = await new(client, ready)
    for q in (
        "How long do refunds take?",
        "Where do orders ship from?",
        "What is the refund payment method?",
    ):
        await say(client, ready, c["id"], q)
    saved = await detail(client, ready, c["id"])
    assert saved["summarized_count"] > 0 and saved["summary"]
    assert "orders" in saved["summary"].lower()  # the fake summariser quotes what it folded
    await say(client, ready, c["id"], "Where do orders ship from, again?")
    last = llm.answer_calls()[-1]
    assert "## Conversation so far" in last[0].content and saved["summary"][:30] in last[0].content
    assert (
        len([m for m in last if m.role != "system"]) <= 3
    )  # the last two messages and the question


async def test_a_summary_failure_does_not_fail_or_lose_the_turn(client, app, ready):
    app.state.settings.answer_history_messages = 2
    app.state.settings.answer_summarize_after = 2

    class NoSummaries(SpyLLM):
        async def complete(self, messages, **kw):
            if "Reply with the updated summary only." in messages[-1].content:
                raise ProviderError("llm", "HTTP 500")
            return await super().complete(messages, **kw)

    use_llm(app, NoSummaries())
    c = await new(client, ready)
    for q in (
        "How long do refunds take?",
        "Where do orders ship from?",
        "What is the refund payment method?",
    ):
        await say(client, ready, c["id"], q)
    saved = await detail(client, ready, c["id"])
    assert len(saved["messages"]) == 6 and saved["summary"] == "" and saved["summarized_count"] == 0


async def test_workspace_facts_and_policy_apply_to_conversation_turns(client, app, ready):
    await client.post(
        f"/v1/workspaces/{ready['ws']}/memory",
        json={"text": "Answer in British English."},
        headers=ready["headers"],
    )
    await client.put(
        f"/v1/workspaces/{ready['ws']}/policy",
        json={"blocked_terms": ["salary"], "disclaimer": "Not legal advice."},
        headers=ready["headers"],
    )
    llm = use_llm(app, SpyLLM())
    c = await new(client, ready)
    out = await say(client, ready, c["id"], "How long do refunds take?")
    assert "- Answer in British English." in llm.answer_calls()[0][0].content
    assert (
        out["answer"]["answer"].endswith("Not legal advice.")
        and out["answer"]["memory"]["facts"] == 1
    )
    refused = await say(client, ready, c["id"], "What is the salary of the CEO?")
    assert refused["answer"]["refused"] is True
    saved = await detail(client, ready, c["id"])
    assert saved["messages"][-1]["metadata"]["refused"] is True  # recorded, not silently dropped


# ---- streaming --------------------------------------------------------------------------------


async def stream_turn(client, space, cid, content, **body):
    async with client.stream(
        "POST", curl(space, cid) + "/messages",
        json={"content": content, "stream": True, **body}, headers=space["headers"],
    ) as r:  # fmt: skip
        assert r.status_code == 200, await r.aread()
        return sse_events((await r.aread()).decode())


async def test_a_streamed_turn_is_saved_and_done_carries_the_ids(client, ready):
    c = await new(client, ready)
    events = await stream_turn(client, ready, c["id"], "How long do refunds take?")
    names = [n for n, _ in events]
    assert (
        names[0] == "retrieval" and names[-1] == "done" and "token" in names and "citation" in names
    )
    done = dict(events)["done"]
    assert done["conversation_id"] == c["id"] and done["message_id"] and done["user_message_id"]
    saved = await detail(client, ready, c["id"])
    assert [m["id"] for m in saved["messages"]] == [done["user_message_id"], done["message_id"]]
    assert (
        saved["messages"][1]["content"]
        == done["answer"]
        == "".join(d["text"] for n, d in events if n == "token")
    )


async def test_a_streamed_turn_uses_memory_too(client, app, ready):
    llm = use_llm(app, SpyLLM())
    c = await new(client, ready)
    await stream_turn(client, ready, c["id"], "How long do refunds take?")
    await stream_turn(client, ready, c["id"], "And what is the refund payment method?")
    assert [m.role for m in llm.answer_calls()[-1]] == ["system", "user", "assistant", "user"]


async def test_a_failure_mid_stream_saves_nothing(client, app, ready):
    class FailsMidStream(LLM):
        name, model = "flaky", "f"

        async def complete(self, messages, **kw):
            raise AssertionError("not used")

        async def stream(self, messages, **kw):
            from app.providers.base import StreamChunk

            yield StreamChunk("Refunds ")
            raise ProviderError("flaky", "connection reset")

    use_llm(app, FailsMidStream())
    c = await new(client, ready)
    events = await stream_turn(client, ready, c["id"], "How long do refunds take?")
    assert events[-1][0] == "error" and "done" not in [n for n, _ in events]
    assert (await detail(client, ready, c["id"]))["messages"] == []


async def test_streaming_validates_before_the_stream_starts(client, ready):
    c = await new(client, ready)
    r = await client.post(
        curl(ready, c["id"]) + "/messages",
        json={"content": "x", "stream": True, "prompt_version": 9},
        headers=ready["headers"],
    )
    assert r.status_code == 400
    r = await client.post(
        curl(ready, str(uuid.uuid4())) + "/messages",
        json={"content": "x", "stream": True},
        headers=ready["headers"],
    )
    assert r.status_code == 404


# ---- privacy ----------------------------------------------------------------------------------


async def test_conversations_are_private_to_their_owner_even_from_admins(client, app, ready):
    mine = await new(client, ready)
    await say(client, ready, mine["id"], "How long do refunds take?")
    member = await add_member(client, app, ready, "mia@example.com", Role.MEMBER)
    for method, suffix, body in (
        ("get", "", None), ("patch", "", {"title": "x"}), ("delete", "", None),
        ("post", "/messages", {"content": "hi"}),
    ):  # fmt: skip
        r = await client.request(
            method.upper(), curl(member, mine["id"]) + suffix, json=body, headers=member["headers"]
        )
        assert r.status_code == 404, (method, suffix)
    assert (await client.get(curl(member), headers=member["headers"])).json() == []
    # the owner (who is also an admin) sees only their own list; the member's is separate
    hers = await new(client, member)
    assert [c["id"] for c in (await client.get(curl(ready), headers=ready["headers"])).json()] == [
        mine["id"]
    ]
    assert (await client.get(curl(ready, hers["id"]), headers=ready["headers"])).status_code == 404


async def test_other_workspaces_and_anonymous_callers_get_nothing(client, ready, signup):
    c = await new(client, ready)
    stranger = await signup("eve@example.com")
    assert (await client.get(curl(ready, c["id"]), headers=stranger)).status_code == 404
    assert (await client.post(curl(ready), json={}, headers=stranger)).status_code == 404
    assert (await client.get(curl(ready))).status_code == 401
    assert (
        await client.get(curl(ready, "not-a-uuid"), headers=ready["headers"])
    ).status_code == 422


async def test_viewers_can_converse(client, app, ready):
    viewer = await add_member(client, app, ready, "vera@example.com", Role.VIEWER)
    c = await new(client, viewer)
    out = await say(client, viewer, c["id"], "How long do refunds take?")
    assert out["answer"]["answer"]


# ---- feedback and cleanup ---------------------------------------------------------------------


async def test_feedback_can_be_given_changed_and_cleared(client, ready):
    c = await new(client, ready)
    out = await say(client, ready, c["id"], "How long do refunds take?")
    url = curl(ready, c["id"]) + f"/messages/{out['message_id']}/feedback"
    for value, stored in ((1, 1), (-1, -1), (0, None)):
        r = await client.post(url, json={"value": value}, headers=ready["headers"])
        assert r.status_code == 200 and r.json()["feedback"] == stored
    assert (await client.post(url, json={"value": 2}, headers=ready["headers"])).status_code == 422
    on_question = curl(ready, c["id"]) + f"/messages/{out['user_message_id']}/feedback"
    assert (
        await client.post(on_question, json={"value": 1}, headers=ready["headers"])
    ).status_code == 404
    saved = await detail(client, ready, c["id"])
    assert saved["messages"][1]["feedback"] is None


async def test_deleting_a_conversation_or_workspace_deletes_its_messages(client, app, ready):
    c = await new(client, ready)
    await say(client, ready, c["id"], "How long do refunds take?")

    async def counts():
        async with app.state.sessionmaker() as s:
            n = (await s.execute(select(func.count()).select_from(Message))).scalar_one()
            k = (await s.execute(select(func.count()).select_from(Conversation))).scalar_one()
            return n, k

    assert await counts() == (2, 1)
    assert (await client.delete(curl(ready, c["id"]), headers=ready["headers"])).status_code == 204
    assert await counts() == (0, 0)
    c2 = await new(client, ready)
    await say(client, ready, c2["id"], "How long do refunds take?")
    assert (
        await client.delete(f"/v1/workspaces/{ready['ws']}", headers=ready["headers"])
    ).status_code == 204
    assert await counts() == (0, 0)


async def test_the_list_counts_messages_and_orders_by_activity(client, ready):
    a, b = await new(client, ready, "A"), await new(client, ready, "B")
    await say(client, ready, a["id"], "How long do refunds take?")  # A is now the most recent
    listed = (await client.get(curl(ready), headers=ready["headers"])).json()
    assert [(c["title"], c["message_count"]) for c in listed] == [("A", 2), ("B", 0)]
    page = (await client.get(curl(ready) + "?limit=1&offset=1", headers=ready["headers"])).json()
    assert [c["id"] for c in page] == [b["id"]]


_ = (json, REFUNDS)
