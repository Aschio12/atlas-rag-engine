"""Streaming answers as server-sent events."""

import json

import pytest
from answer_helpers import (
    REFUNDS,
    SpyLLM,
    add_doc,
    answer_url,
    ask,
    stream_answer,
    use_llm,
)

from app.providers.base import LLM
from app.providers.errors import ProviderError


@pytest.fixture
async def ready(client, app, space):
    doc = await add_doc(client, app, space)
    return {**space, "doc": doc}


def names(events):
    return [n for n, _ in events]


def tokens(events):
    return "".join(d["text"] for n, d in events if n == "token")


async def test_events_arrive_in_order_and_the_text_adds_up(client, ready):
    events, text = await stream_answer(client, ready, "How long do refunds take?")
    assert names(events)[0] == "retrieval" and names(events)[-1] == "done"
    assert names(events).count("done") == 1 and "error" not in names(events)
    done = dict(events)["done"]
    assert tokens(events) == done["answer"]  # nothing was changed after streaming
    assert text.startswith(": stream open")


async def test_the_retrieval_event_comes_first_with_sources_and_confidence(client, ready):
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    retrieval = dict(events)["retrieval"]
    assert retrieval["sources"][0]["label"] == "S1" and REFUNDS in retrieval["sources"][0]["text"]
    assert retrieval["confidence"]["level"] != "none" and retrieval["confidence"]["score"] > 0.15
    assert (
        retrieval["hits"]
        and retrieval["warnings"] == []
        and retrieval["guardrails"]["injection"] == []
    )


async def test_a_citation_event_names_the_source_the_first_time_it_is_cited(client, ready):
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    cites = [d for n, d in events if n == "citation"]
    assert [c["label"] for c in cites] == ["S1"] and cites[0]["title"] == "handbook"
    assert cites[0]["passages"] and cites[0]["char_end"] > cites[0]["char_start"]
    order = names(events)
    assert order.index("retrieval") < order.index("citation") < order.index("done")


async def test_done_carries_the_same_payload_as_a_one_shot_answer(client, ready):
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    streamed = dict(events)["done"]
    shot = await ask(client, ready, "How long do refunds take?")
    for key in ("answer", "abstained", "refused", "citations", "prompt", "model"):
        assert streamed[key] == shot[key], key
    assert streamed["confidence"]["level"] == shot["confidence"]["level"]


async def test_text_is_streamed_in_pieces_when_the_model_streams(client, app, ready):
    use_llm(app, SpyLLM(reply=f"{REFUNDS[:-1]} [S1]. It applies to every order [S1]."))
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    pieces = [d["text"] for n, d in events if n == "token"]
    assert len(pieces) > 5 and "".join(pieces) == dict(events)["done"]["answer"]


async def test_an_abstention_streams_no_text(client, ready):
    events, _ = await stream_answer(
        client, ready, "Explain quantum chromodynamics and gluon confinement"
    )
    assert names(events) == ["retrieval", "done"]
    done = dict(events)["done"]
    assert done["abstained"] is True and done["reason"] == "low_confidence"


async def test_a_not_found_from_the_model_is_never_streamed_as_text(client, app, ready):
    use_llm(app, SpyLLM(reply="NOT_FOUND The documents do not say how refunds are taxed."))
    events, text = await stream_answer(client, ready, "How long do refunds take?")
    assert "token" not in names(events) and "NOT_FOUND" not in text
    done = dict(events)["done"]
    assert done["abstained"] is True and done["reason"] == "model_not_found"
    assert "taxed" in done["answer"]


async def test_invalid_citations_are_removed_from_done_but_the_stream_showed_them(
    client, app, ready
):
    use_llm(app, SpyLLM(reply=f"{REFUNDS[:-1]} [S1][S9]."))
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    done = dict(events)["done"]
    assert "[S9]" in tokens(events) and "[S9]" not in done["answer"]
    assert done["grounding"]["invalid"] == ["S9"]
    assert [d["label"] for n, d in events if n == "citation"] == ["S1"]  # only valid ones


class FailsMidStream(LLM):
    name, model = "flaky", "f"

    async def complete(self, messages, **kw):
        raise AssertionError("not used")

    async def stream(self, messages, **kw):
        from app.providers.base import StreamChunk

        yield StreamChunk("Refunds are issued ")
        raise ProviderError("flaky", "connection reset")


async def test_a_failure_mid_stream_ends_with_an_error_event_and_no_done(client, app, ready):
    use_llm(app, FailsMidStream())
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    assert names(events)[-1] == "error" and "done" not in names(events)
    error = events[-1][1]
    assert error["code"] == "provider_error" and error["provider"] == "flaky"
    assert "connection reset" in error["message"]


async def test_a_failure_before_any_text_is_also_an_error_event(client, app, ready):
    class Down(LLM):
        name, model = "down", "d"

        async def complete(self, messages, **kw):
            raise ProviderError("down", "HTTP 503", 503)

    use_llm(app, Down())
    events, _ = await stream_answer(client, ready, "How long do refunds take?")
    assert names(events) == ["retrieval", "error"] and events[-1][1]["status"] == 503


async def test_a_leaked_reference_stops_the_stream_and_is_replaced(client, app, ready):
    from answer_helpers import LeakyLLM

    use_llm(app, LeakyLLM())
    events, text = await stream_answer(client, ready, "How long do refunds take?")
    done = dict(events)["done"]
    assert done["refused"] is True and done["reason"] == "prompt_leak"
    assert done["answer"] == "I can't share that."
    assert text.count("CANARY-") <= 1 and "CANARY-" not in done["answer"]


CONTACT = "# Contact\n\nRefunds are handled by ada@example.org.\n"


async def test_when_answers_must_be_checked_the_text_is_released_after_the_checks(
    client, app, space
):
    await add_doc(client, app, space, CONTACT, "contact.md")
    use_llm(app, SpyLLM(reply="Write to ada@example.org about refunds [S1]."))
    events, text = await stream_answer(client, space, "Who handles refunds?", pii="all")
    leaked = [e for e in text.split("\n\n") if "ada@example.org" in e]
    assert not leaked, [
        e[:200] + " ... " + e[e.index("ada@example.org") - 60 : e.index("ada@example.org") + 40]
        for e in leaked
    ]
    assert (
        tokens(events) == dict(events)["done"]["answer"] == "Write to [EMAIL_1] about refunds [S1]."
    )
    assert [d["label"] for n, d in events if n == "citation"] == ["S1"]
    order = names(events)
    assert order.index("retrieval") < order.index("token") < order.index("done")


async def test_streaming_validates_the_request_before_the_stream_starts(client, ready):
    r = await client.post(
        answer_url(ready),
        json={"question": "x", "stream": True, "prompt_version": 9},
        headers=ready["headers"],
    )
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/problem+json")
    r = await client.post(answer_url(ready), json={"question": "x", "stream": True})
    assert r.status_code == 401


async def test_the_stream_is_server_sent_events(client, ready):
    async with client.stream(
        "POST",
        answer_url(ready),
        json={"question": "refunds", "stream": True},
        headers=ready["headers"],
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
        body = (await r.aread()).decode()
    for block in body.strip().split("\n\n")[1:]:
        assert block.startswith("event: ") and "\ndata: " in block
        json.loads(block.split("\ndata: ", 1)[1])
