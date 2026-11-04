import json

import httpx
import pytest

from app.providers.anthropic import AnthropicLLM
from app.providers.base import LLM, ChatMessage, LLMResponse, StreamChunk, Usage
from app.providers.errors import ProviderError
from app.providers.fake import FakeLLM
from app.providers.ollama import OllamaLLM
from app.providers.openai import OpenAILLM

MSGS = [ChatMessage("system", "be brief"), ChatMessage("user", "hi")]


async def collect(llm, messages=MSGS):
    chunks = [c async for c in llm.stream(messages)]
    return "".join(c.text for c in chunks), [c.usage for c in chunks if c.usage is not None]


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sse(*events: dict | str) -> bytes:
    lines = []
    for e in events:
        lines.append(f"data: {e if isinstance(e, str) else json.dumps(e)}\n")
    return "\n".join(lines).encode()


# ---- the default: any LLM can stream ---------------------------------------------------------


class Plain(LLM):
    name, model = "plain", "p"

    async def complete(self, messages, *, temperature=0.2, max_tokens=1024):
        return LLMResponse("whole answer", "p", Usage(3, 2))


async def test_a_provider_without_streaming_yields_the_whole_answer_then_the_usage():
    chunks = [c async for c in Plain().stream(MSGS)]
    assert chunks == [StreamChunk("whole answer"), StreamChunk(usage=Usage(3, 2))]


async def test_the_fake_llm_streams_word_by_word_with_usage():
    text, usages = await collect(FakeLLM(reply="one two  three\nfour"))
    assert text == "one two  three\nfour" and len(usages) == 1 and usages[0].completion_tokens == 4
    pieces = [c.text async for c in FakeLLM(reply="one two three").stream(MSGS) if c.text]
    assert pieces == ["one ", "two ", "three"]


# ---- OpenAI-compatible ------------------------------------------------------------------------


async def test_openai_streams_deltas_and_reports_usage():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(
            200,
            content=sse(
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2,
                                          "prompt_tokens_details": {"cached_tokens": 4}}},
                "[DONE]",
            ),
        )  # fmt: skip

    llm = OpenAILLM("sk-test", "gpt-x", client=client(handler))
    text, usages = await collect(llm)
    assert text == "Hello" and usages == [Usage(9, 2, 4)]
    assert seen["body"]["stream"] is True and seen["body"]["stream_options"] == {
        "include_usage": True
    }
    assert seen["auth"] == "Bearer sk-test"


async def test_openai_http_errors_and_bad_stream_data_become_provider_errors():
    with pytest.raises(ProviderError, match="HTTP 429"):
        await collect(
            OpenAILLM("k", "m", client=client(lambda r: httpx.Response(429, text="slow down")))
        )
    with pytest.raises(ProviderError, match="unexpected stream data"):
        await collect(
            OpenAILLM(
                "k", "m", client=client(lambda r: httpx.Response(200, content=b"data: {oops\n"))
            )
        )
    boom = sse({"error": {"message": "overloaded"}})
    with pytest.raises(ProviderError, match="stream error"):
        await collect(
            OpenAILLM("k", "m", client=client(lambda r: httpx.Response(200, content=boom)))
        )


async def test_a_connection_that_drops_is_a_provider_error():
    def handler(req):
        raise httpx.ReadError("connection reset")

    with pytest.raises(ProviderError, match="request failed"):
        await collect(OpenAILLM("k", "m", client=client(handler)))


# ---- Ollama -----------------------------------------------------------------------------------


async def test_ollama_streams_ndjson_and_reports_counts():
    body = "\n".join(
        json.dumps(e)
        for e in (
            {"message": {"content": "Hi "}, "done": False},
            {"message": {"content": "there"}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 7, "eval_count": 3},
        )
    )
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=body.encode())

    text, usages = await collect(OllamaLLM("http://ollama:11434", "llama", client=client(handler)))
    assert text == "Hi there" and usages == [Usage(7, 3)]
    assert seen["body"]["stream"] is True and seen["body"]["options"]["num_predict"] == 1024


async def test_ollama_errors_are_provider_errors():
    llm = OllamaLLM(
        "http://o", "m", client=client(lambda r: httpx.Response(404, text="no such model"))
    )
    with pytest.raises(ProviderError, match="HTTP 404"):
        await collect(llm)
    err = httpx.Response(200, content=json.dumps({"error": "model crashed"}).encode())
    with pytest.raises(ProviderError, match="stream error"):
        await collect(OllamaLLM("http://o", "m", client=client(lambda r: err)))


# ---- Anthropic --------------------------------------------------------------------------------


def anthropic_events(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


async def test_anthropic_streams_text_deltas_and_usage():
    seen = {}

    def handler(req):
        seen["body"], seen["key"] = json.loads(req.content), req.headers["x-api-key"]
        return httpx.Response(
            200,
            content=anthropic_events(
                {"type": "message_start", "message": {"usage": {
                    "input_tokens": 12, "cache_read_input_tokens": 5, "output_tokens": 1}}},
                {"type": "content_block_start", "index": 0},
                {"type": "ping"},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Answer "}},
                {"type": "content_block_delta",
                 "delta": {"type": "input_json_delta", "partial_json": "{"}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "here"}},
                {"type": "message_delta", "usage": {"output_tokens": 6}},
                {"type": "message_stop"},
            ),
        )  # fmt: skip

    llm = AnthropicLLM("key", "claude-x", client=client(handler))
    text, usages = await collect(llm)
    assert text == "Answer here" and usages == [Usage(12, 6, 5)]
    assert seen["body"]["stream"] is True and seen["body"]["system"] == "be brief"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}] and seen["key"] == "key"


async def test_anthropic_stream_errors_are_provider_errors():
    err = anthropic_events({"type": "error", "error": {"type": "overloaded_error"}})
    with pytest.raises(ProviderError, match="stream error"):
        await collect(
            AnthropicLLM("k", "m", client=client(lambda r: httpx.Response(200, content=err)))
        )
    with pytest.raises(ProviderError, match="HTTP 529"):
        await collect(
            AnthropicLLM("k", "m", client=client(lambda r: httpx.Response(529, text="busy")))
        )


async def test_anthropic_complete_still_works_after_the_refactor():
    def handler(req):
        body = json.loads(req.content)
        assert "stream" not in body
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    resp = await AnthropicLLM("k", "m", client=client(handler)).complete(MSGS)
    assert resp.text == "ok"


# ---- the fake LLM's grounded behaviour ---------------------------------------------------------

PROMPT = (
    '<sources>\n<source id="S1" title="policy">\nRefunds are issued within 14 days of purchase.'
    "\nOrders ship from Rotterdam.\n</source>\n"
    '<source id="S2" title="faq">\nSupport answers every weekday morning.\n</source>\n</sources>'
    "\n\nQuestion: {q}"
)


async def ask(q):
    resp = await FakeLLM().complete([ChatMessage("user", PROMPT.format(q=q))])
    return resp.text


async def test_the_fake_llm_answers_extractively_with_citations():
    assert (
        await ask("How long do refunds take?")
        == "Refunds are issued within 14 days of purchase. [S1]"
    )
    both = await ask("refunds and support weekday")
    assert "[S1]" in both and "[S2]" in both


async def test_the_fake_llm_says_not_found_when_nothing_matches():
    assert (await ask("quantum chromodynamics")).startswith("NOT_FOUND")


async def test_the_fake_llm_plays_along_with_rewrite_and_summary_prompts():
    rewrite = (
        "Rewrite the user's latest question so ...\n"
        "<question>what about price?</question>\nReply with JSON only"
    )
    out = await FakeLLM().complete([ChatMessage("user", rewrite)])
    assert json.loads(out.text) == {"query": "what about price?"}
    summary = (
        "Current summary:\nx\n\nNew messages:\nuser: hello there\n"
        "Reply with the updated summary only."
    )
    assert "hello there" in (await FakeLLM().complete([ChatMessage("user", summary)])).text
    assert (
        await FakeLLM().complete([ChatMessage("user", "plain")])
    ).text == "[fake-llm] You said: plain"
