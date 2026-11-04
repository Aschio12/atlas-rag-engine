import json

import httpx
import pytest

from app.providers.anthropic import AnthropicLLM
from app.providers.base import ChatMessage
from app.providers.errors import ProviderError
from app.providers.openai import OpenAIEmbedder, OpenAILLM


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_chat_sends_bearer_and_maps_usage():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    llm = OpenAILLM("sk-1", "gpt-4o-mini", base_url="http://local/v1/", client=client(handler))
    out = await llm.complete([ChatMessage("user", "q")])
    assert seen["auth"] == "Bearer sk-1"
    assert seen["url"] == "http://local/v1/chat/completions"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert (out.text, out.usage.prompt_tokens, out.usage.completion_tokens) == ("ok", 5, 2)


async def test_openai_error_does_not_leak_api_key():
    llm = OpenAILLM("sk-SECRET", "m", client=client(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(ProviderError) as e:
        await llm.complete([ChatMessage("user", "q")])
    assert "sk-SECRET" not in str(e.value)
    assert e.value.status == 401


async def test_openai_without_key_sends_no_auth_header():
    seen = {}

    def handler(req):
        seen["has_auth"] = "authorization" in req.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    await OpenAILLM("", "m", base_url="http://vllm/v1", client=client(handler)).complete(
        [ChatMessage("user", "q")]
    )
    assert seen["has_auth"] is False


async def test_openai_embedder_orders_by_index():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ]
            },
        )

    e = OpenAIEmbedder("k", client=client(handler))
    assert await e.embed(["a", "b"]) == [[1.0, 1.0], [2.0, 2.0]]
    assert e.dimension == 2


async def test_anthropic_splits_system_and_maps_response():
    seen = {}

    def handler(req):
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        )

    llm = AnthropicLLM("ak", "claude-sonnet-5-5", client=client(handler))
    out = await llm.complete([ChatMessage("system", "be kind"), ChatMessage("user", "hi")])
    assert seen["headers"]["x-api-key"] == "ak"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "be kind"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert (out.text, out.usage.prompt_tokens, out.usage.completion_tokens) == ("hello", 9, 4)


async def test_anthropic_omits_system_when_absent_and_rejects_bad_shape():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"nope": True})

    with pytest.raises(ProviderError):
        await AnthropicLLM("k", "m", client=client(handler)).complete([ChatMessage("user", "hi")])
    assert "system" not in seen["body"]


async def test_anthropic_marks_the_cached_system_prefix_and_reports_cache_reads():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 900},
            },
        )

    llm = AnthropicLLM("k", "m", client=client(handler))
    out = await llm.complete(
        [
            ChatMessage("system", "<document>long text</document>", cache=True),
            ChatMessage("system", "Answer briefly."),
            ChatMessage("user", "hi"),
        ]
    )
    assert seen["body"]["system"] == [
        {
            "type": "text",
            "text": "<document>long text</document>",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Answer briefly."},
    ]
    assert out.usage.cached_tokens == 900


async def test_openai_reports_cached_prompt_tokens_and_ignores_the_cache_flag():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 896},
                },
            },
        )

    llm = OpenAILLM("k", "m", client=client(handler))
    out = await llm.complete([ChatMessage("system", "shared prefix", cache=True)])
    assert out.usage.cached_tokens == 896
    assert seen["body"]["messages"] == [{"role": "system", "content": "shared prefix"}]
