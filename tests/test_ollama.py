import json

import httpx
import pytest

from app.providers.base import ChatMessage
from app.providers.errors import ProviderError
from app.providers.ollama import OllamaEmbedder, OllamaLLM


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_chat_request_and_response_mapping():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"message": {"content": "hi"}, "prompt_eval_count": 7, "eval_count": 3}
        )

    llm = OllamaLLM("http://ollama:11434/", "llama3.1:8b", client(handler))
    out = await llm.complete([ChatMessage("user", "yo")], temperature=0.0, max_tokens=50)
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"] == {"temperature": 0.0, "num_predict": 50}
    assert (out.text, out.usage.prompt_tokens, out.usage.completion_tokens) == ("hi", 7, 3)


async def test_http_error_becomes_provider_error():
    llm = OllamaLLM("http://x", "m", client(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(ProviderError) as e:
        await llm.complete([ChatMessage("user", "yo")])
    assert e.value.status == 500


async def test_unexpected_shape_becomes_provider_error():
    llm = OllamaLLM("http://x", "m", client(lambda r: httpx.Response(200, json={"oops": 1})))
    with pytest.raises(ProviderError):
        await llm.complete([ChatMessage("user", "yo")])


async def test_connection_failure_becomes_provider_error():
    def handler(req):
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError):
        await OllamaLLM("http://x", "m", client(handler)).complete([ChatMessage("user", "yo")])


async def test_embedder_returns_vectors_and_learns_dimension():
    def handler(req):
        n = len(json.loads(req.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * n})

    e = OllamaEmbedder("http://x", "nomic-embed-text", client(handler))
    assert e.dimension is None
    vecs = await e.embed(["a", "b"])
    assert len(vecs) == 2 and e.dimension == 3
    assert await e.embed([]) == []


async def test_embedder_detects_count_mismatch():
    e = OllamaEmbedder(
        "http://x", "m", client(lambda r: httpx.Response(200, json={"embeddings": [[1.0]]}))
    )
    with pytest.raises(ProviderError):
        await e.embed(["a", "b"])
