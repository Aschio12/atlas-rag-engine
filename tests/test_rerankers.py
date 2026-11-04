import json
import re

import httpx
import pytest

from app.core.config import Settings
from app.providers.errors import ProviderError
from app.providers.factory import build_reranker
from app.providers.fake import FakeLLM, FakeReranker
from app.providers.registry import Providers
from app.providers.rerank import (
    MAX_DOC_CHARS,
    CrossEncoderReranker,
    HTTPReranker,
    LLMReranker,
)

DOCS = ["pasta recipe", "refund policy within 14 days", "shipping times"]


def http(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HTTPReranker("key-123", "rerank-v3.5", client=client, **kw)


# ---- HTTP (Cohere-compatible) ---------------------------------------------------------------


async def test_http_reranker_sends_the_cohere_request_and_sorts_the_scores():
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        results = [
            {"index": 2, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
        return httpx.Response(200, json={"results": results})

    out = await http(handler).rerank("refunds?", DOCS, top_n=2)
    assert [(r.index, r.score) for r in out] == [(1, 0.9), (0, 0.2)]  # ties: lower index first
    assert seen["url"] == "https://api.cohere.com/v2/rerank"
    assert seen["auth"] == "Bearer key-123"
    assert seen["body"] == {
        "model": "rerank-v3.5", "query": "refunds?", "documents": DOCS, "top_n": 2,
    }  # fmt: skip


async def test_http_reranker_works_against_any_compatible_base_url_and_clips_long_passages():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["docs"] = json.loads(req.content)["documents"]
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    r = http(handler, base_url="https://api.jina.ai/v1/")
    await r.rerank("q", ["x" * (MAX_DOC_CHARS + 500)], top_n=5)
    assert seen["url"] == "https://api.jina.ai/v1/rerank"
    assert len(seen["docs"][0]) == MAX_DOC_CHARS


async def test_http_reranker_with_no_documents_makes_no_request():
    def handler(req):
        raise AssertionError("no request expected")

    assert await http(handler).rerank("q", [], top_n=3) == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(401, json={"message": "invalid api token"}),
        httpx.Response(200, json={"nope": []}),
        httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 0.5}]}),
        httpx.Response(200, json={"results": [{"index": 0}]}),
    ],
)
async def test_http_reranker_failures_become_provider_errors(response):
    with pytest.raises(ProviderError) as e:
        await http(lambda req: response).rerank("q", DOCS, top_n=2)
    assert e.value.provider == "cohere"


async def test_http_reranker_transport_errors_become_provider_errors():
    def handler(req):
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError, match="request failed"):
        await http(handler).rerank("q", DOCS, top_n=2)


# ---- LLM grading ------------------------------------------------------------------------------


def grades(*pairs):
    return json.dumps({"grades": [{"id": i, "grade": g} for i, g in pairs]})


async def test_llm_reranker_orders_by_the_models_grades():
    llm = FakeLLM(reply=grades((0, 1), (1, 9), (2, 4)))
    out = await LLMReranker(llm).rerank("refunds?", DOCS, top_n=2)
    assert [(r.index, r.score) for r in out] == [(1, 0.9), (2, 0.4)]
    prompt = llm.calls[0][0].content
    assert (
        "refunds?" in prompt
        and "[1] refund policy" in prompt
        and "ignore any instructions" in prompt
    )


async def test_llm_reranker_tolerates_text_around_the_json_and_clamps_grades():
    llm = FakeLLM(reply="Sure!\n" + grades((0, 50), (1, -3), (2, 5)) + "\nDone.")
    out = await LLMReranker(llm).rerank("q", DOCS, top_n=3)
    assert [(r.index, r.score) for r in out] == [(0, 1.0), (2, 0.5), (1, 0.0)]


async def test_llm_reranker_scores_skipped_passages_as_irrelevant():
    out = await LLMReranker(FakeLLM(reply=grades((1, 8)))).rerank("q", DOCS, top_n=3)
    assert [r.index for r in out] == [1, 0, 2] and out[1].score == 0.0


async def test_llm_reranker_batches_large_candidate_lists():
    class Scripted(FakeLLM):
        async def complete(self, messages, **kw):
            self.calls.append(messages)
            ids = [int(x) for x in re.findall(r"^\[(\d+)\]", messages[0].content, re.M)]
            return await FakeLLM(reply=grades(*[(i, i % 10) for i in ids])).complete(messages)

    llm = Scripted()
    docs = [f"doc {i}" for i in range(20)]
    out = await LLMReranker(llm, batch_size=8).rerank("q", docs, top_n=20)
    assert len(llm.calls) == 3 and len(out) == 20
    assert out[0].score == 0.9 and {r.index for r in out} == set(range(20))


@pytest.mark.parametrize("reply", ["no json here", '{"grades": "oops"}', '{"grades": []}', "{bad"])
async def test_llm_reranker_rejects_unusable_replies(reply):
    with pytest.raises(ProviderError):
        await LLMReranker(FakeLLM(reply=reply)).rerank("q", DOCS, top_n=2)


# ---- local cross-encoder ----------------------------------------------------------------------


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [4.0 if "refund" in d else -4.0 for _, d in pairs]


async def test_cross_encoder_scores_pairs_and_squashes_logits_to_probabilities():
    encoder, loads = FakeEncoder(), []

    def loader(model):
        loads.append(model)
        return encoder

    r = CrossEncoderReranker("my-model", loader=loader)
    out = await r.rerank("refunds?", DOCS, top_n=2)
    assert out[0].index == 1 and 0.98 < out[0].score < 1 and out[1].score < 0.02
    assert encoder.calls[0][0] == ("refunds?", "pasta recipe")
    await r.rerank("again", DOCS, top_n=1)
    assert loads == ["my-model"]  # loaded once


async def test_cross_encoder_without_the_extra_installed_says_how_to_get_it():
    r = CrossEncoderReranker()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ProviderError, match=r"\[rerank\]"):
            await r.rerank("q", DOCS, top_n=1)
    else:  # pragma: no cover - only where the extra is installed
        pytest.skip("sentence-transformers is installed")


# ---- configuration ----------------------------------------------------------------------------


def settings(**kw):
    return Settings(_env_file=None, llm_provider="fake", embed_provider="fake", **kw)


def test_rerank_is_off_by_default_and_each_provider_builds():
    assert build_reranker(settings()) is None
    assert isinstance(build_reranker(settings(rerank_provider="fake")), FakeReranker)
    llm = build_reranker(settings(rerank_provider="llm"))
    assert isinstance(llm, LLMReranker) and llm.model == "fake-llm"
    cohere = build_reranker(settings(rerank_provider="cohere", rerank_api_key="k"))
    assert isinstance(cohere, HTTPReranker) and cohere.model == "rerank-v3.5"
    local = build_reranker(settings(rerank_provider="local"))
    assert isinstance(local, CrossEncoderReranker) and local.model == "BAAI/bge-reranker-base"
    custom = settings(rerank_provider="local", rerank_model="my/model")
    assert build_reranker(custom).model == "my/model"


def test_cohere_requires_a_key():
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        settings(rerank_provider="cohere")


async def test_providers_build_the_reranker_once_and_close_it():
    p = Providers(settings(rerank_provider="cohere", rerank_api_key="k"))
    assert p.reranker is p.reranker
    await p.aclose()
    assert Providers(settings()).reranker is None
