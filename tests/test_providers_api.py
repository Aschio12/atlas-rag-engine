import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def client(**kw) -> TestClient:
    kw = {"llm_provider": "fake", "embed_provider": "fake", "vector_store": "memory", **kw}
    return TestClient(create_app(Settings(_env_file=None, **kw)))


def test_status_reports_config_without_secrets():
    c = client(llm_provider="openai", openai_api_key="sk-TOPSECRET")
    body = c.get("/v1/providers").json()
    assert body["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert body["vector_store"] == {"provider": "memory"}
    assert "sk-TOPSECRET" not in c.get("/v1/providers").text


def test_chat_playground_with_fake_llm():
    r = client().post("/v1/providers/playground/chat", json={"prompt": "hello world"})
    assert r.status_code == 200
    assert "hello world" in r.json()["text"]


def test_similarity_playground_ranks_relevant_document_first():
    r = client().post(
        "/v1/providers/playground/similarity",
        json={
            "query": "python web framework",
            "documents": ["chocolate cake recipe", "FastAPI is a Python web framework"],
            "limit": 2,
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["text"].startswith("FastAPI")
    assert results[0]["score"] > results[1]["score"]


def test_playground_is_hidden_in_prod():
    c = client(app_env="prod", jwt_secret="x" * 40)
    assert c.post("/v1/providers/playground/chat", json={"prompt": "hi"}).status_code == 404
    assert c.get("/v1/providers").json()["playground_enabled"] is False


@pytest.mark.parametrize("body", [{}, {"prompt": ""}, {"prompt": "x" * 4001}])
def test_chat_validates_input(body):
    assert client().post("/v1/providers/playground/chat", json=body).status_code == 422


def test_unreachable_provider_returns_clear_502():
    # Default Ollama URL is not running in tests -> transport failure -> ProviderError -> 502.
    c = TestClient(
        create_app(Settings(_env_file=None, ollama_url="http://127.0.0.1:1", vector_store="memory"))
    )
    r = c.post("/v1/providers/playground/chat", json={"prompt": "hi"})
    assert r.status_code == 502
    assert r.json()["provider"] == "ollama"
