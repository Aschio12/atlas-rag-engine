import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import BadRequest, Conflict, Forbidden, NotFound, Unauthorized
from app.main import create_app
from app.providers.errors import ProviderError


class Body(BaseModel):
    password: str
    age: int


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None, llm_provider="fake"))

    @app.get("/_t/{kind}")
    def boom(kind: str):
        errors = {
            "bad": BadRequest("nope"),
            "unauth": Unauthorized(),
            "forbidden": Forbidden("not yours"),
            "missing": NotFound("no such thing"),
            "conflict": Conflict("already exists"),
            "provider": ProviderError("ollama", "request failed: ConnectError"),
            "http": HTTPException(418, "teapot"),
        }
        if kind == "crash":
            raise RuntimeError("secret internal detail: db password=hunter2")
        raise errors[kind]

    @app.post("/_validate")
    def validate(body: Body):
        return {}

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [
        ("bad", 400, "bad_request"),
        ("unauth", 401, "unauthorized"),
        ("forbidden", 403, "forbidden"),
        ("missing", 404, "not_found"),
        ("conflict", 409, "conflict"),
        ("provider", 502, "provider_error"),
        ("http", 418, "i'm_a_teapot"),
    ],
)
def test_problem_shape(client, kind, status, code):
    r = client.get(f"/_t/{kind}")
    assert r.status_code == status
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["status"] == status
    assert body["type"].endswith(f"/{code}")
    assert body["instance"] == f"/_t/{kind}"
    assert {"title", "detail", "request_id"} <= body.keys()


def test_unauthorized_sets_www_authenticate(client):
    assert client.get("/_t/unauth").headers["www-authenticate"] == "Bearer"


def test_provider_error_names_the_provider(client):
    assert client.get("/_t/provider").json()["provider"] == "ollama"


def test_unknown_route_is_a_problem_too(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_validation_errors_list_fields_but_never_echo_input(client):
    r = client.post("/_validate", json={"password": {"leak": "S3cr3tValue"}, "age": "x"})
    assert r.status_code == 422
    body = r.json()
    assert {e["loc"][-1] for e in body["errors"]} == {"password", "age"}
    assert all("input" not in e for e in body["errors"])
    assert "S3cr3tValue" not in r.text


def test_unhandled_exception_is_generic_and_leaks_nothing(client):
    r = client.get("/_t/crash")
    assert r.status_code == 500
    assert "hunter2" not in r.text and "RuntimeError" not in r.text
    assert r.json()["title"] == "Internal server error"
