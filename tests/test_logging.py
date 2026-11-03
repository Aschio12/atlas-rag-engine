import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import _HANDLER_MARK, ConsoleFormatter, JsonFormatter, setup_logging
from app.main import create_app


def make_client(**kw) -> TestClient:
    app = create_app(Settings(_env_file=None, llm_provider="fake", **kw))

    @app.get("/_ok")
    def ok():
        return {"ok": True}

    @app.get("/_crash")
    def crash():
        raise RuntimeError("kaboom")

    return TestClient(app, raise_server_exceptions=False)


def access_records(caplog):
    return [r for r in caplog.records if r.name == "atlas.access"]


def test_generates_request_id_and_echoes_it():
    r = make_client().get("/_ok")
    rid = r.headers["x-request-id"]
    assert len(rid) == 32


def test_keeps_well_formed_incoming_id():
    r = make_client().get("/_ok", headers={"X-Request-ID": "trace-1234.abcd"})
    assert r.headers["x-request-id"] == "trace-1234.abcd"


@pytest.mark.parametrize("bad", ["short", "has space in it!!", "x" * 100, "a\tb-1234567"])
def test_replaces_suspicious_incoming_id(bad):
    r = make_client().get("/_ok", headers={"X-Request-ID": bad})
    assert r.headers["x-request-id"] != bad
    assert len(r.headers["x-request-id"]) == 32


def test_access_log_has_fields_and_request_id(caplog):
    with caplog.at_level(logging.INFO, logger="atlas.access"):
        r = make_client().get("/_ok")
    (rec,) = access_records(caplog)
    assert (rec.method, rec.path, rec.status) == ("GET", "/_ok", 200)
    assert rec.duration_ms >= 0
    assert rec.request_id == r.headers["x-request-id"]


def test_access_log_never_contains_query_string_or_auth_header(caplog):
    with caplog.at_level(logging.INFO, logger="atlas.access"):
        make_client().get("/_ok?token=SUPERSECRET", headers={"Authorization": "Bearer SUPERSECRET"})
    ours = [r for r in caplog.records if r.name.startswith("atlas")]
    assert ours
    assert "SUPERSECRET" not in json.dumps([r.__dict__ for r in ours], default=str)


def test_httpx_url_logging_is_quieted(caplog):
    make_client()  # runs setup_logging
    assert logging.getLogger("httpx").level == logging.WARNING


def test_500_has_request_id_in_header_and_body_and_is_logged(caplog):
    with caplog.at_level(logging.INFO):
        r = make_client().get("/_crash")
    assert r.status_code == 500
    assert r.headers["x-request-id"] == r.json()["request_id"]
    (rec,) = access_records(caplog)
    assert rec.status == 500
    errors = [x for x in caplog.records if x.name == "atlas.errors"]
    assert errors and errors[0].request_id == r.json()["request_id"]


def _record(**extra) -> logging.LogRecord:
    rec = logging.LogRecord("atlas.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    rec.__dict__.update(extra)
    return rec


def test_json_formatter_emits_valid_json_with_extras():
    out = json.loads(JsonFormatter().format(_record(request_id="abc12345", status=200)))
    assert out["message"] == "hello world"
    assert out["level"] == "INFO" and out["logger"] == "atlas.test"
    assert out["request_id"] == "abc12345" and out["status"] == 200
    assert out["ts"].endswith("+00:00")


def test_json_formatter_includes_exception_text():
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    assert "ValueError: bad" in json.loads(JsonFormatter().format(rec))["exc_info"]


def test_console_formatter_is_readable():
    line = ConsoleFormatter().format(_record(request_id="abcdef123456", status=200))
    assert (
        "INFO" in line and "hello world" in line and "[abcdef12]" in line and "status=200" in line
    )


def test_setup_logging_is_idempotent_and_honours_settings():
    root = logging.getLogger()
    setup_logging(Settings(_env_file=None, log_format="console", log_level="WARNING"))
    setup_logging(Settings(_env_file=None, log_format="console", log_level="WARNING"))
    ours = [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]
    assert len(ours) == 1
    assert isinstance(ours[0].formatter, ConsoleFormatter)
    assert root.level == logging.WARNING
    setup_logging(Settings(_env_file=None))  # restore defaults for other tests
    assert root.level == logging.INFO
