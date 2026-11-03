"""Error model (Phase 1.7): every error response is RFC 9457 Problem Details.

    {"type": "...", "title": "...", "status": 404, "detail": "...",
     "instance": "/v1/...", "request_id": "..."}

served as application/problem+json. Provider failures add a `provider` member and validation
failures add an `errors` list. Unexpected exceptions return a generic 500 and never leak details.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.providers.errors import ProviderError

log = logging.getLogger("atlas.errors")

PROBLEM_JSON = "application/problem+json"
TYPE_BASE = "https://atlas-rag.dev/problems/"


class AppError(Exception):
    """Base for expected, client-facing errors. Subclasses set status and code."""

    status: int = 500
    code: str = "internal_error"
    title: str = "Internal error"

    def __init__(
        self,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(detail or self.title)
        self.detail = detail or self.title
        self.headers = headers or {}
        self.extra = extra or {}  # extension members of the problem document


class BadRequest(AppError):
    status, code, title = 400, "bad_request", "Bad request"


class Unauthorized(AppError):
    status, code, title = 401, "unauthorized", "Authentication required"

    def __init__(self, detail: str | None = None):
        super().__init__(detail, headers={"WWW-Authenticate": "Bearer"})


class Forbidden(AppError):
    status, code, title = 403, "forbidden", "Forbidden"


class NotFound(AppError):
    status, code, title = 404, "not_found", "Not found"


class Conflict(AppError):
    status, code, title = 409, "conflict", "Conflict"


class PayloadTooLarge(AppError):
    status, code, title = 413, "payload_too_large", "Payload too large"


class UnsupportedMedia(AppError):
    status, code, title = 415, "unsupported_media_type", "Unsupported media type"


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def problem(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    body = {
        "type": f"{TYPE_BASE}{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "request_id": request_id_of(request),
        **extra,
    }
    h = dict(headers or {})
    if rid := body["request_id"]:
        h["X-Request-ID"] = rid
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=h)


def _clean_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    # Drop 'input' and 'ctx': they can echo submitted values (e.g. passwords) or hold
    # non-serialisable objects.
    return [
        {"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return problem(
            request,
            status=exc.status,
            code=exc.code,
            title=exc.title,
            detail=exc.detail,
            headers=exc.headers,
            **exc.extra,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return problem(
            request,
            status=422,
            code="validation_error",
            title="Validation failed",
            detail="The request is invalid. See 'errors' for details.",
            errors=_clean_validation_errors(exc),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        phrase = HTTPStatus(exc.status_code).phrase
        return problem(
            request,
            status=exc.status_code,
            code=phrase.lower().replace(" ", "_"),
            title=phrase,
            detail=str(exc.detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(ProviderError)
    async def _provider(request: Request, exc: ProviderError):
        # 502: our upstream (model / vector DB) failed, not the caller's request.
        return problem(
            request,
            status=502,
            code="provider_error",
            title="Upstream provider error",
            detail=str(exc),
            provider=exc.provider,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error(
            "unhandled exception",
            exc_info=exc,
            extra={"request_id": request_id_of(request), "path": request.url.path},
        )
        return problem(
            request,
            status=500,
            code="internal_error",
            title="Internal server error",
            detail="Something went wrong. Quote the request_id when reporting this.",
        )
