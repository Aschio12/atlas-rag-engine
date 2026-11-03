"""Request-id propagation and access logging, as a pure ASGI middleware.

Pure ASGI (not BaseHTTPMiddleware) so response headers and context variables behave correctly
even when a handler raises.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

log = logging.getLogger("atlas.access")

REQUEST_ID_HEADER = "x-request-id"
# Accept a caller-supplied id only if it is short and boring (prevents log injection).
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def _incoming_id(scope: Scope) -> str:
    for name, value in scope["headers"]:
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1")
            if _VALID_ID.match(candidate):
                return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_id(scope)
        # Handlers (incl. the 500 handler, which runs outside this middleware) read it from here.
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            request_id_var.reset(token)
