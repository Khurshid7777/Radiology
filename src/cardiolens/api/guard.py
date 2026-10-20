"""Request guard in front of every route (D98).

FastAPI parses a multipart form *before* it resolves dependencies, so a key check
in a dependency comes after the whole body was received and spooled to disk: an
unauthenticated client could stream gigabytes to /scans before its 401. This ASGI
middleware answers first:

* protected paths without a valid ``X-API-Key`` get 401 (503 with no keys
  configured) before a byte of the body is read;
* a body over ``max_body_bytes`` gets 413 — by ``Content-Length`` up front, or while
  streaming when the length is not declared.
"""

from __future__ import annotations

import json

PROTECTED_PREFIXES = ("/api/analyze", "/api/analysis", "/api/calibration", "/scans")


class _TooLarge(Exception):
    pass


async def _send_json(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class RequestGuard:
    def __init__(self, app, *, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from ecgscan_service.router import key_name

        headers = dict(scope.get("headers") or [])
        if scope["path"].startswith(PROTECTED_PREFIXES):
            key = headers.get(b"x-api-key")
            status = key_name(key.decode("latin-1") if key is not None else None)
            if isinstance(status, int):
                return await _send_json(send, status, "invalid or missing X-API-Key" if status == 401
                                        else "API keys are not configured")
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            return await _send_json(send, 413, f"request body over {self.max_body_bytes} bytes")

        seen = 0
        started = False

        async def counted_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_body_bytes:
                    raise _TooLarge
            return msg

        async def tracked_send(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except _TooLarge:
            if not started:
                await _send_json(send, 413, f"request body over {self.max_body_bytes} bytes")
