"""
The request guard: which requests the API will answer at all.

Binding to 127.0.0.1 stops other machines. It does not stop your own browser,
and that is the path an attack would take:

* **Cross-site requests.** Any page you visit can make the browser send a
  request to http://127.0.0.1:8775 or :3100. CORS only stops that page reading
  the reply; the request itself still runs. A plain HTML form needs no script
  and no preflight, so an upload, an index rebuild or a delete would be
  reachable from any website.
* **DNS rebinding.** A hostile domain re-points its own name at 127.0.0.1 and
  becomes "same-origin" with the API, able to read answers and documents. The
  one thing it cannot fake is the host name the browser puts in the request.

So, for every request:

1. The host must be a loopback name. Both `Host` and `X-Forwarded-Host` are
   checked: the Next.js proxy rewrites `Host` to the API's address and carries
   the browser's original host in `X-Forwarded-Host` (measured in the trading
   app, 2026-09-23).
2. Anything other than GET/HEAD/OPTIONS must not be cross-site: an `Origin`,
   when present, must be one of the app's own, and `Sec-Fetch-Site: cross-site`
   is refused. Browsers send `Origin` on every POST, so a missing header means
   a non-browser client (curl, the test client), which is not this threat.

Pure ASGI rather than Starlette's BaseHTTPMiddleware, which buffers streaming
responses and would hold back every streamed answer until it finished.
"""
from __future__ import annotations

import json
import os

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def allowed_origins() -> list[str]:
    """The app's own origins: the UI, and the API itself (its /api/_docs page).
    Ports follow the variables `./rag web` exports, so moving a port cannot
    silently lock the UI out or leave the old one trusted."""
    ui = os.environ.get("RAG_STUDIO_UI_PORT", "3100")
    api = os.environ.get("RAG_STUDIO_API_PORT", "8775")
    return [f"http://{host}:{port}" for port in (ui, api) for host in ("localhost", "127.0.0.1")]


def hostname(value: str) -> str:
    """`127.0.0.1:8775` -> `127.0.0.1`; `[::1]:8775` -> `::1`. Lower-cased."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def refusal(scope: dict, origins: frozenset[str]) -> str | None:
    """Why this request is refused, or None to let it through."""
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
    for name in ("host", "x-forwarded-host"):
        if name in headers and hostname(headers[name]) not in LOOPBACK_HOSTS:
            return ("This app only answers requests addressed to this computer "
                    f"by name (127.0.0.1 or localhost), not {headers[name]!r}.")
    if scope.get("method", "GET").upper() in SAFE_METHODS:
        return None
    if headers.get("sec-fetch-site") == "cross-site":
        return "Refused: this change was requested by another website."
    origin = headers.get("origin")
    if origin is not None and origin.rstrip("/") not in origins:
        return f"Refused: changes must come from RAG Studio itself, not {origin!r}."
    return None


class RequestGuard:
    """ASGI middleware applying `refusal()` to every HTTP request."""

    def __init__(self, app, origins: list[str] | None = None):
        self.app = app
        self.origins = frozenset(origins if origins is not None else allowed_origins())

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            reason = refusal(scope, self.origins)
            if reason is not None:
                body = json.dumps({"detail": reason}).encode()
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
