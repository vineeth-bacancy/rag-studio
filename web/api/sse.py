"""Server-sent events from an engine generator, with a heartbeat.

The engine yields `(event, data)` pairs from ordinary blocking code (an Ollama
stream), so it runs in its own thread and hands events over a queue. This side
writes each one as an SSE frame, and a `: ping` comment whenever 15 seconds
pass without one.

Why not FastAPI's built-in `EventSourceResponse`: it pings too, but it sets
`Cache-Control: no-cache` itself and a route can only append a second header,
not replace it. `no-transform` is the one that matters here. Next's rewrite
proxy gzips any response the browser accepts gzip for, and a gzipped event
stream arrives all at once when it ends (measured in the trading app: every
event landed together after a full CPU answer). Next skips compression for a
response marked `no-transform`.

The heartbeat matters because the rewrite proxy drops a socket that is idle
for `proxyTimeout` (330 s here), and a CPU model reading a long prompt sends
nothing for a long time.

Pure ASGI end to end: the request guard is not `BaseHTTPMiddleware`, which
would buffer this whole stream.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Iterator

import anyio
from fastapi.responses import StreamingResponse

PING_S = 15.0
HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
PING = b": ping\n\n"


def frame(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def stream(events: Iterator[tuple[str, Any]], *, ping_s: float = PING_S) -> StreamingResponse:
    async def body():
        q: queue.Queue = queue.Queue(maxsize=64)
        stop = threading.Event()

        def put(item: tuple[str, Any]) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for item in events:
                    if not put(("event", item)):
                        break
            except Exception as e:  # noqa: BLE001 - becomes an `error` event below
                put(("error", str(e) or type(e).__name__))
            finally:
                # Closing the generator closes its Ollama connection, which is
                # how Ollama learns to stop generating for a reader who left.
                close = getattr(events, "close", None)
                if close:
                    close()
                put(("end", None))

        threading.Thread(target=produce, name="sse-producer", daemon=True).start()
        last = time.monotonic()
        try:
            while True:
                try:
                    kind, item = await anyio.to_thread.run_sync(q.get, True, 1.0)
                except queue.Empty:
                    if time.monotonic() - last >= ping_s:
                        last = time.monotonic()
                        yield PING
                    continue
                if kind == "end":
                    break
                last = time.monotonic()
                if kind == "error":
                    yield frame("error", {"error": item})
                    break
                yield frame(*item)
        finally:
            stop.set()

    return StreamingResponse(body(), media_type="text/event-stream", headers=HEADERS)
