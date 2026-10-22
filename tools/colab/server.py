#!/usr/bin/env python3
"""An authenticated gateway in front of Ollama, on a Google Colab GPU runtime.

    laptop --HTTPS--> cloudflared tunnel --> THIS (0.0.0.0:8000) --> ollama (127.0.0.1:11434)

Ollama has no authentication at all, so it must never face the tunnel itself.
This process is the only thing that does, and it is deliberately small:

* **Standard library only.** A Colab runtime keeps nothing between sessions,
  so every dependency would be reinstalled (and could drift) on every start.
* **An allowlist, not a proxy.** Six Ollama paths are forwarded, plus the GPU
  probe. Any other path is a 404 before the token is even looked at, so a
  leaked token still cannot reach `/api/pull`, `/api/create`, `/api/delete` or
  `/api/push`: the worst it buys is inference on a model already pulled.
* **A bearer token**, compared in constant time.
* **An 8 MB body cap.** A batch of 32 passages is about 40 KB; a rendered page
  image for the vision role (M7) is 1-3 MB as base64.
* **Nothing about content is logged.** Method, path, status, sizes, duration,
  the model name and how many inputs arrived. Never a passage, question or
  answer: the runtime is rented and the log outlives the request.
* **`stream` is forced off for chat.** The laptop reads one JSON reply per call.
* **Large replies are gzipped** when the client accepts it. Vectors are bulky
  as JSON (32 passages × 4,096 floats is 1.7 MB) and the free tunnel moved them
  at about 90 KB/s (measured 2026-10-15: 8.4 s on the T4, 29.7 s round trip).
  Cloudflare compresses only its own leg to the laptop, so the gateway
  compresses the runtime's leg itself.
* **Requests may be gzipped** (`Content-Encoding: gzip`): a batch of passages is
  mostly text and compresses well, and the tunnel's upload leg is the slow one.
  The body is inflated under the same 8 MB cap, so a compression bomb gets a
  413; corrupt data a 400; any other encoding a 415. What is forwarded is plain.
* **Two more allowlisted routes go to the worker** (worker.py, loopback only):
  `POST /rerank` and `GET /worker`. Same token, same cap, same 404-before-auth.
* **Every forwarded reply carries `Server-Timing`**, the time the upstream took,
  so the laptop can tell tunnel time from GPU time.
* **No CORS headers.** A web page must never be able to call this from a
  browser; the only client is the RAG Studio engine on the laptop.

Run it (start.sh does this, after proving the lock on loopback):

    RAG_STUDIO_TOKEN=<hex> python3 server.py --port 8000

It knows nothing about RAG Studio's documents and reads no file from the
laptop. See docs/02-remote-gpu.md.
"""
from __future__ import annotations

import argparse
import gzip
import hmac
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICE = "rag-studio-gateway"
VERSION = "2"

# Every path this gateway serves, by method. Anything else is a 404 before the
# token is considered and before a body is read.
FORWARD_GET = ("/api/version", "/api/tags", "/api/ps")
FORWARD_POST = ("/api/chat", "/api/embed", "/api/embeddings")
OWN_GET = ("/api/gpu",)
# The worker (tools/colab/worker.py), behind the same lock. Declared explicitly,
# never "everything under /worker": a new route is a decision.
WORKER_POST = ("/rerank", "/context")
WORKER_GET = ("/worker",)

MAX_BODY_BYTES = 8 * 1024 * 1024
# How much of an over-large body is read and thrown away so the client gets
# its 413 instead of a broken pipe. Past this the connection is just closed.
DRAIN_CAP_BYTES = 64 * 1024 * 1024
# Replies smaller than this go as they are: compressing a few KB saves nothing.
GZIP_MIN_BYTES = 16 * 1024
TOKEN_ENV = "RAG_STUDIO_TOKEN"


class Config:
    ollama = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    token = os.environ.get(TOKEN_ENV, "")
    upstream_timeout = float(os.environ.get("UPSTREAM_TIMEOUT", "900"))
    worker = os.environ.get("WORKER_URL", "http://127.0.0.1:8001").rstrip("/")
    # Under the tunnel's 100 s limit (HTTP 524), so a slow job is told so by us.
    worker_timeout = float(os.environ.get("WORKER_TIMEOUT", "90"))
    require_token = True


def gpu_info() -> dict:
    """What the runtime was actually given. Colab hands out CPU-only runtimes
    without saying so, so this is asked, never assumed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return {"available": False, "reason": "nvidia-smi returned an error"}
        rows = []
        for line in out.stdout.strip().splitlines():
            name, total, used, util = (p.strip() for p in line.split(","))
            rows.append({"name": name, "memory_total_mb": int(total),
                         "memory_used_mb": int(used), "utilization_pct": int(util)})
        return {"available": bool(rows), "gpus": rows}
    except FileNotFoundError:
        return {"available": False, "reason": "no nvidia-smi: this runtime has no GPU"}
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not fail
        return {"available": False, "reason": str(exc)[:120]}


def upstream(method: str, path: str, body: bytes | None, timeout: float,
             base: str | None = None) -> tuple[int, bytes, str]:
    req = urllib.request.Request(f"{base or Config.ollama}{path}", data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason):
            return 504, json.dumps({"error": "the model server on this runtime did not finish in time"}).encode(), \
                "application/json"
        return 502, json.dumps({"error": f"the model server on this runtime is not answering: {reason}"}).encode(), \
            "application/json"
    except TimeoutError:
        return 504, json.dumps({"error": "the model server on this runtime did not finish in time"}).encode(), \
            "application/json"


def _shape(payload: dict, raw: bytes) -> str:
    """What the log may say about a request: its size and shape, never its text."""
    inputs = payload.get("input")
    count = len(inputs) if isinstance(inputs, list) else (1 if inputs else 0)
    documents = payload.get("documents")
    messages = payload.get("messages")
    parts = [f"model={payload.get('model')}", f"in={len(raw)}B"]
    if count:
        parts.append(f"inputs={count}")
    if isinstance(documents, list):
        parts.append(f"documents={len(documents)}")
    if isinstance(messages, list):
        parts.append(f"messages={len(messages)}")
    return " ".join(parts)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE}/{VERSION}"
    sys_version = ""                 # do not advertise the Python version
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - base class API
        """Silenced: `_done` logs what is safe to log."""
        del format, args

    def _send(self, status: int, payload: bytes, content_type: str = "application/json",
              server_ms: float | None = None) -> None:
        accepts = self.headers.get("Accept-Encoding", "")
        gzipped = len(payload) >= GZIP_MIN_BYTES and "gzip" in accepts.lower()
        if gzipped:
            payload = gzip.compress(payload, compresslevel=5)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if server_ms is not None:
            self.send_header("Server-Timing", f"upstream;dur={server_ms:.1f}")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode())

    def _done(self, status: int, sent: int, started: float, note: str = "") -> None:
        print(f"{time.strftime('%H:%M:%S')} {self.command} {self._path()} -> {status} {sent}B "
              f"{time.time() - started:.1f}s {note}".rstrip(), flush=True)

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _authorised(self) -> bool:
        if not Config.require_token:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        # Constant time, so a wrong token cannot be recovered byte by byte
        # from how long the refusal took.
        return hmac.compare_digest(header[7:].strip().encode(), Config.token.encode())

    def _read_body(self) -> bytes | None:
        """The request body, or None when it is too large to accept."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0:
            return b""
        if length > MAX_BODY_BYTES:
            self._drain(length)
            return None
        raw = self.rfile.read(length)
        encoding = self.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding in ("", "identity"):
            return raw
        if encoding != "gzip":
            self._encoding_error = (415, f"Content-Encoding {encoding!r} is not supported; send gzip or nothing")
            return None
        # Inflated under the same cap: a few KB of zeros must not become gigabytes.
        inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        try:
            out = inflater.decompress(raw, MAX_BODY_BYTES + 1)
            if inflater.unconsumed_tail or len(out) > MAX_BODY_BYTES:
                return None
            out += inflater.flush()
        except zlib.error:
            self._encoding_error = (400, "the request body is not valid gzip")
            return None
        return out if len(out) <= MAX_BODY_BYTES else None

    _encoding_error: tuple[int, str] | None = None

    def _drain(self, length: int) -> None:
        if length > DRAIN_CAP_BYTES:
            self.close_connection = True
            return
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _not_served(self, started: float) -> None:
        # A body sent to a path that is not served is never read, so the
        # connection cannot be reused safely.
        if self.command == "POST":
            self.close_connection = True
        self._json(404, {"error": f"{self._path()} is not served by this gateway"})
        self._done(404, 0, started)

    def _unauthorised(self, started: float) -> None:
        if self.command == "POST":
            self.close_connection = True
        self._json(401, {"error": "a bearer token is required"})
        self._done(401, 0, started, "unauthenticated")

    def do_GET(self):  # noqa: N802 - base class API
        started = time.time()
        path = self._path()
        if path == "/healthz":
            # Open on purpose, and it says nothing: a poller can see the
            # process is alive without learning which models it holds.
            self._json(200, {"ok": True, "service": SERVICE, "version": VERSION})
            return self._done(200, 0, started)
        if path not in FORWARD_GET + OWN_GET + WORKER_GET:
            return self._not_served(started)
        if not self._authorised():
            return self._unauthorised(started)
        if path == "/api/gpu":
            body = json.dumps({"gpu": gpu_info(), "service": SERVICE, "version": VERSION}).encode()
            self._send(200, body)
            return self._done(200, len(body), started)
        t = time.time()
        if path in WORKER_GET:
            status, payload, ctype = upstream("GET", path, None, 30, Config.worker)
        else:
            status, payload, ctype = upstream("GET", path, None, 30)
        self._send(status, payload, ctype, server_ms=(time.time() - t) * 1000)
        return self._done(status, len(payload), started)

    def do_POST(self):  # noqa: N802 - base class API
        started = time.time()
        path = self._path()
        if path not in FORWARD_POST + WORKER_POST:
            return self._not_served(started)
        if not self._authorised():
            return self._unauthorised(started)
        raw = self._read_body()
        if raw is None:
            status, message = self._encoding_error or (413, f"the request body is larger than {MAX_BODY_BYTES} bytes")
            self._encoding_error = None
            self.close_connection = status != 413 or self.close_connection
            self._json(status, {"error": message})
            return self._done(status, 0, started)
        try:
            payload = json.loads(raw or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except ValueError as exc:
            self._json(400, {"error": f"the request body is not a JSON object: {exc}"})
            return self._done(400, 0, started)
        if path == "/api/chat":
            payload["stream"] = False
        t = time.time()
        if path in WORKER_POST:
            status, body, ctype = upstream("POST", path, json.dumps(payload).encode(), Config.worker_timeout,
                                           Config.worker)
        else:
            status, body, ctype = upstream("POST", path, json.dumps(payload).encode(), Config.upstream_timeout)
        self._send(status, body, ctype, server_ms=(time.time() - t) * 1000)
        return self._done(status, len(body), started, _shape(payload, raw))

    # Every other method is not served. BaseHTTPRequestHandler would answer
    # 501 with an HTML page that names the server; a plain 404 says less.
    def _other(self):
        self._not_served(time.time())

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _other  # noqa: N815


def wait_for_ollama(seconds: float = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        status, _, _ = upstream("GET", "/api/version", None, 5)
        if status == 200:
            return True
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ollama", default=Config.ollama)
    ap.add_argument("--worker", default=Config.worker, help="the worker's loopback URL")
    ap.add_argument("--no-token", action="store_true",
                    help="serve without authentication; only sensible on loopback for a local test")
    ap.add_argument("--wait-for-ollama", type=float, default=0,
                    help="seconds to wait for the model server before serving")
    args = ap.parse_args()

    Config.ollama = args.ollama.rstrip("/")
    Config.worker = args.worker.rstrip("/")
    Config.require_token = not args.no_token
    if Config.require_token and not Config.token:
        print(f"FATAL: set {TOKEN_ENV}, or pass --no-token deliberately.", file=sys.stderr)
        return 1
    if not Config.require_token and args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: serving WITHOUT authentication on {args.host}. Anyone who reaches "
              "this port can use the GPU.", file=sys.stderr)
    if args.wait_for_ollama:
        print(f"waiting up to {args.wait_for_ollama:g}s for {Config.ollama} ...", flush=True)
        if not wait_for_ollama(args.wait_for_ollama):
            print(f"FATAL: {Config.ollama} never answered.", file=sys.stderr)
            return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    # A long batch can leave the socket idle for minutes; keepalive stops the
    # OS dropping it mid-request.
    httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    print(f"{SERVICE} {VERSION} listening on {args.host}:{args.port}", flush=True)
    print(f"  upstream   {Config.ollama}", flush=True)
    print(f"  auth       {'bearer token' if Config.require_token else 'NONE'}", flush=True)
    print(f"  serving    {', '.join(FORWARD_GET + OWN_GET + FORWARD_POST)}", flush=True)
    print(f"  worker     {Config.worker}: {', '.join(WORKER_GET + WORKER_POST)}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
