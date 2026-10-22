"""The one place that talks to Ollama.

Local Ollama and the Colab gateway (M4) speak the same API, so one client
serves both; the gateway only adds a bearer token. Every call has an explicit
timeout, because a model swap on this shared CPU can stall a request for
minutes and nothing should hang waiting on it silently.

A remote client (`remote=True`) differs in what it says and how long it waits
to connect: a Colab runtime that Colab reclaimed should fail in seconds with
"the runtime is not answering", not with advice to start Ollama here.
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from rag_studio.config import OLLAMA_URL

# Status calls are cheap when Ollama is idle. A short timeout keeps Health
# responsive when it is busy loading a model for another app.
STATUS_TIMEOUT = httpx.Timeout(3.0, connect=1.0)
# Model calls can wait minutes: a cold load, or a swap with the trading app's
# qwen3:8b (an embed once took 257 s). `read` is the gap between two bytes, so
# a streamed answer only needs it to cover the prompt-reading pause.
MODEL_TIMEOUT = httpx.Timeout(connect=2.0, read=600.0, write=60.0, pool=5.0)
# Through a Cloudflare tunnel: a slower first byte, and a runtime that has gone
# should still fail fast. Cloudflare itself ends any request at 100 s (HTTP
# 524), which an embedding batch on a GPU never comes near.
REMOTE_STATUS_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
# The read timeout stays just under that limit: a reply the tunnel will never
# deliver is reported as the tunnel's limit, by us, not waited on for minutes.
REMOTE_MODEL_TIMEOUT = httpx.Timeout(connect=10.0, read=95.0, write=60.0, pool=5.0)
# A request body of at least this many bytes is gzipped on its way up: the
# tunnel's upload leg is the slow one, and passages are text (httpx never
# compresses a request by itself).
GZIP_REQUEST_MIN_BYTES = 4 * 1024


@dataclass
class NetStats:
    """What crossed the tunnel, for the trace and the run record (spec §4.3).
    `raw` is the JSON's size, `wire` what was sent after compression."""
    calls: int = 0
    bytes_out_raw: int = 0
    bytes_out_wire: int = 0
    bytes_in_raw: int = 0
    bytes_in_wire: int = 0
    wall_ms: float = 0.0
    server_ms: float = 0.0     # what the far side says it spent (Server-Timing)
    connections: int = 0       # new TCP/TLS connections opened (keep-alive keeps this at 1)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **kw: float) -> None:
        with self._lock:
            self.calls += int(kw.pop("calls", 0))
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {"calls": self.calls, "bytes_out_raw": self.bytes_out_raw, "bytes_out_wire": self.bytes_out_wire,
                    "bytes_in_raw": self.bytes_in_raw, "bytes_in_wire": self.bytes_in_wire,
                    "wall_ms": round(self.wall_ms, 1), "server_ms": round(self.server_ms, 1),
                    "network_ms": round(max(0.0, self.wall_ms - self.server_ms), 1),
                    "connections": self.connections}


def diff_stats(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: round(after[k] - before.get(k, 0), 1) for k in after}


_SERVER_TIMING = re.compile(r"dur=([0-9.]+)")


class OllamaUnavailable(RuntimeError):
    """Ollama did not answer. `reason` is short and safe to show; `kind` says
    what to fix: unreachable, timeout, auth or error."""

    def __init__(self, reason: str, kind: str = "unreachable"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


@dataclass(frozen=True)
class InstalledModel:
    name: str
    digest: str | None
    size: int
    family: str | None
    parameter_size: str | None
    quantization: str | None
    context_length: int | None
    capabilities: tuple[str, ...]
    modified_at: str | None


@dataclass(frozen=True)
class ResidentModel:
    name: str
    size: int
    size_vram: int
    expires_at: str | None
    context_length: int | None


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, token: str | None = None, *, remote: bool = False):
        self.base_url = base_url.rstrip("/")
        self.remote = remote
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._status_timeout = REMOTE_STATUS_TIMEOUT if remote else STATUS_TIMEOUT
        self._model_timeout = REMOTE_MODEL_TIMEOUT if remote else MODEL_TIMEOUT
        self.stats = NetStats()
        self._http: httpx.Client | None = None
        self._http_lock = threading.Lock()
        self._streams: set[int] = set()

    def _client(self) -> httpx.Client:
        """One kept-alive client per runtime: every request through a tunnel
        otherwise pays a fresh TLS handshake. Closed by `close()` (disconnect)."""
        with self._http_lock:
            if self._http is None:
                self._http = httpx.Client(headers=self._headers, limits=httpx.Limits(
                    max_connections=8, max_keepalive_connections=4, keepalive_expiry=120.0))
            return self._http

    def close(self) -> None:
        with self._http_lock:
            if self._http is not None:
                self._http.close()
                self._http = None

    def _send(self, method: str, path: str, *, timeout: httpx.Timeout, body: dict[str, Any] | None = None) -> httpx.Response:
        """One request. A remote client reuses its connection, gzips a large
        body and records sizes and times; a local one is a plain call."""
        url = f"{self.base_url}{path}"
        if not self.remote:
            if method == "GET":
                return httpx.get(url, headers=self._headers, timeout=timeout)
            return httpx.post(url, json=body, headers=self._headers, timeout=timeout)
        http = self._client()
        t = time.perf_counter()
        if method == "GET":
            r = http.get(url, timeout=timeout)
            raw_out = wire_out = 0
        else:
            raw = json.dumps(body).encode()
            headers = {"Content-Type": "application/json"}
            payload = raw
            if len(raw) >= GZIP_REQUEST_MIN_BYTES:
                payload = gzip.compress(raw, compresslevel=5)
                headers["Content-Encoding"] = "gzip"
            r = http.post(url, content=payload, headers=headers, timeout=timeout)
            raw_out, wire_out = len(raw), len(payload)
        wall = (time.perf_counter() - t) * 1000
        # Each connection has one network stream; a stream seen for the first time is a new connection.
        stream = id(r.extensions.get("network_stream"))
        with self._http_lock:
            new_connection = stream not in self._streams
            self._streams.add(stream)
        m = _SERVER_TIMING.search(r.headers.get("server-timing", ""))
        self.stats.add(calls=1, bytes_out_raw=raw_out, bytes_out_wire=wire_out,
                       bytes_in_raw=len(r.content), bytes_in_wire=r.num_bytes_downloaded,
                       wall_ms=wall, server_ms=float(m.group(1)) if m else 0.0,
                       connections=int(new_connection))
        return r

    @property
    def backend(self) -> str:
        """What the UI calls this machine (docs/product-language.md)."""
        return "remote" if self.remote else "local"

    def _down(self, e: Exception, busy: str) -> OllamaUnavailable:
        """A transport failure, worded for the machine it happened on."""
        if self.remote:
            if isinstance(e, httpx.TimeoutException):
                return OllamaUnavailable("The Colab runtime did not answer in time.", "timeout")
            if isinstance(e, httpx.ConnectError):
                return OllamaUnavailable("The Colab runtime is not answering; Colab may have reclaimed it.")
            return OllamaUnavailable(f"The request to the Colab runtime failed ({type(e).__name__}).")
        if isinstance(e, httpx.ConnectError):
            return OllamaUnavailable("Ollama is not running.")
        if isinstance(e, httpx.TimeoutException):
            return OllamaUnavailable(busy, "timeout")
        return OllamaUnavailable(f"Ollama request failed ({type(e).__name__}).")

    def _refused(self, r: httpx.Response, path: str) -> OllamaUnavailable | None:
        """A reply that is not Ollama's: the gateway's lock, or the tunnel's
        own error page once the runtime behind it has gone."""
        if r.status_code in (401, 403):
            return OllamaUnavailable("The Colab runtime refused the token. Connect again with the "
                                     "code the runtime printed.", "auth")
        if r.status_code == 524:
            return OllamaUnavailable("The request took longer than the tunnel's 100 s limit. "
                                     "Send a smaller batch.", "timeout")
        if self.remote and _looks_like_html(r.text):
            return OllamaUnavailable(f"The tunnel answered {path} with an error page (HTTP {r.status_code}); "
                                     "the Colab runtime has probably stopped.")
        return None

    def _get(self, path: str) -> Any:
        try:
            r = self._send("GET", path, timeout=self._status_timeout)
        except httpx.HTTPError as e:
            raise self._down(e, "Ollama did not answer in time; it may be loading a model.") from None
        if r.status_code != 200:
            raise self._refused(r, path) or OllamaUnavailable(
                f"{'The Colab runtime' if self.remote else 'Ollama'} answered {path} with HTTP {r.status_code}.",
                "error")
        return r.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            r = self._send("POST", path, timeout=self._model_timeout, body=body)
        except httpx.HTTPError as e:
            raise self._down(e, "Ollama stopped answering; it may be busy with another app.") from None
        if r.status_code != 200:
            refused = self._refused(r, path)
            if refused:
                raise refused
            raise OllamaError(r.status_code, _error_text(r.text))
        return r.json()

    def get_json(self, path: str) -> Any:
        """A GET through the same transport (the worker's `/worker`)."""
        return self._get(path)

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        """A POST through the same transport and error wording (the worker's `/rerank`)."""
        return self._post(path, body)

    def version(self) -> str:
        return str(self._get("/api/version").get("version", ""))

    def installed(self) -> list[InstalledModel]:
        models = []
        for m in self._get("/api/tags").get("models", []):
            d = m.get("details") or {}
            models.append(InstalledModel(
                name=m.get("name", ""),
                digest=m.get("digest"),
                size=int(m.get("size") or 0),
                family=d.get("family"),
                parameter_size=d.get("parameter_size"),
                quantization=d.get("quantization_level"),
                context_length=d.get("context_length"),
                capabilities=tuple(m.get("capabilities") or ()),
                modified_at=m.get("modified_at"),
            ))
        return sorted(models, key=lambda m: m.name)

    def resident(self) -> list[ResidentModel]:
        return [
            ResidentModel(
                name=m.get("name", ""),
                size=int(m.get("size") or 0),
                size_vram=int(m.get("size_vram") or 0),
                expires_at=m.get("expires_at"),
                context_length=m.get("context_length"),
            )
            for m in self._get("/api/ps").get("models", [])
        ]


    def unload(self, model: str) -> bool:
        """Free a resident model now (`keep_alive: 0`) instead of after its 5-minute keep-alive.
        Returns False, and sends nothing, when the model is not loaded."""
        if not any(same_model(m.name, model) for m in self.resident()):
            return False
        self._post("/api/chat", {"model": model, "messages": [], "keep_alive": 0})
        return True

    def gpu(self) -> dict[str, Any]:
        """The Colab gateway's GPU probe (nvidia-smi). Local Ollama has none."""
        return self._get("/api/gpu").get("gpu") or {}

    def model_info(self, name: str) -> InstalledModel | None:
        """The installed model with this name, or None. Tags compare loosely
        (`x` is `x:latest`); the digest is what proves two are the same."""
        return next((m for m in self.installed() if same_model(m.name, name)), None)

    def embed(self, model: str, texts: list[str], *, dimensions: int | None = None,
              keep_alive: str | int | None = None, options: dict[str, Any] | None = None) -> list[list[float]]:
        """One batch through `/api/embed`.

        `truncate: false` on purpose: by default Ollama silently cuts an input
        longer than the model's context, and the vector then describes only the
        start of the passage. An error we can see beats a vector that lies.
        """
        body: dict[str, Any] = {"model": model, "input": texts, "truncate": False}
        if dimensions is not None:
            body["dimensions"] = dimensions
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        if options:
            body["options"] = options
        out = self._post("/api/embed", body).get("embeddings") or []
        if len(out) != len(texts):
            raise OllamaError(200, f"asked for {len(texts)} embeddings, got {len(out)}")
        return out

    def chat(self, model: str, messages: list[dict[str, str]], *, options: dict[str, Any], think: bool = False,
             format: dict[str, Any] | str | None = None, keep_alive: str | int | None = None) -> dict[str, Any]:
        """One non-streaming `/api/chat`. `format` is a JSON schema (Ollama's structured outputs):
        the reply's `message.content` is then JSON that follows it. Returns Ollama's whole reply,
        so the caller can read `prompt_eval_count`, `eval_count` and `total_duration`.

        `think` is always sent (see chat_stream). A remote `qwen3:8b` call must use one `num_ctx`
        for the whole run: another value makes Ollama reload the model."""
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "think": think,
                                "options": options}
        if format is not None:
            body["format"] = format
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        return self._post("/api/chat", body)

    def chat_stream(self, model: str, messages: list[dict[str, str]], *,
                    options: dict[str, Any], think: bool = False,
                    keep_alive: str | int | None = None) -> Iterator[dict[str, Any]]:
        """`/api/chat` with `stream: true`, one parsed line at a time.

        `think` is always sent. Qwen 3.5 thinks unless told not to, Gemma 4 the
        opposite, and qwen3:4b here ignores `false` entirely (spec §2.3), so
        the caller must never rely on a model's default.

        Closing the generator early closes the connection, which is how Ollama
        learns to stop generating for a reader who has gone.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                "think": think, "options": options}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=body,
                              headers=self._headers, timeout=self._model_timeout) as r:
                if r.status_code != 200:
                    r.read()
                    refused = self._refused(r, "/api/chat")
                    if refused:
                        raise refused
                    raise OllamaError(r.status_code, _error_text(r.text))
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(200, str(data["error"]))
                    yield data
        except httpx.HTTPError as e:
            raise self._down(e, "Ollama stopped answering; it may be busy with another app.") from None


class OllamaError(RuntimeError):
    """Ollama answered, with an error. `message` is Ollama's own text."""

    def __init__(self, status: int, message: str):
        super().__init__(f"Ollama returned HTTP {status}: {message}")
        self.status = status
        self.message = message


def _error_text(body: str) -> str:
    try:
        return str(json.loads(body).get("error") or body)[:300]
    except (ValueError, AttributeError):
        return body[:300]


def _looks_like_html(body: str) -> bool:
    head = body.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def same_model(a: str, b: str) -> bool:
    """`nomic-embed-text` and `nomic-embed-text:latest` are one model."""
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)
