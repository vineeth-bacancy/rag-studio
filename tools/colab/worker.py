#!/usr/bin/env python3
"""The GPU worker on a Google Colab runtime: reranking (and, later, context lines).

    laptop --> tunnel --> gateway (server.py, token, allowlist) --> THIS (127.0.0.1:8001)

Ollama cannot rerank (it has no rerank route), so the reranker runs here, in
Python, on the GPU. This process has **no authentication of its own and binds
to loopback only**: the gateway is the lock, and the only thing that can reach
it. Standard library for the server; torch and sentence-transformers are
imported **lazily**, so this file loads (and is tested) on a laptop with
neither.

* ``POST /rerank``  ``{model, instruction, query, documents[], max_length}`` ->
  ``{scores[], compute_ms, model, revision, dtype, ...}``. One score per
  document: the yes-minus-no logit difference, the same number the model card
  and the laptop's CrossEncoder compute. NaN or infinity is a 500 with a plain
  message, never a quietly wrong order.
* ``GET /worker``  what is loaded, VRAM, and **package versions**, so a run can
  record what produced its numbers.

Models are allowlisted and **pinned to a revision** (the same table as
``rag_studio/rerank.py``; a test keeps the two equal). One model is resident at
a time. A request for another listed model swaps it (the swap is recorded and
reported); a model that was not downloaded at start-up is refused rather than
downloaded inside a request, because a request the tunnel holds for more than
100 s is killed (HTTP 524).

* ``POST /context``  a batch of context lines (contextual retrieval). The laptop
  sends one shared prefix (the document window) and up to 16 passages; this
  process asks the **local Ollama** (loopback) for one line per passage, in order,
  so Ollama reuses its work on the shared prefix. It is **idempotent**: a
  finished line is kept here by its key, so a retry after a timeout redoes
  nothing, and **its time cap counts from the request's arrival** (queue wait
  included), so a reply always comes back inside the tunnel's 100 s: what was
  not finished is returned as ``remaining`` and asked for again.

Nothing about content is logged: sizes and counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICE = "rag-studio-worker"
VERSION = "1"

# id -> (Hugging Face repo, pinned revision, dtype, quantization, batch size).
# Resolved from the HF API on 2026-10-16. `local` is not served here: the
# laptop runs fp32 itself. `qwen3-0.6b-gpu` is the same weights in fp16.
MODELS: dict[str, dict] = {
    "qwen3-0.6b-gpu": {"model": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                       "dtype": "float16", "quantization": None, "batch_size": 8},
    "qwen3-4b": {"model": "Qwen/Qwen3-Reranker-4B", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
                 "dtype": "float16", "quantization": None, "batch_size": 8},
    "qwen3-8b": {"model": "Qwen/Qwen3-Reranker-8B", "revision": "77d193c791ed757ca307ee72715aa132723da912",
                 "dtype": "float16", "quantization": "int8", "batch_size": 8},
}

CONTEXT_MODEL = "qwen3:8b"            # the one chat model /context will use
CONTEXT_NUM_CTX = 8192                # one value for every call: another makes Ollama reload the model
CONTEXT_BATCH_CAP_S = 45.0            # from arrival, queue wait included
CONTEXT_MAX_CHUNKS = 16
CONTEXT_MAX_PREFIX = 60_000
CONTEXT_MAX_CHUNK = 8_000
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_DOCUMENTS = 64
MAX_DOCUMENT_CHARS = 12_000
MAX_QUERY_CHARS = 2_000
MAX_INSTRUCTION_CHARS = 600
MAX_LENGTH_LIMIT = 2048
DEFAULT_MAX_LENGTH = 1024


class WorkerError(Exception):
    """A request that cannot be served. `message` is shown as is."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def package_versions() -> dict[str, str]:
    """From package metadata, so nothing heavy is imported to answer /worker."""
    from importlib.metadata import PackageNotFoundError, version
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "sentence-transformers", "transformers", "bitsandbytes", "accelerate"):
        try:
            out[name.replace("-", "_")] = version(name)
        except PackageNotFoundError:
            out[name.replace("-", "_")] = "not installed"
    return out


def vram() -> dict | None:
    """GPU memory, only once torch is loaded (a model is resident); never imports it."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(0), "total_mb": round(total / 2**20),
                "used_mb": round((total - free) / 2**20),
                "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    except Exception:  # noqa: BLE001
        return None


class Engine:
    """The one resident model, behind a lock (one inference at a time: two at
    once would each be slower and could run the GPU out of memory)."""

    def __init__(self, downloaded: set[str] | None = None, device: str = "cuda"):
        self.device = device            # "cpu" only to test the real code path on a laptop
        self.lock = threading.Lock()
        self.current: str | None = None
        self.model: Any = None
        self.downloaded = set(downloaded or ())
        self.swaps: list[dict] = []
        self.last_load_s: float | None = None

    def served(self) -> list[str]:
        return sorted(self.downloaded)

    def _free(self) -> None:
        """Drop the resident model and give its GPU memory back. A swap must do
        this BEFORE loading the next model: 4B (9 GB) and 8B (9.5 GB) together
        do not fit a 15 GB T4 (the first live swap ran out of memory)."""
        import gc
        self.model = None
        self.current = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and self.device == "cuda":
            torch.cuda.empty_cache()

    def _load(self, model_id: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder
        spec = MODELS[model_id]
        self._free()
        t = time.perf_counter()
        kwargs: dict = {"dtype": getattr(torch, spec["dtype"] if self.device != "cpu" else "float32")}
        if spec["quantization"] == "int8":
            from transformers import BitsAndBytesConfig
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "dtype": torch.float16}
        self.model = CrossEncoder(spec["model"], revision=spec["revision"], device=self.device,
                                  model_kwargs=kwargs)
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.last_load_s = round(time.perf_counter() - t, 1)
        self.swaps.append({"to": model_id, "at": time.strftime("%H:%M:%S"), "seconds": self.last_load_s})
        self.current = model_id

    def preload(self, ids: list[str]) -> None:
        """Download every model (start-up, not a request) and load the first."""
        for n, model_id in enumerate(ids):
            from huggingface_hub import snapshot_download
            spec = MODELS[model_id]
            try:
                snapshot_download(spec["model"], revision=spec["revision"])
            except Exception as exc:  # noqa: BLE001
                if n == 0:
                    raise                      # the first model is the one asked for: fail loudly
                print(f"could not download {model_id} ({type(exc).__name__}); it will not be served", flush=True)
                continue
            self.downloaded.add(model_id)
        if ids:
            with self.lock:
                self._load(ids[0])

    def _predict(self, spec: dict, instruction: str, query: str, documents: list[str], max_length: int) -> list[float]:
        """The scores from the resident model (the one place torch is touched, so tests can replace it)."""
        import torch
        self.model.max_seq_length = max_length
        with torch.inference_mode():
            raw = self.model.predict([(query, d) for d in documents], prompt=instruction,
                                     batch_size=spec["batch_size"], show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in raw]

    def _peak_mb(self) -> int:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2**20) if self.device == "cuda" else 0

    def rerank(self, model_id: str, instruction: str, query: str, documents: list[str],
               max_length: int) -> dict:
        if model_id not in MODELS:
            raise WorkerError(400, f"Unknown reranker {model_id!r}. Known: {', '.join(MODELS)}.")
        if model_id not in self.downloaded:
            raise WorkerError(409, f"{model_id} was not downloaded when this runtime started. "
                                   "Run the setup cell again with it in the worker list.")
        with self.lock:
            swapped = False
            try:
                if self.current != model_id:
                    self._load(model_id)
                    swapped = True
                spec = MODELS[model_id]
                t = time.perf_counter()
                scores = self._predict(spec, instruction, query, documents, max_length)
                compute_ms = (time.perf_counter() - t) * 1000
                peak = self._peak_mb()
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ != "OutOfMemoryError":
                    if swapped or self.current != model_id:
                        self._free()          # a half-loaded model must not stay resident
                    raise
                self._free()
                raise WorkerError(507, f"The GPU ran out of memory with {model_id}. The model was unloaded; the "
                                       "next request loads it again. Try a smaller reranker or fewer passages.") from None
        check_finite(scores)
        return {"scores": scores, "compute_ms": round(compute_ms, 1), "model": spec["model"],
                "revision": spec["revision"], "dtype": spec["dtype"], "quantization": spec["quantization"],
                "max_memory_mb": peak, "swapped_in": swapped, "load_s": self.last_load_s if swapped else None}

    def info(self) -> dict:
        spec = MODELS.get(self.current or "")
        return {"service": SERVICE, "version": VERSION, "loaded": self.current,
                "loaded_spec": spec, "served": self.served(), "swaps": self.swaps[-10:],
                "vram": vram(), "packages": package_versions()}


def ollama_chat(body: dict) -> dict:
    """One non-streaming chat on the runtime's own Ollama (loopback)."""
    import urllib.request
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def ollama_digest(model: str) -> str | None:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") in (model, f"{model}:latest"):
                    return m.get("digest")
    except Exception:  # noqa: BLE001
        return None
    return None


class ContextEngine:
    """Writes context lines through the runtime's Ollama, remembering each by key."""

    def __init__(self, chat=ollama_chat, digest=ollama_digest):
        self.chat = chat
        self.digest_of = digest
        self.lock = threading.Lock()          # one batch at a time: Ollama serves one request at a time anyway
        self.done: dict[str, dict] = {}
        self._digest: str | None = None

    def run(self, payload: dict, arrived: float) -> dict:
        if payload.get("model") != CONTEXT_MODEL:
            raise WorkerError(400, f"Context lines are written by {CONTEXT_MODEL} only.")
        if payload.get("num_ctx") != CONTEXT_NUM_CTX:
            raise WorkerError(400, f"num_ctx must be {CONTEXT_NUM_CTX}: another value makes Ollama reload the model.")
        system, prefix, suffix = payload.get("system"), payload.get("prefix"), payload.get("suffix")
        chunks = payload.get("chunks")
        if not (isinstance(system, str) and 0 < len(system) <= 2000 and isinstance(suffix, str)
                and 0 < len(suffix) <= 2000 and isinstance(prefix, str) and 0 < len(prefix) <= CONTEXT_MAX_PREFIX):
            raise WorkerError(400, "system, prefix and suffix must be non-empty strings within their limits.")
        if (not isinstance(chunks, list) or not 0 < len(chunks) <= CONTEXT_MAX_CHUNKS
                or not all(isinstance(c, dict) and isinstance(c.get("key"), str) and 8 <= len(c["key"]) <= 128
                           and isinstance(c.get("text"), str) and 0 < len(c["text"]) <= CONTEXT_MAX_CHUNK for c in chunks)):
            raise WorkerError(400, f"chunks must be 1 to {CONTEXT_MAX_CHUNKS} items with a key and text.")
        # `fresh` asks for new generations and remembers none (the determinism check).
        fresh = payload.get("fresh") is True
        lines: dict[str, str] = {}
        per_line: dict[str, dict] = {}
        remaining: list[str] = []
        with self.lock:                        # waiting here counts against the cap: `arrived` is the clock
            if self._digest is None:
                self._digest = self.digest_of(CONTEXT_MODEL)
            # The runtime's memory is keyed by the passage's key AND the prompt that wrote the line:
            # the same passage asked with another prompt is another question (found live, 2026-10-16:
            # a changed prompt got the old prompt's lines back).
            prompt_id = hashlib.sha256(json.dumps([system, suffix]).encode()).hexdigest()[:16]
            for c in chunks:
                key = c["key"]
                memo = f"{prompt_id}:{key}"
                if memo in self.done and not fresh:
                    lines[key] = self.done[memo]["line"]
                    per_line[key] = {**self.done[memo]["info"], "cached_on_runtime": True}
                    continue
                if time.time() - arrived > CONTEXT_BATCH_CAP_S:
                    remaining.append(key)
                    continue
                t = time.perf_counter()
                user = prefix + c["text"] + suffix
                out = self.chat({"model": CONTEXT_MODEL, "stream": False, "think": False, "keep_alive": "30m",
                                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 "options": {"temperature": 0.0, "num_ctx": CONTEXT_NUM_CTX, "num_predict": 160}})
                text = ((out.get("message") or {}).get("content") or "").strip()
                tokens = out.get("prompt_eval_count")
                if tokens and (len(system) + len(user)) / tokens > 8:
                    raise WorkerError(500, "The prompt was cut by the model's context, so the line would describe "
                                           "the wrong text. Use a smaller window.")
                if not text:
                    raise WorkerError(500, "The model returned no text for a passage.")
                info = {"prompt_tokens": tokens, "eval_tokens": out.get("eval_count"),
                        "ms": round((time.perf_counter() - t) * 1000, 1)}
                if not fresh:
                    self.done[memo] = {"line": text, "info": info}
                lines[key] = text
                per_line[key] = info
        return {"lines": lines, "remaining": remaining, "digest": self._digest, "per_line": per_line,
                "model": CONTEXT_MODEL, "num_ctx": CONTEXT_NUM_CTX}


def check_finite(scores: list[float]) -> None:
    import math
    if any(not math.isfinite(x) for x in scores):
        raise WorkerError(500, "The reranker produced NaN or infinite scores on this GPU, so its order "
                               "cannot be trusted. Use the reranker on this computer instead.")


def validate(payload: dict) -> tuple[str, str, str, list[str], int]:
    """A /rerank request, checked. Raises WorkerError(400) with what to fix."""
    model = payload.get("model")
    instruction = payload.get("instruction")
    query = payload.get("query")
    documents = payload.get("documents")
    max_length = payload.get("max_length", DEFAULT_MAX_LENGTH)
    if not isinstance(model, str):
        raise WorkerError(400, "model must be a string.")
    if not isinstance(instruction, str) or not 0 < len(instruction) <= MAX_INSTRUCTION_CHARS:
        raise WorkerError(400, f"instruction must be 1 to {MAX_INSTRUCTION_CHARS} characters.")
    if not isinstance(query, str) or not 0 < len(query.strip()) <= MAX_QUERY_CHARS:
        raise WorkerError(400, f"query must be 1 to {MAX_QUERY_CHARS} characters.")
    if (not isinstance(documents, list) or not 0 < len(documents) <= MAX_DOCUMENTS
            or not all(isinstance(d, str) and d for d in documents)):
        raise WorkerError(400, f"documents must be 1 to {MAX_DOCUMENTS} non-empty strings.")
    if any(len(d) > MAX_DOCUMENT_CHARS for d in documents):
        raise WorkerError(400, f"a document is longer than {MAX_DOCUMENT_CHARS} characters.")
    if not isinstance(max_length, int) or not 64 <= max_length <= MAX_LENGTH_LIMIT:
        raise WorkerError(400, f"max_length must be 64 to {MAX_LENGTH_LIMIT}.")
    return model, instruction, query, documents, max_length


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE}/{VERSION}"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    engine: Engine = None  # type: ignore[assignment]  # set by main() and by tests
    context: "ContextEngine | None" = None

    def log_message(self, format, *args):  # noqa: A002
        del format, args

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _done(self, status: int, started: float, note: str = "") -> None:
        print(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]} -> {status} "
              f"{time.time() - started:.1f}s {note}".rstrip(), flush=True)

    def do_GET(self):  # noqa: N802
        started = time.time()
        if self.path.split("?")[0].rstrip("/") == "/worker":
            self._json(200, {**self.engine.info(), "context": self.context is not None})
            return self._done(200, started)
        self._json(404, {"error": "not served"})
        return self._done(404, started)

    def do_POST(self):  # noqa: N802
        started = time.time()
        route = self.path.split("?")[0].rstrip("/")
        if route not in ("/rerank", "/context"):
            self.close_connection = True
            self._json(404, {"error": "not served"})
            return self._done(404, started)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY_BYTES:
            self.close_connection = True
            self._json(413 if length > 0 else 400, {"error": "the request body is missing or too large"})
            return self._done(413 if length > 0 else 400, started)
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except ValueError as exc:
            self._json(400, {"error": f"the request body is not a JSON object: {exc}"})
            return self._done(400, started)
        try:
            if route == "/context":
                if self.context is None:
                    raise WorkerError(409, "This runtime was started without context lines. Run the setup cell "
                                           "with the context job.")
                result = self.context.run(payload, started)
                self._json(200, result)
                return self._done(200, started, f"model={payload.get('model')} chunks={len(payload['chunks'])} "
                                                f"written={len(result['lines'])} remaining={len(result['remaining'])}")
            model, instruction, query, documents, max_length = validate(payload)
            result = self.engine.rerank(model, instruction, query, documents, max_length)
        except WorkerError as exc:
            self._json(exc.status, {"error": exc.message})
            return self._done(exc.status, started, f"model={payload.get('model')}")
        except Exception as exc:  # noqa: BLE001 - the laptop gets a reason, not a dropped socket
            self._json(500, {"error": f"the worker failed: {type(exc).__name__}"})
            return self._done(500, started, type(exc).__name__)
        self._json(200, result)
        return self._done(200, started, f"model={model} documents={len(documents)} compute={result['compute_ms']:.0f}ms")


def serve(engine: Engine, port: int, host: str = "127.0.0.1", context: "ContextEngine | None" = None
          ) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("The worker has no authentication, so it binds to loopback only.")
    handler = type("BoundHandler", (Handler,), {"engine": engine, "context": context})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", 8001)))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                    help="cpu only to try the real code path on a laptop (float32); a runtime always uses cuda")
    ap.add_argument("--context", action="store_true", help="also serve POST /context through the local Ollama")
    ap.add_argument("--models", default="", help="comma-separated reranker ids to download and serve; the first is loaded")
    args = ap.parse_args()
    ids = [m for m in args.models.split(",") if m]
    if args.context and not ids:
        print("note: a context-only worker holds no reranker", flush=True)
    unknown = [m for m in ids if m not in MODELS]
    if unknown:
        print(f"FATAL: unknown reranker(s) {', '.join(unknown)}. Known: {', '.join(MODELS)}.", file=sys.stderr)
        return 1
    engine = Engine(device=args.device)
    print(f"{SERVICE} {VERSION}: downloading and loading {', '.join(ids) or 'nothing'} ...", flush=True)
    engine.preload(ids)
    httpd = serve(engine, args.port, context=ContextEngine() if args.context else None)
    print(f"{SERVICE} {VERSION} listening on 127.0.0.1:{args.port} (serving {', '.join(engine.served())})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
