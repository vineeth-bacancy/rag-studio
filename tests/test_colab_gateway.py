"""The Colab gateway: the only thing between a public tunnel and an Ollama
that has no authentication of its own.

The real tools/colab/server.py runs on loopback against a stub Ollama that
records what it receives. A lock fails silently in the direction of letting
people in, so every property is tested over real HTTP, nothing mocked inside
the gateway.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent / "tools" / "colab" / "server.py"
TOKEN = "test-token-0123456789abcdef0123456789abcdef"


def _load():
    spec = importlib.util.spec_from_file_location("colab_gateway", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream():
    received: list[dict] = []

    class Ollama(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def _reply(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            received.append({"method": "GET", "path": self.path})
            if self.path == "/api/tags":
                return self._reply({"models": [{"name": "qwen3-embedding:8b", "digest": "abc"}]})
            return self._reply({"version": "0.32.6"})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            received.append({"method": "POST", "path": self.path, "body": body})
            if self.path == "/api/embed":
                return self._reply({"embeddings": [[0.6, 0.8]] * len(body.get("input") or [])})
            return self._reply({"message": {"content": "ok"}})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, received
    httpd.shutdown()


@pytest.fixture(scope="module")
def gateway(upstream):
    mod = _load()
    mod.Config.ollama = f"http://127.0.0.1:{upstream[0].server_address[1]}"
    mod.Config.token = TOKEN
    mod.Config.require_token = True
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield mod, httpd.server_address[1]
    httpd.shutdown()


def call(port: int, method: str, path: str, body: bytes | None = None, token: str | None = None,
         headers: dict | None = None) -> tuple[int, dict, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = dict(headers or {})
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    if body is not None:
        h.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=h)
    r = conn.getresponse()
    raw = r.read()
    out = (r.status, json.loads(raw) if raw else {}, dict(r.getheaders()))
    conn.close()
    return out


EMBED = json.dumps({"model": "qwen3-embedding:8b", "input": ["a", "b"],
                    "options": {"num_ctx": 1024}}).encode()


# --- the lock ------------------------------------------------------------------------

def test_no_token_is_refused(gateway):
    assert call(gateway[1], "GET", "/api/tags")[0] == 401
    assert call(gateway[1], "POST", "/api/embed", EMBED)[0] == 401


def test_a_wrong_token_is_refused(gateway):
    assert call(gateway[1], "GET", "/api/tags", token="nope")[0] == 401


def test_a_prefix_of_the_token_is_refused(gateway):
    assert call(gateway[1], "GET", "/api/tags", token=TOKEN[:-1])[0] == 401
    assert call(gateway[1], "GET", "/api/tags", token=TOKEN + "x")[0] == 401


def test_the_right_token_is_accepted(gateway):
    status, body, _ = call(gateway[1], "GET", "/api/tags", token=TOKEN)
    assert status == 200 and body["models"][0]["name"] == "qwen3-embedding:8b"


def test_a_scheme_other_than_bearer_is_refused(gateway):
    assert call(gateway[1], "GET", "/api/tags", headers={"Authorization": f"Basic {TOKEN}"})[0] == 401


# --- the allowlist --------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/pull", "/api/push", "/api/create", "/api/delete", "/api/copy",
                                  "/api/generate", "/api/show", "/v1/chat/completions", "/"])
def test_anything_not_allowlisted_is_404_even_with_the_token(gateway, upstream, path):
    before = len(upstream[1])
    for method in ("GET", "POST"):
        status, body, _ = call(gateway[1], method, path, b"{}" if method == "POST" else None, token=TOKEN)
        assert status == 404, (method, path)
        assert "not served" in body["error"]
    assert len(upstream[1]) == before, "a refused path reached Ollama"


def test_an_unknown_path_is_404_before_the_token_is_considered(gateway):
    # Spec §13: 404 before the auth check, so no token is ever weighed for it.
    assert call(gateway[1], "POST", "/api/pull", b"{}")[0] == 404
    assert call(gateway[1], "DELETE", "/api/delete")[0] == 404


def test_a_get_on_a_post_route_is_not_forwarded(gateway):
    assert call(gateway[1], "GET", "/api/embed", token=TOKEN)[0] == 404
    assert call(gateway[1], "POST", "/api/tags", b"{}", token=TOKEN)[0] == 404


@pytest.mark.parametrize("path", ["/api/version", "/api/tags", "/api/ps"])
def test_the_read_only_routes_are_forwarded(gateway, path):
    assert call(gateway[1], "GET", path, token=TOKEN)[0] == 200


def test_the_gpu_probe_needs_the_token_and_answers_honestly(gateway):
    assert call(gateway[1], "GET", "/api/gpu")[0] == 401
    status, body, _ = call(gateway[1], "GET", "/api/gpu", token=TOKEN)
    assert status == 200 and "available" in body["gpu"]


# --- what is forwarded -----------------------------------------------------------------

def test_an_embed_batch_arrives_intact_with_its_options(gateway, upstream):
    status, body, _ = call(gateway[1], "POST", "/api/embed", EMBED, token=TOKEN)
    assert status == 200 and len(body["embeddings"]) == 2
    sent = upstream[1][-1]
    assert sent["path"] == "/api/embed"
    assert sent["body"]["input"] == ["a", "b"] and sent["body"]["options"] == {"num_ctx": 1024}


def test_a_large_reply_is_gzipped_for_a_client_that_accepts_it(gateway, upstream):
    """Vectors are bulky as JSON; the gateway compresses the runtime's leg."""
    import gzip
    mod, port = gateway
    many = json.dumps({"model": "qwen3-embedding:8b", "input": ["x"] * 4000}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/embed", body=many, headers={"Authorization": f"Bearer {TOKEN}",
                                                            "Accept-Encoding": "gzip"})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    assert r.status == 200 and r.getheader("Content-Encoding") == "gzip"
    assert len(json.loads(gzip.decompress(raw))["embeddings"]) == 4000
    # Without Accept-Encoding, plain JSON as before; small replies are never compressed.
    status, body, headers = call(port, "POST", "/api/embed", many, token=TOKEN)
    assert status == 200 and "Content-Encoding" not in headers and len(body["embeddings"]) == 4000
    _, _, small = call(port, "GET", "/api/tags", token=TOKEN, headers={"Accept-Encoding": "gzip"})
    assert "Content-Encoding" not in small


def test_httpx_reads_a_gzipped_reply_transparently(gateway):
    import httpx
    _, port = gateway
    r = httpx.post(f"http://127.0.0.1:{port}/api/embed", headers={"Authorization": f"Bearer {TOKEN}"},
                   json={"model": "m", "input": ["x"] * 4000})
    assert r.headers.get("content-encoding") == "gzip" and len(r.json()["embeddings"]) == 4000


def test_chat_streaming_is_forced_off(gateway, upstream):
    body = json.dumps({"model": "m", "messages": [], "stream": True}).encode()
    assert call(gateway[1], "POST", "/api/chat", body, token=TOKEN)[0] == 200
    assert upstream[1][-1]["body"]["stream"] is False


def test_a_body_that_is_not_a_json_object_is_refused(gateway):
    assert call(gateway[1], "POST", "/api/embed", b"[1, 2]", token=TOKEN)[0] == 400
    assert call(gateway[1], "POST", "/api/embed", b"not json", token=TOKEN)[0] == 400


def test_an_oversized_body_is_refused(gateway):
    mod, port = gateway
    big = b'{"input": "' + b"x" * (mod.MAX_BODY_BYTES + 10) + b'"}'
    assert call(port, "POST", "/api/embed", big, token=TOKEN)[0] == 413


# --- what it says about itself ------------------------------------------------------------

def test_healthz_is_open_and_says_nothing_useful(gateway):
    status, body, _ = call(gateway[1], "GET", "/healthz")
    assert status == 200
    assert set(body) == {"ok", "service", "version"}


def test_no_cors_header_and_no_python_version(gateway):
    _, _, headers = call(gateway[1], "GET", "/api/tags", token=TOKEN)
    assert not any(k.lower().startswith("access-control") for k in headers)
    assert "Python" not in headers.get("Server", "")


def test_a_dead_model_server_is_a_502_with_a_reason(gateway):
    mod, port = gateway
    saved = mod.Config.ollama
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    mod.Config.ollama = f"http://127.0.0.1:{free_port}"
    try:
        status, body, _ = call(port, "GET", "/api/tags", token=TOKEN)
    finally:
        mod.Config.ollama = saved
    assert status == 502 and "not answering" in body["error"]


def test_passages_are_never_written_to_the_log(gateway, capsys):
    secret = "Global growth is projected to slow to 2.3 percent"
    body = json.dumps({"model": "qwen3-embedding:8b", "input": [secret]}).encode()
    call(gateway[1], "POST", "/api/embed", body, token=TOKEN)
    out = capsys.readouterr().out
    assert secret not in out and TOKEN not in out
    assert "inputs=1" in out and "model=qwen3-embedding:8b" in out


def test_it_refuses_to_start_without_a_token(monkeypatch, capsys):
    mod = _load()
    mod.Config.token = ""
    monkeypatch.setattr("sys.argv", ["server.py", "--port", "0"])
    assert mod.main() == 1
    assert "RAG_STUDIO_TOKEN" in capsys.readouterr().err
