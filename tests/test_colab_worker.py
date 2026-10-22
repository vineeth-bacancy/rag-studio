"""The Colab worker (tools/colab/worker.py), tested on a laptop: its HTTP
surface, request limits and NaN guard run for real; the model is a stand-in
(torch and sentence-transformers are imported only when a model is loaded)."""
from __future__ import annotations

import http.client
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from rag_studio import rerank

WORKER = Path(__file__).resolve().parent.parent / "tools" / "colab" / "worker.py"


def _load():
    spec = importlib.util.spec_from_file_location("colab_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


class FakeEngine(mod.Engine):
    """Scores by document length; no torch, no GPU."""

    def __init__(self, scores=None):
        super().__init__(downloaded={"qwen3-4b", "qwen3-0.6b-gpu"})
        self.fixed = scores
        self.loads: list[str] = []

    def _load(self, model_id):
        self.loads.append(model_id)
        self.current = model_id
        self.last_load_s = 0.1
        self.swaps.append({"to": model_id, "at": "now", "seconds": 0.1})

    def _predict(self, spec, instruction, query, documents, max_length):
        return self.fixed if self.fixed is not None else [float(len(d)) for d in documents]

    def _peak_mb(self):
        return 1234


@pytest.fixture
def server():
    engine = FakeEngine()
    httpd = mod.serve(engine, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield engine, httpd.server_address[1]
    httpd.shutdown()


def call(port, method, path, body=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
    r = conn.getresponse()
    out = (r.status, json.loads(r.read() or b"{}"))
    conn.close()
    return out


GOOD = {"model": "qwen3-4b", "instruction": "Find the answer", "query": "Q?", "documents": ["a", "bbb"]}


def test_a_rerank_request_returns_one_score_per_document_and_what_made_them(server):
    engine, port = server
    status, out = call(port, "POST", "/rerank", GOOD)
    assert status == 200 and out["scores"] == [1.0, 3.0]
    assert out["model"] == "Qwen/Qwen3-Reranker-4B" and len(out["revision"]) == 40 and out["dtype"] == "float16"
    assert out["max_memory_mb"] == 1234 and out["swapped_in"] is True and engine.loads == ["qwen3-4b"]
    assert call(port, "POST", "/rerank", GOOD)[1]["swapped_in"] is False          # already resident


def test_another_served_model_swaps_in_and_the_swap_is_reported(server):
    engine, port = server
    call(port, "POST", "/rerank", GOOD)
    status, out = call(port, "POST", "/rerank", GOOD | {"model": "qwen3-0.6b-gpu"})
    assert status == 200 and out["swapped_in"] is True and out["model"] == "Qwen/Qwen3-Reranker-0.6B"
    info = call(port, "GET", "/worker")[1]
    assert info["loaded"] == "qwen3-0.6b-gpu" and [s["to"] for s in info["swaps"]] == ["qwen3-4b", "qwen3-0.6b-gpu"]
    assert "packages" in info and info["served"] == ["qwen3-0.6b-gpu", "qwen3-4b"]


def test_a_model_that_was_not_downloaded_is_refused_not_downloaded_in_a_request(server):
    status, out = call(server[1], "POST", "/rerank", GOOD | {"model": "qwen3-8b"})
    assert status == 409 and "setup cell" in out["error"] and server[0].loads == []
    assert call(server[1], "POST", "/rerank", GOOD | {"model": "gpt"})[0] == 400


@pytest.mark.parametrize("change", [
    {"documents": []}, {"documents": ["x"] * 65}, {"documents": ["ok", ""]}, {"documents": "text"},
    {"documents": ["x" * 12_001]}, {"query": ""}, {"query": "q" * 2001}, {"instruction": ""},
    {"instruction": "i" * 601}, {"max_length": 10}, {"max_length": 99999}, {"model": 5},
])
def test_requests_outside_the_limits_are_400(server, change):
    status, out = call(server[1], "POST", "/rerank", GOOD | change)
    assert status == 400 and out["error"]
    assert server[0].loads == []


def test_an_out_of_memory_swap_is_a_507_that_unloads_and_recovers():
    class OutOfMemoryError(RuntimeError):
        pass

    class Tight(FakeEngine):
        boom = True

        def _load(self, model_id):
            if model_id == "qwen3-0.6b-gpu" and self.boom:
                raise OutOfMemoryError("CUDA out of memory")
            super()._load(model_id)

    engine = Tight()
    httpd = mod.serve(engine, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        assert call(port, "POST", "/rerank", GOOD)[0] == 200 and engine.current == "qwen3-4b"
        status, out = call(port, "POST", "/rerank", GOOD | {"model": "qwen3-0.6b-gpu"})
        assert status == 507 and "ran out of memory" in out["error"] and engine.current is None
        engine.boom = False
        assert call(port, "POST", "/rerank", GOOD | {"model": "qwen3-0.6b-gpu"})[0] == 200      # recovers
    finally:
        httpd.shutdown()


def test_a_swap_frees_the_old_model_before_loading_the_next():
    # the real _load calls _free first; check the source rather than a GPU
    src = WORKER.read_text()
    assert src.index("self._free()\n        t = time.perf_counter()") < src.index("self.model = CrossEncoder(")


def test_nan_scores_are_a_500_with_a_plain_message():
    engine = FakeEngine(scores=[1.0, float("nan")])
    httpd = mod.serve(engine, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, out = call(httpd.server_address[1], "POST", "/rerank", GOOD)
    finally:
        httpd.shutdown()
    assert status == 500 and "NaN or infinite" in out["error"] and "scores" not in out


def test_bad_bodies_and_other_paths(server):
    port = server[1]
    assert call(port, "POST", "/rerank", raw=b"not json")[0] == 400
    assert call(port, "POST", "/rerank", raw=b"[1]")[0] == 400
    assert call(port, "POST", "/rerank")[0] == 400                          # no body
    assert call(port, "POST", "/other", {})[0] == 404
    assert call(port, "GET", "/rerank")[0] == 404
    assert call(port, "GET", "/api/tags")[0] == 404


def test_it_binds_to_loopback_only_and_has_no_auth_of_its_own():
    with pytest.raises(SystemExit, match="loopback"):
        mod.serve(FakeEngine(), 0, host="0.0.0.0")
    httpd = mod.serve(FakeEngine(), 0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()
    assert "Authorization" not in WORKER.read_text()


def test_the_log_has_counts_and_never_content(server, capsys):
    call(server[1], "POST", "/rerank", GOOD | {"query": "SECRET QUESTION", "documents": ["SECRET PASSAGE"]})
    # The server thread logs after it has replied, so under load the line can arrive a moment later.
    out = ""
    for _ in range(50):
        out += capsys.readouterr().out
        if "documents=" in out:
            break
        time.sleep(0.02)
    assert "documents=1" in out and "SECRET" not in out


def test_the_worker_serves_exactly_the_pinned_models_the_laptop_records():
    """The worker and rerank.RERANKERS name the same weights: same repo, same
    revision, same dtype and quantisation. A run records the laptop's table, so
    the two must not drift."""
    for model_id, spec in mod.MODELS.items():
        mine = rerank.RERANKERS[model_id]
        assert (spec["model"], spec["revision"], spec["dtype"], spec["quantization"]) == \
            (mine.model, mine.revision, mine.dtype, mine.quantization), model_id
        assert spec["batch_size"] == mine.batch_size and mine.where == "colab"
    assert {s.id for s in rerank.RERANKERS.values() if s.where == "colab"} == set(mod.MODELS)


def test_an_unknown_reranker_at_start_up_is_fatal(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["worker.py", "--port", "0", "--models", "gpt"])
    assert mod.main() == 1 and "unknown reranker" in capsys.readouterr().err


# ------------------------------------------------------------------ POST /context

KEY = "a" * 64


class FakeOllama:
    def __init__(self, text="A line.", tokens=500, delay=0.0):
        self.calls = []
        self.text, self.tokens, self.delay = text, tokens, delay

    def chat(self, body):
        import time
        time.sleep(self.delay)
        self.calls.append(body)
        return {"message": {"content": self.text}, "prompt_eval_count": self.tokens, "eval_count": 12}


def ctx_payload(n=2, **over):
    base = {"model": "qwen3:8b", "num_ctx": 8192, "system": "sys", "prefix": "<document>\nW\n</document>\n<passage>\n",
            "suffix": "\n</passage>\nWrite.", "chunks": [{"key": f"{i:064x}", "text": f"passage {i}"} for i in range(n)]}
    return base | over


def ctx_server(fake, **kw):
    engine = mod.ContextEngine(chat=fake.chat, digest=lambda m: "sha256:8b")
    httpd = mod.serve(FakeEngine(), 0, context=engine)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return engine, httpd


def test_a_context_request_writes_one_line_per_passage_through_ollama():
    fake = FakeOllama()
    engine, httpd = ctx_server(fake)
    try:
        status, out = call(httpd.server_address[1], "POST", "/context", ctx_payload(3))
    finally:
        httpd.shutdown()
    assert status == 200 and len(out["lines"]) == 3 and out["remaining"] == [] and out["digest"] == "sha256:8b"
    sent = fake.calls[0]
    assert sent["model"] == "qwen3:8b" and sent["think"] is False and sent["stream"] is False
    assert sent["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 160}
    assert sent["messages"][1]["content"] == "<document>\nW\n</document>\n<passage>\npassage 0\n</passage>\nWrite."
    assert [c["messages"][1]["content"].startswith("<document>") for c in fake.calls] == [True] * 3


def test_a_retry_redoes_nothing():
    fake = FakeOllama()
    engine, httpd = ctx_server(fake)
    port = httpd.server_address[1]
    try:
        call(port, "POST", "/context", ctx_payload(3))
        status, out = call(port, "POST", "/context", ctx_payload(3))
    finally:
        httpd.shutdown()
    assert status == 200 and len(fake.calls) == 3                       # the second request cost no model call
    assert all(v.get("cached_on_runtime") for v in out["per_line"].values())


def test_fresh_asks_again_and_remembers_nothing():
    fake = FakeOllama()
    engine, httpd = ctx_server(fake)
    port = httpd.server_address[1]
    try:
        call(port, "POST", "/context", ctx_payload(2, fresh=True))
        call(port, "POST", "/context", ctx_payload(2, fresh=True))
    finally:
        httpd.shutdown()
    assert len(fake.calls) == 4 and engine.done == {}


def test_the_time_cap_counts_from_arrival_and_returns_what_is_left(monkeypatch):
    fake = FakeOllama()
    engine = mod.ContextEngine(chat=fake.chat, digest=lambda m: None)
    out = engine.run(ctx_payload(4), arrived=__import__("time").time() - 100)      # waited 100 s in the queue
    assert out["lines"] == {} and len(out["remaining"]) == 4 and fake.calls == []
    out = engine.run(ctx_payload(4), arrived=__import__("time").time())
    assert len(out["lines"]) == 4 and out["remaining"] == []


@pytest.mark.parametrize("change", [
    {"model": "qwen3:4b"}, {"num_ctx": 4096}, {"chunks": []}, {"chunks": [{"key": "k", "text": "x"}]},
    {"chunks": [{"key": KEY, "text": ""}]}, {"system": ""}, {"prefix": "p" * 60_001},
    {"chunks": [{"key": f"{i:064x}", "text": "x"} for i in range(17)]},
])
def test_context_requests_outside_the_limits_are_400(change):
    fake = FakeOllama()
    engine, httpd = ctx_server(fake)
    try:
        status, out = call(httpd.server_address[1], "POST", "/context", ctx_payload() | change)
    finally:
        httpd.shutdown()
    assert status == 400 and out["error"] and fake.calls == []


def test_a_cut_prompt_is_an_error_not_a_line():
    fake = FakeOllama(tokens=20)                      # a long prompt reported as 20 tokens: the start was dropped
    engine, httpd = ctx_server(fake)
    try:
        status, out = call(httpd.server_address[1], "POST", "/context",
                           ctx_payload(1, prefix="<document>\n" + "w" * 5000 + "\n</document>\n<passage>\n"))
    finally:
        httpd.shutdown()
    assert status == 500 and "cut by the model's context" in out["error"] and engine.done == {}


def test_a_worker_without_the_context_job_says_so(server):
    status, out = call(server[1], "POST", "/context", ctx_payload())
    assert status == 409 and "setup cell" in out["error"]


def test_context_lines_are_never_logged(capsys):
    fake = FakeOllama(text="SECRET LINE")
    engine, httpd = ctx_server(fake)
    try:
        call(httpd.server_address[1], "POST", "/context", ctx_payload(1, prefix="<document>\nSECRET WINDOW\n</document>\n"))
    finally:
        httpd.shutdown()
    out = capsys.readouterr().out
    assert "SECRET" not in out and "chunks=1" in out


def test_a_changed_prompt_is_not_answered_from_the_runtimes_memory():
    """Found live: lines written under an old prompt came back for a new one."""
    fake = FakeOllama(text="First style.")
    engine = mod.ContextEngine(chat=fake.chat, digest=lambda m: None)
    import time
    first = engine.run(ctx_payload(1), time.time())
    fake.text = "Second style."
    same_prompt = engine.run(ctx_payload(1), time.time())
    new_system = engine.run(ctx_payload(1, system="a different system prompt"), time.time())
    new_suffix = engine.run(ctx_payload(1, suffix="\n</passage>\nWrite differently."), time.time())
    assert list(first["lines"].values()) == list(same_prompt["lines"].values()) == ["First style."]
    assert list(new_system["lines"].values()) == ["Second style."] == list(new_suffix["lines"].values())
    assert len(fake.calls) == 3
