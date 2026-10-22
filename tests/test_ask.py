"""Answering: the event contract, code-built citations, and the SSE response."""
from __future__ import annotations

import pytest

from rag_studio import answer, ingest, profiles, retrieve
from tests.fakes import FakeOllama


@pytest.fixture
def setup(tmp_data):
    fake = FakeOllama(answer="GPO staff visited 166 libraries [1][2]. Also [9].")
    ingest.build(profiles.BASELINE, client=fake)
    return fake, retrieve.Retriever(profiles.BASELINE.id, client=fake)


def test_event_order_and_citations(setup):
    fake, r = setup
    events = list(answer.ask("How many depository libraries did GPO staff visit?",
                             retriever=r, client=fake))
    names = [e for e, _ in events]
    assert names[:5] == ["stage", "stage", "stage", "stage", "ready"]
    assert names[-2:] == ["stage", "done"] and "delta" in names
    assert [d["id"] for e, d in events if e == "stage"] == ["embed", "search", "guard", "prompt", "generate"]
    done = events[-1][1]
    assert done["answer"] == "GPO staff visited 166 libraries [1][2]. Also [9]."
    assert done["citations"]["valid"] == [1, 2] and done["citations"]["invalid"] == [9]
    assert done["citations"]["map"][0]["page"] == done["sources"][0]["page"]
    assert done["refused"] is False
    assert done["usage"]["prompt_tokens"] == 1350 and done["usage"]["truncated"] is False


def test_the_model_is_called_with_the_baseline_settings(setup):
    fake, r = setup
    answer.answer("Q?", retriever=r, client=fake, prompt_version="v2")
    call = fake.chats[-1]
    assert call["think"] is False
    assert call["options"] == {"temperature": 0.0, "num_ctx": 8192}
    assert call["messages"][1]["content"].startswith("Context:\n[1] ")


def test_refusal_is_flagged(setup):
    _, r = setup
    fake = FakeOllama(answer="I could not find this in the documents.")
    done = answer.answer("What is the capital of France?", retriever=r, client=fake)
    assert done["refused"] is True and done["citations"]["numbers"] == []


def test_empty_question_is_an_error_not_a_crash():
    events = list(answer.ask("   "))
    assert events == [("error", {"stage": "input", "error": "Type a question first."})]


def test_api_streams_with_no_transform(client, monkeypatch):
    def fake_ask(question, k=5, **_):
        yield "stage", {"id": "embed"}
        yield "ready", {"sources": []}
        yield "delta", {"text": "Hi"}
        yield "done", {"answer": "Hi"}

    monkeypatch.setattr(answer, "ask", fake_ask)
    with client.stream("POST", "/api/ask", json={"question": "Q?"}) as r:
        assert r.status_code == 200
        assert "no-transform" in r.headers["cache-control"]
        assert r.headers["x-accel-buffering"] == "no"
        body = r.read().decode()
    frames = [f for f in body.split("\n\n") if f]
    assert [f.split("\n")[0] for f in frames] == ["event: stage", "event: ready", "event: delta",
                                                  "event: done"]


def test_api_turns_an_engine_crash_into_an_error_event(client, monkeypatch):
    def broken(question, k=5, **_):
        yield "stage", {"id": "embed"}
        raise RuntimeError("boom")

    monkeypatch.setattr(answer, "ask", broken)
    with client.stream("POST", "/api/ask", json={"question": "Q?"}) as r:
        body = r.read().decode()
    assert body.rstrip().endswith('event: error\ndata: {"error": "boom"}')


def test_api_rejects_an_empty_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
