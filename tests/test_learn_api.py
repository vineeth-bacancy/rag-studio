"""How it works API (M3) and the search settings on Ask and runs: input
validation and response shapes, on a fake-embedded build of the iMac report."""
from __future__ import annotations

import pytest

from rag_studio import ingest, learn, paths, profiles, retrieve
from tests.fakes import FakeOllama

IMAC = "apple_imac_per_2024_10"


@pytest.fixture
def built(tmp_data, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    r = retrieve.Retriever(profiles.PYMUPDF_STRUCTURE.id, client=fake)
    monkeypatch.setattr(learn, "retriever", lambda profile_id: r)
    return fake


def test_chunk_lab_cuts_a_page(client):
    r = client.get("/api/learn/chunk", params={"doc_id": IMAC, "page": 10, "method": "fixed",
                                                "size": 400, "overlap": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["passages"] and body["baseline"]
    assert all(not p["starts_mid_word"] for p in body["passages"])
    text = body["text"]
    assert all(0 <= p["start"] < p["end"] <= len(text) for p in body["passages"])


@pytest.mark.parametrize("params, status", [
    ({"method": "bogus"}, 400),
    ({"size": 50}, 422),
    ({"page": 99}, 404),
    ({"doc_id": "nope"}, 404),
])
def test_chunk_lab_refuses_bad_input(client, params, status):
    q = {"doc_id": IMAC, "page": 10, "method": "structure"} | params
    assert client.get("/api/learn/chunk", params=q).status_code == status


def test_search_lab_returns_three_lists(client, built):
    r = client.get("/api/learn/search", params={"q": "How much recycled content does the iMac contain?",
                                                 "profile": "pymupdf-structure", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["test_question"] == "B5" and body["targets"][0]["doc_id"] == IMAC
    assert set(body["lists"]) == {"dense", "bm25", "hybrid"}
    hybrid = body["lists"]["hybrid"]["hits"]
    assert len(hybrid) == 5 and set(hybrid[0]["ranks"]) == {"dense", "bm25"}
    assert isinstance(hybrid[0]["answer_page"], bool)


def test_store_facts_describe_the_build(client, built):
    body = client.get("/api/learn/store", params={"profile": "pymupdf-structure"}).json()
    assert body["rows"] > 0 and body["dimensions"] == 768
    assert any(i["type"] == "FTS" for i in body["indexes"])
    assert {c["name"] for c in body["columns"]} >= {"vector", "search_text", "label"}
    assert body["builds"][0]["active"] is True


def test_unknown_profile_is_a_400(client):
    assert client.get("/api/learn/store", params={"profile": "nope"}).status_code == 400


def test_ask_and_runs_validate_the_search_method(client):
    assert client.post("/api/ask", json={"question": "Q?", "search": "sparse"}).status_code == 422
    assert client.post("/api/ask", json={"question": "Q?", "section_cap": 0}).status_code == 422
    assert client.post("/api/eval/runs", json={"sets": ["set-b"], "search": "vector"}).status_code == 422


def test_indexes_say_which_can_search_by_words(client):
    body = client.get("/api/indexes").json()
    assert [m["id"] for m in body["search_methods"]] == ["dense", "bm25", "hybrid"]
    p = next(p for p in body["profiles"] if p["id"] == "pymupdf-structure")
    assert p["labels"] is True and p["chunker"] == "structure"


def test_chunk_lab_semantic_returns_its_curve(client, tmp_data, monkeypatch):
    """Through the API, with an empty sentence cache (it once crashed there)."""
    from rag_studio import embed

    fake = FakeOllama()
    monkeypatch.setattr(learn, "Embedder", lambda spec: embed.Embedder(spec, client=fake))
    r = client.get("/api/learn/chunk", params={"doc_id": "wb_gep_2026_06", "page": 19,
                                                "method": "semantic", "percentile": 80})
    assert r.status_code == 200, r.text
    body = r.json()
    # Fake vectors are random: similarities sit near 0, so only the range is checked.
    assert body["curve"] and -1 <= body["threshold"] <= 1 and body["model"] == "nomic-embed-text:latest"
    assert len(body["curve"]) < len(body["sentences"])      # one gap fewer than sentences, per block
    # Every cut ends a passage; a piece over the size limit is split again.
    assert sum(c["cut"] for c in body["curve"]) <= len(body["passages"]) - 1
    assert fake.embed_calls >= 1
