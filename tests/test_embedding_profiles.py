"""M4's index profiles: recorded configs, where each builds and is queried,
resumable builds, the guards around a build, and the vector match check.

Builds run the real ingest on the real PDFs with fake models (small vectors)
into a temporary data folder, like tests/test_index.py.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rag_studio import embed, eval as evaluation, ingest, profiles, retrieve, store
from rag_studio.embed import Embedder, ProfileMismatch, RemoteUnavailable
from tests.fakes import FakeOllama

OLD_KEYS = {"model", "dimensions", "doc_prefix", "query_prefix", "backend", "batch_size"}
PRE_M4 = ("baseline", "pymupdf-clean", "docling-clean", "pymupdf-snap", "pymupdf-structure",
          "docling-structure", "pymupdf-semantic")

# Baseline passages (fast to cut) with a small Qwen-like model: built on
# either machine, queried locally.
EITHER = replace(profiles.BASELINE, id="test-either", label="Test either",
                 embedding=replace(profiles.QWEN3_06B, dimensions=16))
# Built and queried on Colab only, like the 8B.
COLAB_ONLY = replace(profiles.BASELINE, id="test-colab", label="Test Colab",
                     embedding=replace(profiles.QWEN3_8B, dimensions=16))


def qwen(**kw) -> FakeOllama:
    return FakeOllama(models=("qwen3-embedding:0.6b", "qwen3-embedding:8b"), dim=16, **kw)


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setitem(profiles.PROFILES, EITHER.id, EITHER)
    monkeypatch.setitem(profiles.PROFILES, COLAB_ONLY.id, COLAB_ONLY)


# --------------------------------------------------------------------------- recorded configuration

@pytest.mark.parametrize("pid", PRE_M4)
def test_profiles_built_before_m4_record_exactly_what_they_did(pid):
    """A new field must never change an existing build's config, or every
    build made before it refuses to open (CLAUDE.md, M3 gotcha)."""
    assert set(profiles.get(pid).config()["embedding"]) == OLD_KEYS


def test_where_a_profile_runs_is_never_part_of_its_config():
    for p in profiles.PROFILES.values():
        assert "build_on" not in p.config()["embedding"] and "query_on" not in p.config()["embedding"]


def test_what_changes_a_vector_is_recorded():
    q = profiles.get("docling-structure-qwen3-0.6b").config()["embedding"]
    assert q["options"] == {"num_ctx": 1024} and q["doc_prefix"] == ""
    assert q["query_prefix"].endswith("\nQuery:")
    g = profiles.get("docling-structure-embeddinggemma2").config()["embedding"]
    assert g["revision"] and g["dtype"] == "float32" and g["backend"] == "sentence-transformers"


def test_row_5_is_both_parsers_times_three_models():
    row5 = [p for p in profiles.PROFILES.values() if p.id not in PRE_M4]
    assert {(p.parser, p.embedding.model) for p in row5} == {
        (parser, model) for parser in ("pymupdf", "docling")
        for model in ("qwen3-embedding:0.6b", "google/embeddinggemma-2", "qwen3-embedding:8b")}
    for p in row5:
        assert len(p.id) <= 40                                   # the API's limit
        assert p.chunker == profiles.PYMUPDF_STRUCTURE.chunker   # one change from row 3


def test_the_8b_is_built_and_queried_on_colab_only():
    spec = profiles.QWEN3_8B
    assert spec.build_on == "colab" and spec.query_on == "colab" and not spec.builds_on("local")


# --------------------------------------------------------------------------- where a build may run

def test_the_8b_refuses_to_build_here():
    with pytest.raises(ingest.BuildRefused, match="Colab GPU"):
        ingest.check_where(profiles.get("docling-structure-qwen3-8b"), "local")


def test_embeddinggemma_refuses_to_build_on_colab():
    with pytest.raises(ingest.BuildRefused, match="runs in this app"):
        ingest.check_where(profiles.get("docling-structure-embeddinggemma2"), "colab")
    with pytest.raises(ProfileMismatch):
        Embedder(profiles.EMBEDDINGGEMMA2, on="colab")


def test_a_build_here_refuses_when_memory_is_short(monkeypatch, tmp_data):
    monkeypatch.setattr(ingest, "available_ram_gb", lambda: 1.0)
    with pytest.raises(ingest.BuildRefused, match=r"Only 1\.0 GB of memory is free"):
        ingest.build(EITHER, client=qwen())
    assert ingest.check_memory(EITHER, "colab") is None          # the GPU's memory is not ours


def test_one_build_at_a_time(tmp_data):
    lock = ingest._BuildLock()
    lock.acquire()
    try:
        with pytest.raises(ingest.BuildRefused, match="Another index is being built"):
            ingest.build(EITHER, client=qwen())
    finally:
        lock.release()


def test_a_build_here_waits_for_an_evaluation_run(tmp_data):
    lock = evaluation._RunLock()
    lock.acquire()
    try:
        with pytest.raises(ingest.BuildRefused, match="evaluation run"):
            ingest.build(EITHER, client=qwen())
    finally:
        lock.release()


def test_an_evaluation_run_waits_for_a_build_here(tmp_data):
    lock = ingest._BuildLock()
    lock.acquire()
    ingest._Job(EITHER, "local", False)
    try:
        assert ingest.active_build()["profile"] == EITHER.id
        with pytest.raises(evaluation.RunBusy, match="being built on this computer"):
            evaluation.execute(evaluation.RunSpec(sets=["set-a"], mode="retrieval"))
    finally:
        lock.release()
    assert ingest.active_build() is None


# --------------------------------------------------------------------------- builds

def test_a_colab_build_records_where_it_ran_and_sends_the_options(tmp_data):
    fake = qwen()
    m = ingest.build(EITHER, client=fake, on="colab").manifest
    assert m["embedding"]["built_on"] == "colab" and m["embedding"]["gpu"] == "Tesla T4"
    assert m["config"] == EITHER.config()
    assert fake.embed_options and all(o == {"num_ctx": 1024} for o in fake.embed_options)
    assert ingest.last_job(EITHER.id)["status"] == "done"


def test_colab_vectors_never_feed_a_local_build(tmp_data):
    ingest.build(EITHER, client=qwen(), on="colab")
    local = ingest.build(EITHER, client=qwen(), on="local").manifest
    assert local["embedding"]["built_on"] == "local"
    assert local["reused"] == 0 and local["from_cache"] == 0 and local["embedded"] == local["rows"]


def test_a_build_cut_off_resumes_from_its_saved_vectors(tmp_data):
    with pytest.raises(embed.RemoteUnavailable):
        ingest.build(EITHER, client=qwen(fail_after=10), on="colab")   # Colab reclaimed the runtime
    job = ingest.last_job(EITHER.id)
    assert job["status"] == "failed" and "reclaimed" in job["error"]
    assert store.active_manifest(EITHER.id) is None                   # nothing half-built is active
    again = ingest.build(EITHER, client=qwen(), on="colab").manifest
    assert again["from_cache"] == 256                                 # saved every 256, kept
    assert again["embedded"] == again["rows"] - 256


def test_a_cancelled_build_keeps_its_vectors(tmp_data):
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 9
    with pytest.raises(ingest.BuildCancelled):
        ingest.build(EITHER, client=qwen(), on="colab", cancel=cancel)
    assert ingest.last_job(EITHER.id)["status"] == "cancelled"
    assert ingest.build(EITHER, client=qwen(), on="colab").manifest["from_cache"] == 256


def test_a_build_whose_process_died_reads_as_interrupted(tmp_data):
    ingest._Job(EITHER, "local", False)          # written, but no lock held: the process is gone
    job = ingest.last_job(EITHER.id)
    assert job["status"] == "interrupted" and "resumes" in job["error"]


# --------------------------------------------------------------------------- asking

def test_a_colab_index_with_no_runtime_refuses_clearly(tmp_data, registered):
    ingest.build(COLAB_ONLY, client=qwen(), on="colab")
    r = retrieve.Retriever(COLAB_ONLY.id)
    assert r.embedder.on == "colab"
    with pytest.raises(RemoteUnavailable, match="no runtime is connected"):
        r.search("What is global growth?", 5)


def test_a_colab_index_asks_through_the_runtime(tmp_data, registered):
    fake = qwen()
    ingest.build(COLAB_ONLY, client=fake, on="colab")
    r = retrieve.Retriever(COLAB_ONLY.id, client=fake)
    hits, _ = r.search("What is global growth?", 5)
    assert len(hits) == 5 and r.index_info()["query_on"] == "colab"


def test_a_runtime_serving_another_build_is_refused(tmp_data, registered):
    ingest.build(COLAB_ONLY, client=qwen(), on="colab")
    r = retrieve.Retriever(COLAB_ONLY.id, client=qwen(digest="sha256:other"))
    with pytest.raises(ProfileMismatch, match="different build"):
        r.search("What is global growth?", 5)


# --------------------------------------------------------------------------- the vector match check

def test_the_match_check_passes_on_float_noise(tmp_data, registered):
    ingest.build(EITHER, client=qwen(), on="local")
    r = embed.match_check(EITHER.id, clients={"local": qwen(), "colab": qwen(noise=0.001)})
    assert r["cross_machine"] and r["passed"] is True
    kinds = {(p["a"], p["b"], p["what"], p["kind"]) for p in r["pairs"]}
    assert kinds == {
        ("local", "local", "passages", "noise"), ("colab", "colab", "passages", "noise"),
        ("local", "stored", "passages", "index"), ("colab", "stored", "passages", "index"),
        ("local", "colab", "passages", "cross"), ("local", "colab", "questions", "cross")}
    sizes = {p["kind"]: p["n"] for p in r["pairs"] if p["what"] == "passages"}
    assert sizes == {"noise": embed.NOISE_PASSAGES, "index": 50, "cross": 50}
    assert r["noise_floor"] is not None
    assert embed.last_match(EITHER.id)["checked_at"] == r["checked_at"]


def test_it_embeds_one_text_per_request(tmp_data, registered):
    """A batched call is path-dependent through Ollama (measured on the T4):
    the check sends one text per request on every machine."""
    ingest.build(EITHER, client=qwen(), on="local")
    here, there = qwen(), qwen()
    embed.match_check(EITHER.id, clients={"local": here, "colab": there})
    assert here.embed_calls == there.embed_calls == 50 + embed.NOISE_PASSAGES + embed.MATCH_QUESTIONS


def test_the_match_check_fails_on_real_drift(tmp_data, registered):
    ingest.build(EITHER, client=qwen(), on="local")
    r = embed.match_check(EITHER.id, clients={"local": qwen(), "colab": qwen(noise=0.5)})
    assert r["passed"] is False
    assert min(p["min"] for p in r["pairs"] if p["kind"] == "cross") < embed.MATCH_THRESHOLD


def test_with_one_machine_there_is_no_verdict(tmp_data, registered):
    ingest.build(COLAB_ONLY, client=qwen(), on="colab")
    r = embed.match_check(COLAB_ONLY.id, clients={"local": FakeOllama(models=(), dim=16), "colab": qwen()})
    assert not r["cross_machine"] and r["passed"] is None
    assert not r["sides"]["local"]["available"]
    assert [(p["a"], p["b"]) for p in r["pairs"]] == [("colab", "colab"), ("colab", "stored")]


def test_the_match_check_needs_a_build(tmp_data, registered):
    with pytest.raises(LookupError, match="not built"):
        embed.match_check(EITHER.id)


# --------------------------------------------------------------------------- the API

def test_the_api_refuses_an_8b_build_here(client):
    r = client.post("/api/indexes/docling-structure-qwen3-8b/build", json={"on": "local"})
    assert r.status_code == 409 and "Colab" in r.json()["detail"]


def test_the_indexes_page_says_what_can_run_and_why(client, tmp_data):
    data = client.get("/api/indexes").json()
    row = {p["id"]: p for p in data["profiles"]}["docling-structure-qwen3-8b"]
    assert row["can"]["build_local"]["ok"] is False and "Colab GPU" in row["can"]["build_local"]["reason"]
    assert row["can"]["query"]["ok"] is False and "not connected" in row["can"]["query"]["reason"]
    assert data["remote"]["connected"] is False
    assert all(isinstance(p["builds"], int) for p in data["profiles"])   # the pickers read a count


def test_no_llm_route_ever_returns_the_token(client):
    from rag_studio import remote
    secret = "ab" * 24
    remote.save_state({"url": "https://x.trycloudflare.com", "token": secret, "models": ["qwen3-embedding:8b"]})
    remote.remember_token(secret)
    for r in (client.get("/api/llm/status"), client.get("/api/llm/colab"), client.get("/api/indexes"),
              client.get("/api/status")):
        assert r.status_code == 200 and secret not in r.text
    r = client.post("/api/llm/disconnect", json={})
    assert r.status_code == 200 and secret not in r.text and not remote.connected()


# --------------------------------------------------------------------------- real models

@pytest.mark.st
def test_embeddinggemma_prefix_equals_its_own_query_prompt():
    e = Embedder(profiles.EMBEDDINGGEMMA2)
    q = np.array(e.query("What is the global growth forecast for 2026?"))
    st = embed.STModel.get(profiles.EMBEDDINGGEMMA2).model
    ref = st.encode(["What is the global growth forecast for 2026?"], prompt_name="query",
                    normalize_embeddings=True)[0]
    assert float(np.dot(q, ref)) > 0.99999 and len(q) == 768


@pytest.mark.colab
@pytest.mark.ollama
def test_local_and_colab_vectors_match():
    """Spec §2.2: the same model and tag give the same vectors on this CPU and
    on the Colab GPU (cosine >= 0.999), so a Colab-built index can be queried
    here. One text per request, as the match check does: a batch adds the
    server's own path noise (docs/02-remote-gpu.md)."""
    spec = profiles.QWEN3_06B
    texts = [f"Passage {i}: global growth is projected at {2 + i / 10:.1f} percent in 2026." for i in range(20)]
    here = np.array(embed._one_by_one(Embedder(spec, on="local"), texts))
    there = np.array(embed._one_by_one(Embedder(spec, on="colab"), texts))
    cos = (here * there).sum(1)
    assert cos.min() >= embed.MATCH_THRESHOLD, cos.min()
