"""Index builds: atomic switch, incremental reuse, and refusing a mismatched model.

Runs the real ingest on the real PDFs, with a fake embedding model, into a
temporary data folder.
"""
from __future__ import annotations

import pytest

from rag_studio import ingest, profiles, retrieve, store
from rag_studio.embed import ProfileMismatch
from tests.fakes import FakeOllama, fake_vector

PROFILE = profiles.BASELINE


@pytest.fixture
def built(tmp_data):
    fake = FakeOllama()
    result = ingest.build(PROFILE, client=fake)
    return fake, result


def test_build_records_its_configuration(built):
    _, result = built
    m = result.manifest
    assert not result.up_to_date and m["rows"] == 1190
    assert m["embedding"]["digest"] == "sha256:aaaa"
    assert m["embedding"]["doc_prefix"] == "search_document: "
    assert m["config"] == PROFILE.config()
    assert m["config"]["metric"] == "cosine"
    assert {d["doc_id"]: d["passages"] for d in m["documents"]} == {
        "apple_imac_per_2024_10": 34, "gpo_2018_annual_report": 233, "wb_gep_2026_06": 923}
    assert store.active_manifest(PROFILE.id)["table"] == m["table"]


def test_unchanged_documents_make_no_new_build(built):
    fake, first = built
    calls = fake.embed_calls
    again = ingest.build(PROFILE, client=fake)
    assert again.up_to_date and again.manifest["table"] == first.manifest["table"]
    assert fake.embed_calls == calls


def test_forced_rebuild_reuses_every_vector_and_switches(built):
    fake, first = built
    calls = fake.embed_calls
    second = ingest.build(PROFILE, client=fake, force=True)
    assert second.manifest["table"] != first.manifest["table"]
    assert second.manifest["reused"] == 1190 and second.manifest["embedded"] == 0
    assert fake.embed_calls == calls
    assert store.active_manifest(PROFILE.id)["table"] == second.manifest["table"]
    # The previous build is kept for rollback.
    assert {b["table"] for b in store.builds(PROFILE.id)} == {first.manifest["table"],
                                                             second.manifest["table"]}


def test_exact_cosine_search_finds_a_passage_by_its_own_vector(built):
    r = retrieve.Retriever(PROFILE.id, client=built[0])
    text = r.table.to_arrow().slice(5, 1).to_pylist()[0]
    hits = store.search_dense(r.table, fake_vector(text["text"]), 3)
    assert hits[0]["chunk_id"] == text["chunk_id"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert [h["rank"] for h in hits] == [1, 2, 3]


def test_a_different_model_build_is_refused_loudly(built):
    other = FakeOllama(digest="sha256:bbbb")
    r = retrieve.Retriever(PROFILE.id, client=other)
    with pytest.raises(ProfileMismatch, match="different build"):
        r.dense("How many depository libraries?", 5)


def test_a_missing_model_is_refused(built):
    none = FakeOllama(models=("qwen3:1.7b",))
    r = retrieve.Retriever(PROFILE.id, client=none)
    with pytest.raises(ProfileMismatch, match="not installed"):
        r.dense("anything", 5)


def test_a_changed_profile_setup_is_refused(built, monkeypatch):
    manifest = store.active_manifest(PROFILE.id)
    manifest["config"]["chunker"]["size"] = 800
    store.activate(manifest)
    with pytest.raises(ProfileMismatch, match="different setup"):
        retrieve.Retriever(PROFILE.id, client=built[0])


def test_no_index_is_reported_plainly(tmp_data):
    with pytest.raises(retrieve.IndexMissing, match="./rag ingest"):
        retrieve.retriever(PROFILE.id)
