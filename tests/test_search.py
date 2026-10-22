"""Search (M3): BM25 over the word index, hybrid fusion (checked against
LanceDB's own RRF), the section cap, and runs that record how they searched."""
from __future__ import annotations

import pyarrow as pa
import pytest
from lancedb.rerankers import RRFReranker

from rag_studio import eval as ev
from rag_studio import ingest, paths, profiles, retrieve, store
from rag_studio.retrieve import SearchSpec
from tests.fakes import FakeOllama

IMAC = "apple_imac_per_2024_10"


@pytest.fixture
def imac_only(tmp_data, monkeypatch):
    """Only the 12-page iMac report, so a cleaned, structure-aware build takes seconds."""
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    return tmp_data


@pytest.fixture
def structure(imac_only):
    fake = FakeOllama()
    result = ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    return fake, result, retrieve.Retriever(profiles.PYMUPDF_STRUCTURE.id, client=fake)


def test_a_build_gets_a_word_index_over_what_was_embedded(structure):
    _, result, r = structure
    m = result.manifest
    assert m["word_index"]["column"] == "search_text"
    assert m["word_index"]["settings"]["remove_stop_words"] is False
    row = r.table.to_arrow().slice(0, 1).to_pylist()[0]
    assert row["search_text"] == row["label"] + "\n" + row["text"]
    assert row["label"].startswith("Document: iMac Product Environmental Report\nSection: ")


def test_bm25_finds_the_words(structure):
    _, _, r = structure
    hits, t = r.bm25("ENERGY STAR requirement", 5)
    assert hits and "ENERGY STAR" in hits[0]["text"] and hits[0]["page"] == 7
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    assert t["bm25_ms"] >= 0


def test_a_build_without_a_word_index_gets_one_in_place(structure):
    fake, result, r = structure
    r.table.drop_index(store.fts_index(r.table)["name"])
    manifest = {k: v for k, v in result.manifest.items() if k != "word_index"}
    store.activate(manifest)
    with pytest.raises(store.WordIndexMissing, match="./rag ingest"):
        store.search_bm25(r.table, "iMac", 5)
    again = ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    assert again.up_to_date and again.manifest["table"] == result.manifest["table"]
    assert again.manifest["word_index"]["column"] == "search_text"
    assert store.search_bm25(store.open_table(again.manifest["table"]), "iMac", 5)


def _hits(ids: list[str]) -> list[dict]:
    return [{"chunk_id": c, "rank": i, "doc_id": "d", "score": 0.0} for i, c in enumerate(ids, start=1)]


def test_rrf_matches_lancedbs_own_reranker():
    dense = _hits(["a", "b", "c", "d", "e", "f"])
    bm25 = _hits(["e", "a", "x", "b", "y"])
    ours = retrieve.rrf({"dense": dense, "bm25": bm25})
    rowid = {c: i for i, c in enumerate("abcdefxy")}

    def table(hits):
        return pa.table({"_rowid": pa.array([rowid[h["chunk_id"]] for h in hits], pa.uint64()),
                         "chunk_id": [h["chunk_id"] for h in hits]})

    theirs = RRFReranker(K=60).rerank_hybrid("q", table(dense), table(bm25)).to_pylist()
    want = {r["chunk_id"]: r["_relevance_score"] for r in theirs}
    assert {h["chunk_id"] for h in ours} == set(want)
    for h in ours:
        assert h["score"] == pytest.approx(want[h["chunk_id"]], rel=1e-6)
    assert [h["score"] for h in ours] == sorted((h["score"] for h in ours), reverse=True)
    assert ours[0]["chunk_id"] == "a"                     # ranks 1 and 2 beat rank 1 alone


def test_rrf_breaks_ties_by_the_dense_rank():
    ours = retrieve.rrf({"dense": _hits(["a", "b"]), "bm25": _hits(["b", "a"])})
    assert [h["chunk_id"] for h in ours] == ["a", "b"]


def _sec(cid: str, section: str | None, kind: str = "body", doc: str = "d") -> dict:
    return {"chunk_id": cid, "doc_id": doc, "section_path": section, "kind": kind}


def test_the_section_cap_moves_passages_behind_and_drops_none():
    hits = [_sec("r1", "Ch 3 > References", "references"), _sec("r2", "Ch 3 > References", "references"),
            _sec("s1", "Ch 3"), _sec("s2", "Ch 3"), _sec("s3", "Ch 3"), _sec("o1", "Ch 1"),
            _sec("n1", None), _sec("n2", None), _sec("n3", None)]
    out = retrieve.cap_sections(hits, 2)
    assert [h["chunk_id"] for h in out] == ["r1", "s1", "s2", "o1", "n1", "n2", "n3", "r2", "s3"]
    assert [h.get("capped") for h in out[-2:]] == [True, True]


def test_a_deeper_search_starts_with_the_same_passages(structure):
    _, _, r = structure
    for spec in (SearchSpec(), SearchSpec("bm25"), SearchSpec("hybrid"), SearchSpec("hybrid", 1)):
        top, _ = r.search("What is the carbon footprint of the iMac?", 5, spec)
        deep, _ = r.search("What is the carbon footprint of the iMac?", 50, spec)
        assert [h["chunk_id"] for h in top] == [h["chunk_id"] for h in deep[:5]], spec


def test_hybrid_hits_say_where_they_ranked_in_each_list(structure):
    _, _, r = structure
    hits, info = r.search("recycled aluminum stand", 5, SearchSpec("hybrid"))
    assert set(info["lists"]) == {"dense", "bm25"}
    for h in hits:
        assert set(h["ranks"]) == {"dense", "bm25"}
        assert any(v is not None for v in h["ranks"].values())


def test_a_dense_run_records_what_m1_runs_recorded(structure, monkeypatch):
    fake, _, r = structure
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    run = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id), client=fake)
    assert run["config"]["retrieval"] == {"method": "dense", "metric": "cosine", "exact": True}


def test_a_hybrid_run_records_its_fusion_and_is_its_own_ladder_variant(structure, monkeypatch):
    fake, _, r = structure
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    dense = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id), client=fake)
    hybrid = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id,
                                   search={"method": "hybrid", "section_cap": 2}), client=fake)
    cfg = hybrid["config"]["retrieval"]
    assert cfg["method"] == "hybrid" and cfg["rrf_k"] == 60 and cfg["depth"] == 50
    assert cfg["section_cap"] == 2 and cfg["word_index"]["column"] == "search_text"
    assert ev.variant_key(dense["config"]) != ev.variant_key(hybrid["config"])
    assert all("ranks" in h for res in hybrid["results"] for h in res["hits"])


def test_bm25_on_an_index_without_a_word_index_is_refused(structure, monkeypatch):
    fake, _, r = structure
    r.table.drop_index(store.fts_index(r.table)["name"])
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    with pytest.raises(RuntimeError, match="no word index"):
        ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id,
                              search={"method": "bm25", "section_cap": None}), client=fake)


def test_unknown_search_settings_are_refused():
    with pytest.raises(ValueError, match="Unknown search method"):
        SearchSpec("sparse")
    with pytest.raises(ValueError, match="at least 1"):
        SearchSpec("dense", 0)


def test_a_changed_label_with_the_same_counts_is_a_new_build(structure, monkeypatch):
    """The box-end fix changed section labels without changing a passage count;
    the build must notice (it once said "up to date")."""
    from rag_studio import chunk

    fake, first, _ = structure
    calls = fake.embed_calls
    assert ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake).up_to_date
    assert fake.embed_calls == calls
    real = chunk.section_label
    monkeypatch.setattr(chunk, "section_label", lambda t, p, k="body": real(t, [*(p or []), "Renamed"], k))
    again = ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    assert not again.up_to_date and again.manifest["embedded"] == again.manifest["rows"]
    assert again.manifest["rows_sha256"] != first.manifest["rows_sha256"]


# ------------------------------------------------------------------ the document filter (M5, the agent's one tool)

def test_a_document_filter_is_applied_before_the_k_best_are_taken(structure):
    _, _, r = structure
    known = r.doc_ids()
    where = store.doc_filter(known[0], known)
    for spec in (SearchSpec(), SearchSpec("bm25")):
        hits, _ = r.search("What is the carbon footprint?", 5, spec, where=where)
        assert len(hits) == 5 and {h["doc_id"] for h in hits} == {known[0]}
    # a filter that matches nothing returns nothing: a prefilter, not "k best, then drop"
    none = "doc_id = 'no_such_document'"
    assert r.search("What is the carbon footprint?", 5, SearchSpec(), where=none)[0] == []
    assert r.search("carbon footprint", 5, SearchSpec("bm25"), where=none)[0] == []
    assert r.search("What is the carbon footprint?", 5, SearchSpec(), where=None)[0]      # default unchanged


@pytest.mark.parametrize("bad", ["x' OR '1'='1", "gpo_2018_annual_report; DROP", "", "doc_id = 'x'", "' --"])
def test_free_text_never_reaches_a_where_string(bad):
    with pytest.raises(ValueError, match="is not a document in this index"):
        store.doc_filter(bad, ["gpo_2018_annual_report", "apple_imac_per_2024_10"])
    assert store.doc_filter("apple_imac_per_2024_10", ["apple_imac_per_2024_10"]) == "doc_id = 'apple_imac_per_2024_10'"
