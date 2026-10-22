"""The reranker (M5): ordering, the depth contract, NaN, the memory guard, and
that a plain search records exactly what it recorded before."""
from __future__ import annotations

import pytest

from rag_studio import eval as ev
from rag_studio import ingest, paths, profiles, rerank, retrieve
from rag_studio.retrieve import SearchSpec
from tests.fakes import FakeOllama

IMAC = "apple_imac_per_2024_10"


class FakeScorer:
    """Scores by a rule, and remembers what it was asked."""

    def __init__(self, score=lambda q, d: 0.0):
        self.spec = rerank.RERANKERS["qwen3-0.6b"]
        self.rule = score
        self.asked: list[tuple[str, list[str]]] = []

    def score(self, query, documents):
        self.asked.append((query, list(documents)))
        return [self.rule(query, d) for d in documents]


@pytest.fixture
def index(tmp_data, monkeypatch):
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    fake = FakeOllama()
    ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    return fake, retrieve.Retriever(profiles.PYMUPDF_STRUCTURE.id, client=fake)


def found(r, spec, scorer, k=5, deep=0):
    out = None
    for kind, payload in retrieve.run("What is the carbon footprint of the iMac?", retriever=r, k=k,
                                      spec=spec, deep=deep, scorer=scorer):
        if kind == "found":
            out = payload
    return out


def test_the_reranker_reorders_the_top_20_by_score(index):
    _, r = index
    base = found(r, SearchSpec(), None)
    # The scorer prefers whatever the search ranked last among the 20.
    scorer = FakeScorer(lambda q, d: -len(d))
    spec = SearchSpec(rerank="qwen3-0.6b")
    out = found(r, spec, scorer, deep=50)
    assert len(scorer.asked[0][1]) == 20
    by_len = sorted(range(20), key=lambda i: (len(scorer.asked[0][1][i]), i))
    plain, _ = r.search("What is the carbon footprint of the iMac?", 50)
    assert [h["chunk_id"] for h in out.hits[:20]] == [plain[i]["chunk_id"] for i in by_len]
    # Everything below the reranked 20 keeps its order and its place.
    assert [h["chunk_id"] for h in out.hits[20:50]] == [h["chunk_id"] for h in plain[20:50]]
    assert [h["rank"] for h in out.hits] == list(range(1, len(out.hits) + 1))
    assert out.hits[0]["ranks"]["reranked"] == 1 and out.hits[0]["rank_before"] == plain[by_len[0]]["rank"]
    assert out.hits[25]["ranks"]["reranked"] is None
    assert base.hits[0]["chunk_id"] == plain[0]["chunk_id"]


def test_the_top_k_of_a_reranked_search_is_the_first_k_of_a_deeper_one(index):
    _, r = index
    spec = SearchSpec(rerank="qwen3-0.6b")
    scorer = FakeScorer(lambda q, d: hash(d) % 1000)
    top = found(r, spec, scorer, k=5)
    deep = found(r, spec, scorer, k=5, deep=50)
    assert [h["chunk_id"] for h in top.hits] == [h["chunk_id"] for h in deep.hits[:5]]


def test_a_shallower_depth_reads_fewer_passages(index):
    _, r = index
    scorer = FakeScorer()
    found(r, SearchSpec(rerank="qwen3-0.6b", rerank_depth=8), scorer)
    assert len(scorer.asked[0][1]) == 8


def test_the_rerank_step_says_what_moved(index):
    _, r = index
    out = found(r, SearchSpec(rerank="qwen3-0.6b"), FakeScorer(lambda q, d: -len(d)))
    step = next(s for s in out.steps if s.id == "rerank")
    assert step.detail["depth"] == 20 and step.detail["reranker"] == "qwen3-0.6b"
    assert {"chunk_id", "from", "to"} <= set(step.detail["moves"][0])
    assert [s.id for s in out.steps] == ["embed", "search", "rerank"]


def test_the_rerank_reads_what_was_embedded_not_what_the_ui_shows(index):
    _, r = index
    scorer = FakeScorer()
    found(r, SearchSpec(rerank="qwen3-0.6b"), scorer)
    assert scorer.asked[0][1][0].startswith("Document: ")      # the label line is part of search_text


def test_nan_scores_are_an_error_not_an_order():
    with pytest.raises(rerank.RerankError, match="NaN or infinite"):
        rerank.check_scores([1.0, float("nan")])
    with pytest.raises(rerank.RerankError):
        rerank.check_scores([float("inf")])
    rerank.check_scores([1.0, -2.5])


def test_loading_refuses_when_memory_is_short(monkeypatch):
    monkeypatch.setattr(ingest, "available_ram_gb", lambda: 1.0)
    monkeypatch.setattr(rerank.LocalReranker, "_loaded", {})
    with pytest.raises(rerank.RerankError, match="only 1.0 GB is free"):
        rerank.LocalReranker.get(rerank.RERANKERS["qwen3-0.6b"])


def test_a_remote_reranker_is_not_loaded_in_this_app():
    with pytest.raises(rerank.RerankError, match="Remote AI"):
        rerank.LocalReranker.get(rerank.RERANKERS["qwen3-4b"])


def test_a_plain_search_describes_itself_exactly_as_before():
    assert SearchSpec().describe() == {"method": "dense", "metric": "cosine", "exact": True}
    assert SearchSpec("hybrid", 2).describe() == {"method": "hybrid", "metric": "cosine", "exact": True,
                                                  "fusion": "rrf", "rrf_k": 60, "depth": 50, "section_cap": 2}
    assert SearchSpec().as_args() == {"method": "dense", "section_cap": None}


def test_a_reranked_search_adds_only_its_own_keys():
    d = SearchSpec(rerank="qwen3-0.6b").describe()
    assert d["rerank"] == "qwen3-0.6b" and d["rerank_depth"] == 20 and "order" in d
    assert SearchSpec(rerank="qwen3-0.6b").label == "Dense, reranked"
    assert SearchSpec(**SearchSpec(rerank="qwen3-0.6b", rerank_depth=10).as_args()).rerank_depth == 10


def test_bad_switches_are_refused_with_the_known_values():
    for kw in ({"rerank": "gpt"}, {"rerank_depth": 1}, {"rewrite": "magic"}, {"agent_rounds": 1}):
        with pytest.raises(ValueError):
            SearchSpec(**kw)


def test_a_run_records_the_reranker_and_where_the_answer_page_moved(index, monkeypatch):
    fake, r = index
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    monkeypatch.setattr(rerank, "scorer_for", lambda spec, cache=True: FakeScorer(lambda q, d: -len(d)))
    monkeypatch.setattr(rerank, "runtime_facts", lambda rid: {"reranker": {"where": "local"}})
    spec = ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id,
                      search=SearchSpec(rerank="qwen3-0.6b").as_args())
    run = ev.execute(spec, client=fake)
    cfg = run["config"]["retrieval"]
    assert cfg["rerank"]["model"] == "Qwen/Qwen3-Reranker-0.6B" and cfg["rerank"]["depth"] == 20
    assert len(cfg["rerank"]["revision"]) == 40 and len(cfg["rerank"]["instruction_sha256"]) == 64
    assert "rerank_depth" not in cfg and run["config"]["runtime"]["reranker"]["where"] == "local"
    row = run["results"][0]
    assert {"before_rank", "after_rank", "cache", "compute_ms"} == set(row["rerank"]) and "rerank_ms" in row["timings"]


@pytest.mark.rerank
def test_the_cross_encoder_matches_the_cards_manual_formula():
    """Three pairs, scored by sentence-transformers and by the model card's own
    code (research B §3): they must agree to float precision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = rerank.RERANKERS["qwen3-0.6b"]
    folder = rerank.snapshot(spec)
    q = "How many depository libraries did GPO staff visit?"
    docs = ["GPO staff visited 166 libraries in FY2019 to review their operations.",
            "The iMac stand is made of aluminium and recycled materials.",
            "Global growth is forecast at 2.4 percent in 2026, down from 2.7 percent in 2025."]
    tok = AutoTokenizer.from_pretrained(folder, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    assert (yes, no) == (9693, 2152)
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
              "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instr = rerank.INSTRUCTIONS[rerank.INSTRUCTION_VERSION]
    texts = [prefix + f"<Instruct>: {instr}\n<Query>: {q}\n<Document>: {d}" + suffix for d in docs]
    with torch.inference_mode():
        logits = lm(**tok(texts, return_tensors="pt", padding=True)).logits[:, -1, :]
    manual = (logits[:, yes] - logits[:, no]).tolist()
    del lm
    got = rerank.LocalReranker.get(spec).score(q, docs)
    rerank.LocalReranker.release()
    assert max(abs(a - b) for a, b in zip(manual, got)) < 1e-3
    assert got[0] > got[2] > got[1] or got[0] > got[1]


# ------------------------------------------------------------------ the score cache

class Counting:
    spec = rerank.RERANKERS["qwen3-4b"]

    def __init__(self):
        self.calls = 0
        self.last = {"compute_ms": 321.0}
        self.last_net = {"calls": 1}

    def score(self, query, documents):
        self.calls += 1
        return [float(len(d)) for d in documents]


def test_a_score_is_computed_once_and_then_replayed(tmp_data):
    inner = Counting()
    scorer = rerank.CachedScorer(inner)
    assert scorer.score("q", ["aa", "b"]) == [2.0, 1.0] and scorer.last_cache == "miss"
    assert scorer.score("q", ["aa", "b"]) == [2.0, 1.0] and scorer.last_cache == "hit"
    assert inner.calls == 1 and scorer.last["compute_ms"] == 321.0 and scorer.last["from_cache"] is True
    scorer.score("q", ["b", "aa"])                    # another order is another question
    scorer.score("other", ["aa", "b"])
    assert inner.calls == 3


def test_the_cache_is_per_reranker_setup(tmp_data):
    four, eight = Counting(), Counting()
    eight.spec = rerank.RERANKERS["qwen3-8b"]
    rerank.CachedScorer(four).score("q", ["a"])
    rerank.CachedScorer(eight).score("q", ["a"])
    assert four.calls == eight.calls == 1


def test_a_damaged_cache_entry_is_recomputed_not_trusted(tmp_data):
    inner = Counting()
    scorer = rerank.CachedScorer(inner)
    scorer.score("q", ["a", "b"])
    path = scorer._path("q", ["a", "b"])
    path.write_text('{"scores": [1.0, NaN]}')
    scorer.score("q", ["a", "b"])
    assert inner.calls == 2
    path.write_text("not json")
    assert scorer.score("q", ["a", "b"]) == [1.0, 1.0] and inner.calls == 3


def test_a_replayed_run_never_builds_the_scorer(tmp_data):
    built = []
    lazy = rerank._Lazy(rerank.RERANKERS["qwen3-4b"], lambda: built.append(1) or Counting())
    scorer = rerank.CachedScorer(lazy)
    scorer.score("q", ["a"])
    assert built == [1]
    again = rerank.CachedScorer(rerank._Lazy(rerank.RERANKERS["qwen3-4b"], lambda: built.append(2) or Counting()))
    again.score("q", ["a"])
    assert built == [1]                              # a hit needs no model and no runtime


def test_a_run_says_where_each_score_came_from(index, monkeypatch):
    fake, r = index
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    inner = Counting()
    inner.spec = rerank.RERANKERS["qwen3-0.6b"]
    monkeypatch.setattr(rerank, "scorer_for", lambda spec, cache=True: rerank.CachedScorer(inner) if cache else inner)
    monkeypatch.setattr(rerank, "runtime_facts", lambda rid: {})
    spec = ev.RunSpec(sets=["set-b"], mode="retrieval", profile=r.profile.id,
                      search=SearchSpec(rerank="qwen3-0.6b").as_args())
    first = ev.execute(spec, client=fake)
    second = ev.execute(spec, client=fake)
    assert {x["rerank"]["cache"] for x in first["results"]} == {"miss"}
    assert {x["rerank"]["cache"] for x in second["results"]} == {"hit"}
    assert [x["best_rank"] for x in first["results"]] == [x["best_rank"] for x in second["results"]]
    assert second["results"][0]["cache"] == {"rerank": "hit"}
