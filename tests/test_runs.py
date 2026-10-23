"""Evaluation runs: recorded configuration, one run at a time, and the API."""
from __future__ import annotations

import pytest

from rag_studio import eval as ev
from rag_studio import ingest, profiles, retrieve
from tests.fakes import FakeOllama


@pytest.fixture
def fake(tmp_data, monkeypatch):
    f = FakeOllama(answer="GPO staff visited 166 depository libraries [1].")
    ingest.build(profiles.BASELINE, client=f)
    r = retrieve.Retriever(profiles.BASELINE.id, client=f)
    monkeypatch.setattr(ev, "active_retriever", lambda profile_id: r)
    return f


def test_retrieval_run_records_config_and_recall(fake):
    run = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    assert run["status"] == "done" and run["progress"] == {"done": 6, "total": 6}
    assert run["config"]["index"]["embedding"]["digest"] == "sha256:aaaa"
    assert run["config"]["answer"] is None
    assert run["summary"]["recall"]["set-b"]["total"] == 6
    assert run["summary"]["outcomes"] is None
    assert all(len(r["hits"]) == 5 for r in run["results"])
    assert ev.get_run(run["id"])["status"] == "done"
    assert ev.list_runs()[0]["id"] == run["id"]


def test_answer_run_scores_every_scored_question(fake):
    run = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b", profile="baseline", prompt_version="v2"), client=fake)
    o = run["summary"]["outcomes"]
    assert sum(o.values()) == 11
    a1 = next(r for r in run["results"] if r["id"] == "A1")
    assert a1["outcome"] == "PASS" and a1["found"] == ["166"]
    t1 = next(r for r in run["results"] if r["id"] == "T1")
    assert t1["outcome"] == "WRONG"            # the fake answers a trap: dangerous
    cfg = run["config"]["answer"]
    assert cfg["model"] == "qwen3:1.7b" and cfg["think"] is False and cfg["prompt_version"] == "v2"
    assert len(cfg["prompt_sha256"]) == 64


def test_cancel_stops_after_the_current_question(fake):
    seen = []
    run = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake,
                     cancel=lambda: len(seen) >= 2, on_result=lambda run, res: seen.append(res))
    assert run["status"] == "cancelled" and run["progress"]["done"] == 2


def test_only_one_run_at_a_time(fake):
    held = ev._RunLock()
    held.acquire()
    try:
        assert ev.run_is_active()
        with pytest.raises(ev.RunBusy):
            ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    finally:
        held.release()
    assert not ev.run_is_active()


def test_a_run_left_running_is_reported_interrupted(fake):
    run = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    path = ev.run_path(run["id"])
    path.write_text(path.read_text().replace('"status": "done"', '"status": "running"'))
    assert ev.get_run(run["id"])["status"] == "interrupted"


def test_unknown_set_is_refused(fake):
    with pytest.raises(ValueError, match="Unknown"):
        ev.execute(ev.RunSpec(sets=["set-z"], mode="retrieval"), client=fake)


def test_api_refuses_a_second_run(client, fake):
    held = ev._RunLock()
    held.acquire()
    try:
        r = client.post("/api/eval/runs", json={"sets": ["set-b"], "mode": "retrieval"})
        assert r.status_code == 409
        assert "already in progress" in r.json()["detail"]
    finally:
        held.release()


def test_api_lists_sets_and_unknown_runs_404(client, fake):
    sets = client.get("/api/golden").json()["sets"]
    assert {s["id"] for s in sets} >= {"set-a", "set-b", "set-d"}
    assert client.get("/api/eval/runs/20990101-000000-abcd").status_code == 404
    assert client.get("/api/eval/runs/..%2Fsecrets").status_code == 404


def test_compare_marks_the_rows_that_changed(fake):
    before = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b", profile="baseline", prompt_version="v2"), client=fake)
    fake.answer_text = "I could not find this in the documents."
    after = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b",
                                  profile="baseline", prompt_version="v3"), client=fake)
    cmp = ev.compare(before["id"], after["id"])
    keys = {c["key"] for c in cmp["config"]}
    assert "answer.prompt_version" in keys and "answer.prompt_sha256" in keys
    assert not any(k.startswith("index.built_at") for k in keys)       # noise is left out
    a1 = next(r for r in cmp["rows"] if r["id"] == "A1")
    assert (a1["a"]["outcome"], a1["b"]["outcome"]) == ("PASS", "REFUSED") and a1["outcome_changed"]
    t1 = next(r for r in cmp["rows"] if r["id"] == "T1")
    assert (t1["a"]["outcome"], t1["b"]["outcome"]) == ("WRONG", "PASS")
    assert cmp["changed"] == sum(1 for r in cmp["rows"] if r["outcome_changed"] or r["retrieval_changed"])


def test_ladder_groups_tagged_runs_into_variants_by_setup(fake):
    a = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval",
                              ladder={"row": 2, "label": "Variant one"}), client=fake)
    b = ev.execute(ev.RunSpec(sets=["set-a"], mode="retrieval",
                              ladder={"row": 2, "label": "Variant one: more sets"}), client=fake)
    row2 = next(r for r in ev.ladder() if r["row"] == 2)
    assert len(row2["variants"]) == 1                      # same setup: one variant, both sets
    v = row2["variants"][0]
    assert v["label"] == "Variant one" and set(v["sets"]) == {"set-a", "set-b"} and not v["mixed"]
    assert {x["id"] for x in v["runs"]} == {a["id"], b["id"]}
    assert all(r["variants"] == [] for r in ev.ladder() if r["row"] not in (2,))


def test_api_compare(client, fake):
    one = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    two = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    r = client.get(f"/api/eval/compare?a={one['id']}&b={two['id']}")
    assert r.status_code == 200 and r.json()["changed"] == 0 and len(r.json()["rows"]) == 6
    assert client.get("/api/eval/compare?a=nope&b=nope").status_code == 404


def test_api_refuses_an_unknown_profile_or_prompt(client, fake):
    r = client.post("/api/eval/runs", json={"sets": ["set-b"], "mode": "retrieval", "profile": "nope"})
    assert r.status_code == 400 and "index profile" in r.json()["detail"]
    r = client.post("/api/eval/runs", json={"sets": ["set-b"], "mode": "retrieval", "prompt": "v9"})
    assert r.status_code == 400 and "prompt version" in r.json()["detail"]


# ------------------------------------------------------------------ M5: keys, ranking, two passes

def _cfg(retrieval, table="t1", k=5):
    return {"index": {"table": table, "embedding": {"digest": "d"}}, "k": k,
            "answer": {"model": "m", "digest": "x", "prompt_sha256": "p"}, "retrieval": retrieval}


def test_the_new_config_key_groups_runs_exactly_as_the_old_one_did():
    from rag_studio.retrieve import SearchSpec
    cases = [SearchSpec().describe(), SearchSpec("bm25").describe(), SearchSpec("hybrid").describe(),
             SearchSpec("dense", 2).describe(), SearchSpec("hybrid", 2).describe(),
             SearchSpec("hybrid", 3).describe()]
    cases += [d | {"word_index": {"built": n}} for d in cases[1:3] for n in (1, 2)]     # a rebuilt word index
    cfgs = [_cfg(c, table=t) for c in cases for t in ("t1", "t2")]
    for a in cfgs:
        for b in cfgs:
            assert (ev._legacy_key(a) == ev._legacy_key(b)) == (ev.config_key(a) == ev.config_key(b))


def test_every_real_run_on_disk_keeps_its_group():
    """Runs made before M5 (when the data folder has them): the new key
    separates exactly the runs the old key separated. A run with a reranker,
    rewriting or an agent is what the old key could not see, so it is not compared."""
    from rag_studio import paths
    files = sorted(paths.RUNS.glob("*.json")) if paths.RUNS.is_dir() else []
    if len(files) < 5:
        pytest.skip("No earlier runs to check against")
    from rag_studio.files import read_json
    cfgs = [c for c in (read_json(f).get("config") for f in files) if c
            and not {"rerank", "rewrite", "agent", "agent_rounds"} & set(c.get("retrieval") or {})]   # made before M5
    # (A multi-step run records its controller under "agent": two of them shared the old key with a
    # plain 5d run, a difference the old key never could see, and the check failed on real data, 2026-10-23.)
    old = {ev._legacy_key(c) for c in cfgs}
    new = {ev.config_key(c) for c in cfgs}
    assert len(old) == len(new)
    assert all(len({ev.config_key(c) for c in cfgs if ev._legacy_key(c) == k}) == 1 for k in old)


def test_a_switch_that_is_on_makes_its_own_variant():
    from rag_studio.retrieve import SearchSpec
    plain = _cfg(SearchSpec().describe())
    reranked = _cfg(SearchSpec(rerank="qwen3-0.6b").describe())
    other = _cfg(SearchSpec(rerank="qwen3-4b").describe())
    assert len({ev.variant_key(c) for c in (plain, reranked, other)}) == 3


def test_mrr_and_recall_at_20_come_from_the_deep_rank():
    def row(best, targets=True):
        return {"targets": [{"doc_id": "d", "pages": [1]}] if targets else [], "best_rank": best}
    s = ev.ranking_stats([row(1), row(4), row(None), row(25), row(0 or None, targets=False)])
    assert s["n"] == 4 and s["recall20"] == 2 and s["mrr"] == round((1 + .25 + 0 + 1 / 25) / 4, 4)
    assert ev.ranking_stats([row(None, targets=False)])["mrr"] is None        # "-", never 0
    two_docs = {"targets": [{"doc_id": "a", "pages": [1]}, {"doc_id": "b", "pages": [1]}], "best_rank": 3}
    out = ev.ranking_stats([row(1), two_docs])
    assert out["single"]["n"] == 1 and out["multi"]["n"] == 1 and out["multi"]["mrr"] == round(1 / 3, 4)


def test_an_old_run_gets_mrr_without_being_rewritten(fake):
    run = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    path = ev.run_path(run["id"])
    from rag_studio.files import read_json, write_json_atomic
    saved = read_json(path)
    del saved["summary"]["ranking"]                  # as a run saved before M5
    write_json_atomic(path, saved)
    again = ev.get_run(run["id"])
    assert again["summary"]["ranking"]["n"] == 6 and again["summary"]["ranking"]["recall20"] is not None
    assert next(b for b in ev.list_runs() if b["id"] == run["id"])["summary"]["ranking"]["n"] == 6


def test_a_run_searches_once_and_answers_from_that_search(fake, monkeypatch):
    searches = []
    real = retrieve.Retriever.search
    monkeypatch.setattr(retrieve.Retriever, "search",
                        lambda self, q, k, spec=retrieve.DENSE: (searches.append(q), real(self, q, k, spec))[1])
    run = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b", profile="baseline",
                                prompt_version="v2"), client=fake)
    assert len(searches) == len(run["results"]) == 11 and len(set(searches)) == 11
    assert not any(r["error"] for r in run["results"])


def test_a_two_pass_run_releases_the_search_models_before_answering(fake, monkeypatch):
    from rag_studio import rerank
    from rag_studio.retrieve import SearchSpec
    order = []

    class Scorer:
        spec = rerank.RERANKERS["qwen3-0.6b"]

        def score(self, q, docs):
            order.append("search")
            return [-len(d) for d in docs]

    monkeypatch.setattr(rerank, "scorer_for", lambda spec, cache=True: Scorer())
    monkeypatch.setattr(rerank, "runtime_facts", lambda rid: {})
    monkeypatch.setattr(ev, "release_search_models", lambda: order.append("release"))
    real_chat = fake.chat_stream
    fake.chat_stream = lambda *a, **k: (order.append("answer"), real_chat(*a, **k))[1]
    fake.unload = lambda model: (order.append(f"unload {model}"), True)[1]
    spec = ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b", profile="baseline", prompt_version="v2",
                      search=SearchSpec(rerank="qwen3-0.6b").as_args())
    run = ev.execute(spec, client=fake)
    # The previous run's answer model is freed first, so the search models fit beside nothing.
    assert order == ["unload qwen3:1.7b"] + ["search"] * 11 + ["release"] + ["answer"] * 11
    assert run["progress"]["done"] == run["progress"]["total"]


# ------------------------------------------------------------------ Set E runs

def _remote_for_followups(monkeypatch, standalone):
    """A runtime that rewrites follow-ups with `standalone(user_text)`."""
    from tests.test_rewrite import FakeRemote
    import rag_studio.remote as remote_mod
    rt = FakeRemote({"standalone": lambda user: {"standalone": standalone(user)}})
    monkeypatch.setattr(remote_mod, "client", lambda: rt)
    return rt


def test_a_set_e_run_judges_the_rewrite_then_retrieval_then_the_answer(fake, monkeypatch):
    rt = _remote_for_followups(monkeypatch, lambda user: user.split("Last message: ")[1].replace("that", "the iMac"))
    run = ev.execute(ev.RunSpec(sets=["set-e"], mode="retrieval", profile="baseline"), client=fake)
    assert run["status"] == "done" and len(run["results"]) == 8
    row = next(x for x in run["results"] if x["id"] == "E2")
    assert row["standalone_question"] == "How much of the iMac is transportation?" and row["rewrite_ok"] is True
    assert row["history"][0]["question"].startswith("What is the total carbon footprint")
    assert run["config"]["followup"]["prompt_version"] == "standalone-v1" and "followup" not in run["config"]["retrieval"]
    assert run["summary"]["rewrite"]["total"] == 8 and 0 < run["summary"]["rewrite"]["ok"] < 8
    assert run["results"][0]["rewrites"][0]["cache"] == "miss" and len(rt.calls) == 8


def test_asked_as_typed_makes_no_remote_call_and_says_so(fake, monkeypatch):
    rt = _remote_for_followups(monkeypatch, lambda user: "never used")
    run = ev.execute(ev.RunSpec(sets=["set-e"], mode="retrieval", profile="baseline", as_typed=True), client=fake)
    assert rt.calls == [] and run["config"]["followup"] == {"as_typed": True}
    assert all(x.get("standalone_question") is None and x["rewrite_ok"] is None for x in run["results"])
    assert run["summary"]["rewrite"] is None


def test_a_runtime_that_goes_away_fails_the_run_instead_of_finishing_it_full_of_errors(fake, monkeypatch):
    from rag_studio import rewrite
    calls = []

    def flaky(user):
        calls.append(1)
        if len(calls) == 3:
            raise rewrite.RewriteUnavailable("The Colab runtime is not answering. Check Remote AI in Settings.")
        return "A standalone question about the iMac?"
    from tests.test_rewrite import FakeRemote
    import rag_studio.remote as remote_mod
    rt = FakeRemote({"standalone": lambda user: {"standalone": flaky(user)}})
    monkeypatch.setattr(remote_mod, "client", lambda: rt)
    with pytest.raises(rewrite.RewriteUnavailable):
        ev.execute(ev.RunSpec(sets=["set-e"], mode="retrieval", profile="baseline"), client=fake, run_id="20260101-000000-aaaa")
    saved = ev.get_run("20260101-000000-aaaa")
    assert saved["status"] == "failed" and "Check Remote AI" in saved["error"] and len(saved["results"]) == 2


def test_no_set_means_every_set_that_runs_here(capsys, monkeypatch):
    from rag_studio import cli
    seen = {}
    monkeypatch.setattr(ev, "execute", lambda spec, **k: seen.setdefault("sets", spec.sets) and (_ for _ in ()).throw(RuntimeError("stop")))
    cli.main(["eval", "--mode", "retrieval", "--profile", "baseline"])
    assert seen["sets"] == ["set-a", "set-b", "set-c", "set-d"]   # Set E needs Remote AI: it is named on purpose


# ------------------------------------------------------------------ what the Evaluation page shows

def test_the_brief_names_each_m5_switch_as_a_run_records_it():
    """A run records the reranker, the rewrite and the multi-step search as full descriptions; the
    history, the ladder and a comparison show them as names."""
    from rag_studio import agent, rerank, rewrite
    cfg = _cfg({"method": "dense", "rerank": rerank.get("qwen3-4b").describe(20),
                "rewrite": rewrite.describe("multi"), "agent": agent.describe(3)})
    cfg |= {"followup": rewrite.describe("standalone"), "index": {"config": {"contextual": True}}}
    b = ev._brief(cfg)
    assert b["rerank"] == "qwen3-4b (top 20)" and b["rerank_name"] == "Remote AI · 4B" and b["rerank_depth"] == 20
    assert b["rewrite"] == "multi" and b["agent_rounds"] == 3
    assert b["followup"] == "rewritten" and b["contextual"] is True
    plain = ev._brief(_cfg({"method": "dense"}))
    assert plain["rerank"] is plain["rerank_name"] is plain["rewrite"] is plain["agent_rounds"] is None
    assert plain["followup"] is None and plain["contextual"] is False
    assert ev._brief(_cfg({"method": "dense"}) | {"followup": {"as_typed": True}})["followup"] == "as typed"


def test_follow_up_stats_judge_the_rewrite_then_retrieval_then_the_answer():
    def row(ok, hit, outcome):
        return {"history": [{"question": "q", "answer": "a"}], "rewrite_ok": ok, "hit_rank": hit, "outcome": outcome}
    rows = [row(True, 1, "PASS"), row(True, None, "REFUSED"), row(False, 2, "WRONG"), row(None, 3, None),
            {"history": None, "rewrite_ok": None, "hit_rank": 1, "outcome": "PASS"}]       # not a follow-up
    assert ev.followup_stats(rows) == {"n": 4, "rewritten": 3, "rewrite_ok": 2, "hits": 3, "scored": 3, "passes": 1}
    assert ev.followup_stats(rows[-1:]) is None                     # no follow-up asked: "-", never 0


def test_a_set_e_run_summarises_its_follow_ups_and_old_runs_get_none(fake, monkeypatch):
    _remote_for_followups(monkeypatch, lambda user: user.split("Last message: ")[1])
    run = ev.execute(ev.RunSpec(sets=["set-e"], mode="retrieval", profile="baseline"), client=fake)
    f = run["summary"]["followups"]
    assert f["n"] == 8 and f["rewritten"] == 8 and f["scored"] == 0
    other = ev.execute(ev.RunSpec(sets=["set-b"], mode="retrieval"), client=fake)
    from rag_studio.files import read_json, write_json_atomic
    saved = read_json(ev.run_path(other["id"]))
    del saved["summary"]["followups"]                            # as a run saved before M5
    write_json_atomic(ev.run_path(other["id"]), saved)
    assert ev.get_run(other["id"])["summary"]["followups"] is None
