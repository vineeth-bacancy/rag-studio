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
    r = retrieve.Retriever(profiles.DEFAULT, client=f)
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
    run = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b"), client=fake)
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
    before = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b"), client=fake)
    fake.answer_text = "I could not find this in the documents."
    after = ev.execute(ev.RunSpec(sets=["set-a"], mode="answer", model="qwen3:1.7b",
                                  prompt_version="v3"), client=fake)
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
