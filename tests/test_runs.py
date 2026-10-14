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
