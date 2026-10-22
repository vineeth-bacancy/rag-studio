"""How it works step 8: rewriting replayed from recorded runs (no model call)."""
from __future__ import annotations

import pytest

from rag_studio import learn, paths, profiles
from rag_studio.files import write_json_atomic

PROFILE = profiles.DEFAULT


def _run(run_id: str, *, search: dict, results: list[dict], ladder: dict | None = None, as_typed: bool = False,
         profile: str = PROFILE) -> None:
    write_json_atomic(paths.RUNS / f"{run_id}.json", {
        "id": run_id, "status": "done", "started_at": "2026-10-22T00:00:00+00:00", "finished_at": None,
        "label": run_id, "ladder": ladder, "sets": [], "progress": None, "summary": None, "seconds": 1.0,
        "error": None, "config": {},
        "spec": {"sets": ["set-b"], "mode": "retrieval", "profile": profile, "search": search, "as_typed": as_typed},
        "results": results})


def _row(qid: str, rank: int | None, **extra) -> dict:
    return {"id": qid, "set": "set-b", "question": "q", "hit_rank": rank, "best_rank": rank, "targets": [], **extra}


def test_each_technique_replays_its_newest_ladder_run(tmp_data):
    _run("20261022-000001-aaaa", search={"method": "dense"}, results=[_row("B4", 8)], ladder={"row": 6, "label": "x"})
    _run("20261022-000002-bbbb", search={"method": "dense"}, results=[_row("B4", 3)])      # newer, not a ladder run
    _run("20261022-000003-cccc", search={"method": "dense", "rewrite": "multi"},
         results=[_row("B4", 1, rewrites=[{"technique": "multi", "value": {"queries": ["a", "b", "c"]}}])])
    _run("20261022-000004-dddd", search={"method": "dense", "agent_rounds": 3}, results=[_row("B1", 1)])  # not B4
    _run("20261022-000005-eeee", search={"method": "dense", "rewrite": "hyde"}, results=[_row("B4", 2)],
         profile="baseline")                                                                  # another index
    lab = learn.rewrite_lab("B4", PROFILE)
    by = {t["technique"]: t for t in lab["techniques"]}
    assert list(by) == ["asked", "decompose", "multi", "hyde", "agent"]
    assert by["asked"]["run_id"] == "20261022-000001-aaaa"        # the ladder run beats the newer plain one
    assert by["multi"]["result"]["rewrites"][0]["value"]["queries"] == ["a", "b", "c"]
    assert by["decompose"]["run_id"] is None and by["hyde"]["run_id"] is None and by["agent"]["run_id"] is None


def test_as_asked_uses_the_same_reranker_as_the_techniques(tmp_data):
    """A row-8 run reranks with the 4B; "as asked" must be the 4B run (row 6d), not the newer 8B one."""
    _run("20261022-000001-aaaa", search={"method": "dense", "rerank": "qwen3-4b"}, results=[_row("B4", 1)],
         ladder={"row": 6, "label": "4B"})
    _run("20261022-000002-bbbb", search={"method": "dense", "rerank": "qwen3-8b"}, results=[_row("B4", 1)],
         ladder={"row": 6, "label": "8B"})
    _run("20261022-000003-cccc", search={"method": "dense", "rerank": "qwen3-4b", "rewrite": "multi"},
         results=[_row("B4", 2)], ladder={"row": 8, "label": "multi"})
    by = {t["technique"]: t for t in learn.rewrite_lab("B4", PROFILE)["techniques"]}
    assert by["asked"]["run_id"] == "20261022-000001-aaaa"


def test_a_follow_up_compares_rewritten_with_as_typed(tmp_data):
    _run("20261022-000001-aaaa", search={"method": "dense"}, results=[_row("E1", 1, standalone_question="South Asia 2027")])
    _run("20261022-000002-bbbb", search={"method": "dense"}, results=[_row("E1", None)], as_typed=True)
    lab = learn.rewrite_lab("E1", PROFILE)
    by = {t["technique"]: t for t in lab["techniques"]}
    assert list(by) == ["followup", "as_typed"] and lab["history"]
    assert by["followup"]["result"]["standalone_question"] == "South Asia 2027"
    assert by["as_typed"]["run_id"] == "20261022-000002-bbbb"


def test_unknown_question_and_technique_are_refused(tmp_data, client):
    with pytest.raises(LookupError):
        learn.rewrite_lab("Z9", PROFILE)
    assert client.get("/api/learn/rewrite", params={"question": "Z9"}).status_code == 404
    assert client.get("/api/learn/rewrite", params={"question": "../x"}).status_code == 422
    assert client.get("/api/learn/rewrite/live", params={"q": "x", "technique": "rm -rf"}).status_code == 422
    with pytest.raises(ValueError):
        learn.rewrite_live("x", PROFILE, "standalone")
