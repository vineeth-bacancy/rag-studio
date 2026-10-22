"""Baseline retrieval reproduction on the real index (spec §7): Set A 8/9 with
the miss at rank 45, Set B 5/6 with the miss at rank 11. Needs Ollama and a
built index (`./rag ingest`); skips cleanly without them. About a second."""
from __future__ import annotations

import pytest

from rag_studio import eval as ev
from rag_studio import profiles, store

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def retriever():
    if store.active_manifest(profiles.BASELINE.id) is None:
        pytest.skip("No baseline index yet: run ./rag ingest")
    from rag_studio.retrieve import retriever as active

    return active(profiles.BASELINE.id)


def ranks(retriever, set_id: str) -> dict[str, tuple[int | None, int | None]]:
    out = {}
    for q in ev.load_sets()[set_id]["questions"]:
        where = ev.targets(q)
        if not where:
            continue
        hits, _ = retriever.dense(q["question"], ev.DEEP_K)
        out[q["id"]] = (ev.first_answer_rank(hits[:5], where), ev.first_answer_rank(hits, where))
    return out


def test_set_a_recall_matches_the_hand_built_baseline(retriever):
    r = ranks(retriever, "set-a")
    assert sum(1 for top, _ in r.values() if top) == 8
    assert r["A6"] == (None, 45)


def test_set_b_recall_matches_the_hand_built_baseline(retriever):
    r = ranks(retriever, "set-b")
    assert sum(1 for top, _ in r.values() if top) == 5
    assert r["B4"] == (None, 11)
