"""The test questions: valid, pointing at real pages, and the user's 9 kept verbatim."""
from __future__ import annotations

import pytest

from rag_studio import eval as ev

# The user's questions (2026-10-15: "include these questions in the tests and
# evaluations"), word for word, grouped by document. They are Set A.
USER_QUESTIONS = {
    "gpo_2018_annual_report": [
        "How many depository libraries did GPO staff visit in FY 2018?",
        "What new digital system replaced FDsys, and when was it fully deployed?",
        "How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?",
    ],
    "wb_gep_2026_06": [
        "What is the growth outlook for South Asia in this report?",
        "What does Chapter 3 say about sovereign debt in developing economies?",
        "According to the executive summary, what are the main risks to the global outlook?",
    ],
    "apple_imac_per_2024_10": [
        "How much less energy does iMac use compared to the ENERGY STAR requirement?",
        "What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?",
        "What material is the iMac stand made from, and what percentage of it is recycled?",
    ],
}


@pytest.fixture(scope="module")
def sets():
    return ev.load_sets()


def test_all_sets_load_and_point_at_real_pages(sets):
    assert {"set-a", "set-b", "set-d"} <= set(sets)
    assert ev.check_pages(sets) == []


def test_user_questions_are_in_set_a_verbatim(sets):
    by_text = {q["question"]: q for q in sets["set-a"]["questions"]}
    for doc_id, questions in USER_QUESTIONS.items():
        for text in questions:
            assert text in by_text, f"missing from Set A: {text}"
            q = by_text[text]
            assert q["doc_id"] == doc_id
            assert q["answer_pages"] and isinstance(q["expect"], list) and q["expect"]


def test_set_a_keeps_the_baseline_facts_and_traps(sets):
    a = {q["id"]: q for q in sets["set-a"]["questions"]}
    assert a["A1"]["expect"] == ["166"] and a["A1"]["answer_pages"] == [5, 21]
    assert a["A6"]["answer_pages"] == [19]
    assert a["A8"]["expect"] == ["346", "45", "50"]
    assert a["T1"]["expect"] == a["T2"]["expect"] == "REFUSE"
    assert len(a) == 11


def test_set_b_is_retrieval_only(sets):
    b = sets["set-b"]
    assert b["scored"] is False and all(q["expect"] is None for q in b["questions"])


def test_set_d_covers_the_edge_categories(sets):
    cats = {q.get("category") for q in sets["set-d"]["questions"]}
    for c in ("Near-miss number", "Arithmetic", "Table cell", "Flattened table", "Paraphrase",
              "Typos", "Another language", "Two documents", "False premise",
              "Instruction override", "Wrong product", "Wrong year"):
        assert c in cats


def test_a_two_document_question_has_two_targets(sets):
    d10 = next(q for q in sets["set-d"]["questions"] if q["id"] == "D10")
    assert [t["doc_id"] for t in ev.targets(d10)] == ["gpo_2018_annual_report", "apple_imac_per_2024_10"]


@pytest.mark.parametrize("bad", [
    {"id": "X", "question": "q", "doc_id": "d", "answer_pages": [], "expect": ["a"]},
    {"id": "X", "question": "q", "doc_id": "d", "answer_pages": [0], "expect": ["a"]},
    {"id": "X", "question": "q", "doc_id": "d", "answer_pages": [1], "expect": "REFUSE"},
    {"id": "X", "question": "q", "doc_id": "d", "answer_pages": [1], "expect": [{"pattern": "("}]},
    {"id": "X", "question": " ", "doc_id": "d", "answer_pages": [1], "expect": ["a"]},
])
def test_malformed_questions_are_refused(bad):
    with pytest.raises(ev.GoldenError):
        ev._validate("t.json", {"id": "t", "name": "T", "description": "", "scored": True,
                                "questions": [bad]})
