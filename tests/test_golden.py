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


# ------------------------------------------------------------------ Set E (follow-ups)

def test_set_e_is_valid_remote_only_and_every_case_has_a_recorded_first_turn():
    from rag_studio import eval as ev
    e = ev.load_sets()["set-e"]
    assert e["remote_only"] is True and e["scored"] is True and 6 <= len(e["questions"]) <= 8
    for q in e["questions"]:
        assert q["history"] and all(t["question"] and t["answer"] for t in q["history"])
        assert q["standalone_expect"] and q["question"].strip()
    traps = [q for q in e["questions"] if q["expect"] == "REFUSE"]
    assert len(traps) == 1 and traps[0]["id"] == "E7" and not ev.targets(traps[0])
    assert any(q.get("standalone_forbid") for q in e["questions"])             # the change of subject


def test_set_e_answer_pages_exist_in_the_pdfs():
    from rag_studio import eval as ev
    assert [p for p in ev.check_pages({"set-e": ev.load_sets()["set-e"]})] == []


@pytest.mark.parametrize("change, message", [
    ({"history": []}, "recorded turns"),
    ({"history": [{"question": "q"}]}, "recorded turns"),
    ({"standalone_expect": []}, "must contain"),
    ({"standalone_forbid": [3]}, "must not contain"),
])
def test_a_bad_follow_up_case_is_refused_with_its_name(change, message):
    from rag_studio import eval as ev
    q = {"id": "X1", "question": "And?", "history": [{"question": "q", "answer": "a"}], "doc_id": "d",
         "answer_pages": [1], "expect": ["x"]} | change
    with pytest.raises(ev.GoldenError, match=message):
        ev._validate("set_x.json", {"id": "set-x", "name": "X", "description": "d", "scored": True, "questions": [q]})


def test_a_rewrite_check_needs_a_history_to_check():
    from rag_studio import eval as ev
    q = {"id": "X1", "question": "?", "standalone_expect": ["a"], "doc_id": "d", "answer_pages": [1], "expect": ["x"]}
    with pytest.raises(ev.GoldenError, match="only be checked for a question with a 'history'"):
        ev._validate("set_x.json", {"id": "set-x", "name": "X", "description": "d", "scored": True, "questions": [q]})


def test_the_rewrite_is_judged_by_its_words():
    from rag_studio import eval as ev
    q = {"history": [{"question": "q", "answer": "a"}], "standalone_expect": ["iMac", "stand"],
         "standalone_forbid": ["GPO"]}
    assert ev.rewrite_ok(q, "What is the iMac stand made of?") is True
    assert ev.rewrite_ok(q, "What is the iMac stand made of, GPO?") is False       # carried the old subject over
    assert ev.rewrite_ok(q, "What is it made of?") is False
    assert ev.rewrite_ok(q, None) is None and ev.rewrite_ok({"standalone_expect": ["a"]}, "a") is None
