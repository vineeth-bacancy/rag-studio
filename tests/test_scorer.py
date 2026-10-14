"""The scorer on fake answers (spec §5.10: under a second, and it catches scorer
bugs before a 6-minute real run)."""
from __future__ import annotations

import pytest

from rag_studio.answer import cited_numbers, citations
from rag_studio.eval import first_answer_rank, score_answer
from rag_studio.prompt import REFUSAL

SOURCES = [{"chunk_id": f"d:p{p}:c0", "doc_id": "d", "page": p} for p in (5, 21, 9)]


@pytest.mark.parametrize("expect,text,outcome", [
    (["166"], "GPO staff visited 166 libraries [1].", "PASS"),
    (["govinfo", "2018"], "It was govinfo [2].", "WRONG"),                    # a fact missing
    (["govinfo", "2018"], "GOVINFO, fully deployed in FY 2018 [1].", "PASS"),  # case-insensitive
    (["166"], REFUSAL, "REFUSED"),
    (["166"], f"  {REFUSAL}\n", "REFUSED"),
    ("REFUSE", REFUSAL, "PASS"),
    ("REFUSE", "Paris.", "WRONG"),
    # The lenient-check bug from the hand-built project: a wrong fact followed
    # by the refusal is an answer, so a trap answered this way is WRONG.
    ("REFUSE", f"Net income was $20.4 million. {REFUSAL}", "WRONG"),
    ([["20.4", "20,384"]], "It was $20,384 thousand [2].", "PASS"),
    ([{"pattern": r"(?<![\d.])5\s?(%|percent)", "label": "5%"}], "Product use is 45% [1].", "WRONG"),
    ([{"pattern": r"(?<![\d.])5\s?(%|percent)", "label": "5%"}], "Transportation is 5% [1].", "PASS"),
    ([{"pattern": r"(?<![\d.])5\s?(%|percent)", "label": "5%"}], "Transportation: 5 percent.", "PASS"),
])
def test_outcomes(expect, text, outcome):
    assert score_answer(expect, text)["outcome"] == outcome


def test_found_and_missing_use_readable_labels():
    r = score_answer(["govinfo", ["20.4", "20,384"], {"pattern": "x", "label": "<1%"}], "govinfo")
    assert r["found"] == ["govinfo"]
    assert r["missing"] == ["20.4 or 20,384", "<1%"]


def test_retrieval_only_questions_have_no_outcome():
    assert score_answer(None, "anything")["outcome"] is None


def test_citation_forms():
    assert cited_numbers("A [1]. B [1][2]. C [1, 3].") == [1, 1, 2, 1, 3]
    assert cited_numbers("No citations here, 2018.") == []


def test_citations_map_to_pages_and_flag_invalid_numbers():
    c = citations("One [1]. Two [2][7]. Zero [0].", SOURCES)
    assert c["valid"] == [1, 2]
    assert c["invalid"] == [0, 7]
    assert [(m["n"], m["page"]) for m in c["map"]] == [(1, 5), (2, 21)]


def test_hit_rank_one_and_two_documents():
    hits = [{"rank": 1, "doc_id": "g", "page": 3}, {"rank": 2, "doc_id": "g", "page": 5},
            {"rank": 3, "doc_id": "a", "page": 10}]
    assert first_answer_rank(hits, [{"doc_id": "g", "pages": [5, 21]}]) == 2
    assert first_answer_rank(hits, [{"doc_id": "g", "pages": [5]}, {"doc_id": "a", "pages": [10]}]) == 3
    assert first_answer_rank(hits, [{"doc_id": "g", "pages": [5]}, {"doc_id": "a", "pages": [2]}]) is None
    assert first_answer_rank(hits, []) is None
