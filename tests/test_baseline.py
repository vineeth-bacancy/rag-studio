"""Baseline reproduction (spec §6-7): the same pages, passages and IDs as the
hand-built pipeline in ~/Dev/RAG/advanced-rag."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

from rag_studio import chunk, paths
from rag_studio.parse import parse_pdf, passage_boxes

# sha256 over every "chunk_id\ntext\n" of the 1,190 baseline passages, taken
# from a run that matched advanced-rag/outputs/chunks.jsonl passage for passage.
FINGERPRINT = "abf51fa956712c5565f46e371f12678f2c820b7bd051658c51e553d9189a5986"
BASELINE_CHUNKS = Path.home() / "Dev/RAG/advanced-rag/outputs/chunks.jsonl"


@pytest.fixture(scope="module")
def docs():
    return {p.stem: parse_pdf(p) for p in sorted(paths.PDFS.glob("*.pdf"))}


@pytest.fixture(scope="module")
def passages(docs):
    return [p for d in docs.values() for p in chunk.chunk_pages(d.pages, chunk.BASELINE)]


def test_page_counts(docs):
    assert {k: d.page_count for k, d in docs.items()} == {
        "apple_imac_per_2024_10": 12, "gpo_2018_annual_report": 82, "wb_gep_2026_06": 200}


def test_near_empty_pages_match_the_spec(docs):
    near = {k: [p.page for p in d.pages if p.is_near_empty] for k, d in docs.items()}
    assert near["gpo_2018_annual_report"] == [1, 2, 29, 30, 31, 32, 34, 50, 51, 53, 57]
    assert near["wb_gep_2026_06"] == [1, 2, 3, 4, 5, 8, 14, 16, 22, 23, 24, 69, 70, 106, 107,
                                      108, 143, 144, 184, 185, 186, 198, 199]
    assert near["apple_imac_per_2024_10"] == []


def test_1190_passages_per_document(passages):
    assert len(passages) == 1190
    assert collections.Counter(p.doc_id for p in passages) == {
        "wb_gep_2026_06": 923, "gpo_2018_annual_report": 233, "apple_imac_per_2024_10": 34}


def test_passages_are_identical_to_the_baseline(passages):
    h = hashlib.sha256()
    for p in passages:
        h.update(f"{p.chunk_id}\n{p.text}\n".encode())
    assert h.hexdigest() == FINGERPRINT


@pytest.mark.skipif(not BASELINE_CHUNKS.is_file(), reason="advanced-rag outputs not on this machine")
def test_passages_match_advanced_rag_file(passages):
    base = [json.loads(line) for line in BASELINE_CHUNKS.open(encoding="utf-8")]
    assert [(b["chunk_id"], b["text"]) for b in base] == [(p.chunk_id, p.text) for p in passages]


def test_spans_point_at_the_passage_text(docs, passages):
    pages = {(d.doc_id, p.page): p for d in docs.values() for p in d.pages}
    for p in passages[::37]:
        assert pages[(p.doc_id, p.page)].text[p.char_start:p.char_end] == p.text


def test_chunker_keeps_the_mid_word_start_defect():
    # Spec §5.11: the start is end - overlap, not moved to a space. M3 fixes it
    # as a separate chunker; the baseline must keep it to be the "before" row.
    text = "the quick brown fox jumps over the lazy dog " * 40
    spans = chunk.split_fixed(text, size=100, overlap=15)
    assert any(not text[a - 1].isspace() for a, _ in spans[1:])   # some start mid-word


def test_chunker_never_ends_mid_word():
    text = "word " * 400
    for _, b in chunk.split_fixed(text, 1000, 150):
        assert b == len(text.rstrip()) or text[b] == " "


def test_passage_boxes_land_on_the_page():
    r = passage_boxes(paths.PDFS / "wb_gep_2026_06.pdf", 19, 900, 1800)
    assert r["aligned"] and r["boxes"]
    for b in r["boxes"]:
        assert 0 <= b["x0"] < b["x1"] <= 1 and 0 <= b["y0"] < b["y1"] <= 1
