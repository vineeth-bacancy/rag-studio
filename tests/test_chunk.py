"""Chunkers (M3): the word-boundary fix, structure-aware passages with section
labels, semantic cuts and the playground's chunkers, on the real PDFs."""
from __future__ import annotations

import pytest

from rag_studio import chunk, clean, ingest, paths
from rag_studio.parse import parse_pdf

GEP = paths.PDFS / "wb_gep_2026_06.pdf"
IMAC = paths.PDFS / "apple_imac_per_2024_10.pdf"


@pytest.fixture(scope="module")
def parsed():
    return {p.stem: parse_pdf(p) for p in sorted(paths.PDFS.glob("*.pdf"))}


@pytest.fixture(scope="module")
def cleaned():
    """PyMuPDF + cleaning, from the shared cache (computed once if missing)."""
    return {p.stem: ingest.cleaned_document(p, "pymupdf") for p in sorted(paths.PDFS.glob("*.pdf"))}


def _starts_mid_word(text: str, a: int) -> bool:
    return a > 0 and not text[a - 1].isspace() and not text[a].isspace()


def test_the_baseline_starts_mid_word_and_the_fix_does_not(parsed):
    base = fixed = 0
    for d in parsed.values():
        for page in d.pages:
            if page.is_near_empty:
                continue
            base += sum(_starts_mid_word(page.text, a) for a, _ in chunk.split_fixed(page.text, 1000, 150))
            fixed += sum(_starts_mid_word(page.text, a)
                         for a, _ in chunk.split_fixed(page.text, 1000, 150, snap_start=True))
    assert base > 100            # "n the upside": the defect the baseline keeps on purpose
    assert fixed == 0


def test_the_baseline_spec_still_describes_itself_as_in_m1():
    assert chunk.BASELINE.describe() == {"id": "fixed-1000-150", "size": 1000, "overlap": 150,
                                         "snap_start": False}


def test_structure_passages_are_exact_slices_within_one_section(cleaned):
    spanning = total = 0
    for d in cleaned.values():
        by_page = {p.page: p for p in d.pages}
        passages = chunk.chunk_document(d.pages, chunk.STRUCTURE, title="T")
        assert passages
        for p in passages:
            page = by_page[p.page]
            assert p.text == page.text[p.char_start:p.char_end]
            assert len(p.text) <= chunk.STRUCTURE.size + chunk.MIN_FRAGMENT
            touched = [page.at(page.sections, p.char_start, "path") or []] + [
                s["path"] for s in page.sections if p.char_start < s["start"] < p.char_end]
            assert p.meta["section_path"] in {" > ".join(t) or None for t in touched}, p.chunk_id
            assert p.text.strip() not in {h["text"] for h in page.headings}, p.chunk_id
            total += 1
            spanning += len({tuple(t) for t in touched}) > 1
    # Only a fragment of under 100 characters ever joins another section.
    assert spanning / total < 0.05


def test_the_executive_summary_risks_paragraph_is_one_labelled_passage(cleaned):
    passages = chunk.chunk_document(cleaned["wb_gep_2026_06"].pages, chunk.STRUCTURE,
                                    title="Global Economic Prospects -- June 2026")
    risks = next(p for p in passages if p.page == 19 and p.text.startswith("Risks to the outlook"))
    assert risks.text.endswith("could lift activity.")
    assert risks.label == "Document: Global Economic Prospects -- June 2026\nSection: Executive Summary"
    assert risks.embed_text == risks.label + "\n" + risks.text


def test_a_line_break_inside_a_sentence_does_not_end_a_paragraph(cleaned):
    page = cleaned["wb_gep_2026_06"].pages[18]
    assert "inflation with\nsupporting activity" in page.text
    passages = chunk.chunk_document([page], chunk.STRUCTURE, title="T")
    assert not any(p.text.startswith("supporting activity") for p in passages)


def test_contents_pages_are_labelled_as_such(cleaned):
    passages = chunk.chunk_document(cleaned["wb_gep_2026_06"].pages, chunk.STRUCTURE, title="T")
    toc = [p for p in passages if p.meta["kind"] == "toc"]
    assert toc and all("contents page" in p.label.lower() for p in toc)


def _page(text: str) -> clean.CleanPage:
    return clean.CleanPage(doc_id="x", page=1, width=1, height=1, text_raw=text, text=text,
                           marks=[], raw_marks=[], sections=[{"start": 0, "path": ["S"]}],
                           kinds=[{"start": 0, "kind": "body"}])


def _topic_embed(texts: list[str]) -> list[list[float]]:
    """Sentences about cats point one way, about tax another."""
    return [[1.0, 0.1] if "cat" in t else [0.1, 1.0] for t in texts]


def test_semantic_cuts_where_the_topic_changes():
    text = ("The cat sat down. A cat likes milk. Every cat purrs. "
            "Tax rates rose. The tax burden grew. Tax revenue fell.")
    out = chunk.playground(_page(text), "semantic", 1000, 0, embed=_topic_embed, percentile_p=50)
    spans = [text[a:b] for a, b in out["spans"]]
    assert spans == ["The cat sat down. A cat likes milk. Every cat purrs.",
                     "Tax rates rose. The tax burden grew. Tax revenue fell."]
    assert sum(c["cut"] for c in out["curve"]) == 1


@pytest.mark.parametrize("method", ["fixed", "recursive", "sentence", "structure"])
def test_playground_spans_lie_in_the_page_and_respect_the_size(cleaned, method):
    page = cleaned["wb_gep_2026_06"].pages[18]
    out = chunk.playground(page, method, 400, 80)
    spans = out["spans"]
    assert spans and all(0 <= a < b <= len(page.text) for a, b in spans)
    assert all(b - a <= 400 + chunk.MIN_FRAGMENT for a, b in spans)
    assert [a for a, _ in spans] == sorted(a for a, _ in spans)


def test_sentence_chunks_overlap_by_whole_sentences(cleaned):
    page = cleaned["wb_gep_2026_06"].pages[18]
    spans = chunk.playground(page, "sentence", 400, 150)["spans"]
    starts = {a for a, _ in chunk.sentences(page.text, 0, len(page.text))}
    assert all(a in starts for a, _ in spans)
    assert any(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))   # some overlap


def test_a_gep_box_ends_where_its_pages_stop_naming_it(cleaned):
    """Bookmarks mark where a box starts, not where it ends (M2 left this for M3)."""
    pages = cleaned["wb_gep_2026_06"].pages
    ch3 = "Ch. 3 A Rising Challenge: Sovereign Debt Levels and Interest Rates in EMDEs"

    def path(n: int) -> list[str]:
        return pages[n - 1].sections[0]["path"]

    assert path(117)[:2] == [ch3, "Box 3.1 The costs of hidden debt"]
    assert path(119)[:2] == [ch3, "Box 3.1 The costs of hidden debt"]
    assert path(120)[0] == ch3 and not any(p.startswith("Box") for p in path(120))
    assert path(139) == [ch3, "References"]
    assert path(36)[0] == "Ch. 1 Global Outlook" and not any(p.startswith("Box") for p in path(36))
