"""Cleaning on the real corpus: the M2 targets from the build spec (§17).

Ligatures -> 0, broken "Th" -> 0, every real compound keeps its hyphen, hidden
text dropped and flagged; plus printed page labels, headers moved out of the
body, and contents pages and reference lists tagged.

The cleaned documents come from `ingest.cleaned_document`, which caches them
under data/cache (rebuilt, gitignored): about 25 s the first time, about 1 s
after. Docling's checks skip until `./rag ingest --profile docling-clean` has
converted the PDFs (that takes about 15 minutes).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from rag_studio import clean, ingest, parse, parse_docling, paths

GEP = paths.PDFS / "wb_gep_2026_06.pdf"
GPO = paths.PDFS / "gpo_2018_annual_report.pdf"
IMAC = paths.PDFS / "apple_imac_per_2024_10.pdf"
LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"
# The 15 broken words FINDINGS.md found in GEP (#7). Used here only to MEASURE;
# the cleaner repairs from the font's glyph names and never sees this list.
BROKEN_TH = re.compile(r"\bT(?:e|is|ese|ey|at|eir|ere|erefore|roughout|rough|riving|ird|ereafter|eory|elejane)\b")


@pytest.fixture(scope="module")
def py() -> dict[str, clean.CleanDoc]:
    return {p.stem: ingest.cleaned_document(p, "pymupdf") for p in (GEP, GPO, IMAC)}  # type: ignore[misc]


@pytest.fixture(scope="module")
def dl(py) -> dict[str, clean.CleanDoc]:
    if not all(parse_docling.is_converted(p) for p in (GEP, GPO, IMAC)):
        pytest.skip("Docling has not converted the PDFs. Run: ./rag ingest --profile docling-clean")
    return {p.stem: ingest.cleaned_document(p, "docling") for p in (GEP, GPO, IMAC)}  # type: ignore[misc]


def text(doc: clean.CleanDoc) -> str:
    return "\n".join(p.text for p in doc.pages)


# --------------------------------------------------------------------------- the M2 targets

def test_layout_reader_rebuilds_get_text_exactly():
    for pdf in (IMAC, GPO):
        with pymupdf.open(pdf) as doc:
            plain = [p.get_text() for p in doc]
        assert [lay.text_raw for _, lay in parse.read_layout(pdf)] == plain


def test_ligatures_go_to_zero(py, dl):
    raw = "".join(p.text_raw for p in py["wb_gep_2026_06"].pages)
    assert sum(raw.count(c) for c in LIGATURES) == 1507          # FINDINGS #4
    assert py["wb_gep_2026_06"].counts["ligatures"] == 1507
    for docs in (py, dl):
        assert sum(text(d).count(c) for d in docs.values() for c in LIGATURES) == 0


def test_broken_th_goes_to_zero_from_glyph_names(py, dl):
    gep = py["wb_gep_2026_06"]
    raw = "\n".join(p.text_raw for p in gep.pages)
    assert len(BROKEN_TH.findall(unicodedata.normalize("NFKC", raw))) == 325   # FINDINGS #7
    assert gep.counts["th"] == 325                                             # one repair per glyph
    for docs in (py, dl):
        assert sum(len(BROKEN_TH.findall(text(d))) for d in docs.values()) == 0
    assert "Trafficking" in text(py["gpo_2018_annual_report"])                 # no doubled letters
    assert "Traffiicking" not in text(py["gpo_2018_annual_report"])


def _compound_oracle() -> list[tuple[int, str, str]]:
    """Line-end hyphens whose hyphenated form the document writes elsewhere:
    the FINDINGS #3 "quick check", rebuilt independently from the raw text."""
    with pymupdf.open(GEP) as doc:
        pages = [unicodedata.normalize("NFKC", p.get_text()) for p in doc]
    full = "\n".join(pages).lower()
    out = []
    for n, t in enumerate(pages, start=1):
        lines = t.split("\n")
        for i, ln in enumerate(lines[:-1]):
            m = re.search(r"([A-Za-z]+)-\s*$", ln)
            h = re.match(r"\s*([A-Za-z]+)", lines[i + 1])
            if m and h and re.search(rf"\b{re.escape(m.group(1).lower())}-{re.escape(h.group(1).lower())}\b", full):
                out.append((n, m.group(1), h.group(1)))
    return out


def test_every_real_compound_keeps_its_hyphen(py):
    oracle = _compound_oracle()
    assert len(oracle) >= 127                                    # FINDINGS counted 127; this oracle finds 140
    calls = {(c["page"], c["left"], c["right"]): c["decision"] for c in py["wb_gep_2026_06"].info["hyphen_calls"]}
    lost = [o for o in oracle if calls.get(o) != "kept"]
    assert lost == []


def test_docling_keeps_the_compounds_it_used_to_merge(py, dl):
    gep_py, gep_dl = py["wb_gep_2026_06"], dl["wb_gep_2026_06"]
    assert gep_dl.counts["hyphen"] > 100                         # Docling had merged them
    kept = [c for c in gep_py.info["hyphen_calls"] if c["decision"] == "kept"]
    missing = [c for c in kept if f"{c['left']}{c['right']}" in re.findall(r"[A-Za-z]+", gep_dl.pages[c["page"] - 1].text)]
    assert missing == []                                          # no "conflictrelated" left


def test_hidden_text_is_dropped_and_recorded(py, dl):
    gpo = py["gpo_2018_annual_report"]
    p29 = gpo.pages[28]
    assert [h["reason"] for h in p29.hidden] == ["covered", "covered"]
    assert "GOVERNMENT" in p29.text_raw.replace(" ", "") and "GOVERNMENT" not in p29.text.replace(" ", "")
    reasons = Counter(h["reason"] for p in gpo.pages for h in p.hidden)
    assert reasons["off_page"] >= 24                              # p15's copy of p14's figures
    assert "375.7 million" in gpo.pages[13].text                  # the visible copy on p14 stays
    gpo_dl = dl["gpo_2018_annual_report"]
    assert "GOVERNMENT" not in gpo_dl.pages[28].text.replace(" ", "")


# --------------------------------------------------------------------------- structure

def test_printed_page_labels(py):
    assert py["wb_gep_2026_06"].pages[95].label == "74"          # spec §4: printed 74 = PDF p96
    assert py["wb_gep_2026_06"].pages[18].label == "xvii"
    assert py["gpo_2018_annual_report"].pages[3].label == "2"     # narrative: printed = PDF - 2
    assert py["gpo_2018_annual_report"].pages[57].label == "27"   # financial: printed = PDF - 31


def test_running_headers_leave_the_body(py):
    gep = py["wb_gep_2026_06"]
    assert sum(1 for p in gep.pages if p.furniture) >= 163       # FINDINGS #2: 163 of 200
    assert not any("GLOBAL ECONOMIC PROSPECTS | JUNE 2026" in p.text for p in gep.pages)
    gpo = py["gpo_2018_annual_report"]
    assert sum(1 for p in gpo.pages if p.furniture) >= 73        # FINDINGS #2: 73 of 82


def test_contents_and_references_are_tagged(py):
    gep, gpo = py["wb_gep_2026_06"], py["gpo_2018_annual_report"]
    toc = {p.page for p in gep.pages if any(k["kind"] == "toc" for k in p.kinds)}
    assert toc == {7, 9, 10, 11, 12, 13}                          # spec §4 #10
    assert any(k["kind"] == "toc" for k in gpo.pages[32].kinds)   # GPO p33
    for n in (139, 140, 141, 142):                                # Chapter 3's bibliography (spec §5.3)
        assert any(k["kind"] == "references" for k in gep.pages[n - 1].kinds)
    assert all(k["kind"] == "body" for k in gep.pages[108].kinds)  # p109, the chapter summary


def test_sections_come_from_bookmarks(py):
    gep = py["wb_gep_2026_06"]
    assert gep.pages[18].sections[0]["path"] == ["Executive Summary"]
    assert gep.pages[95].sections[0]["path"][:2] == ["Ch. 2 Regional Outlooks", "South Asia"]
    assert gep.pages[29].sections[0]["path"][:2] == ["Ch. 1 Global Outlook",
                                                    "Box 1.1 How much will AI affect global growth?"]


def test_the_reports_raise_no_instruction_flag(py):
    from rag_studio import guard
    assert [p.page for d in py.values() for p in d.pages if guard.instruction_flags(p.text)] == []


# --------------------------------------------------------------------------- rules in isolation

@pytest.mark.parametrize("left, right, words, compounds, expected", [
    ("produc", "tion", {"production": 3}, set(), ("joined", "joined_form")),
    ("all", "time", {}, {"all-time"}, ("kept", "hyphenated_form")),
    ("debt", "to", {}, {"debt-to", "to-gdp"}, ("kept", "hyphenated_form")),
    ("Vance", "Cooks", {}, set(), ("kept", "capital_or_digit")),
    ("1,242", "page", {}, set(), ("kept", "capital_or_digit")),
    ("fear", "driven", {"fear": 1, "driven": 2}, set(), ("kept", "both_words")),
    ("in", "stalled", {"in": 900, "stalled": 1}, set(), ("joined", "default")),
    ("war", "ranting", {"war": 3}, set(), ("joined", "default")),
])
def test_hyphen_decisions(left, right, words, compounds, expected):
    assert clean.decide_hyphen(left, right, Counter(words), compounds) == expected


def test_marked_text_keeps_the_original_through_edits():
    m = clean.Marked()
    m.append("in")
    m.append("fl", "nfkc", "ﬂ")
    m.append("ation  rose")
    m.sub(re.compile(r" {2,}"), " ", None)
    assert m.text == "inflation rose"
    assert [(x.start, x.end, x.before) for x in m.marks] == [(2, 4, "ﬂ")]
    m.replace(0, 9, "inflation", "hyphen", None)
    assert m.marks[0].before == "inﬂation" and m.marks[0].rule == "hyphen"


def test_page_labels_follow_the_neighbours():
    def page(texts: list[str]) -> list[clean.Line]:
        return [clean.Line(page=0, block=0, box=(0, 0, 1, 1), size=9, bold=False, raw=t, raw_start=0,
                           text=clean.Marked(t), role="furniture") for t in texts]
    pages = [page(["CHAPTER 1", str(n - 22)]) for n in range(30, 36)]
    assert clean.printed_labels(pages, list(range(30, 36))) == [str(n - 22) for n in range(30, 36)]
