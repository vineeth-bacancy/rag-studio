# 1. Extract and parse

**What it does:** reads the text out of every PDF page, one record per page,
numbered the way the PDF counts pages (1-based). That number is the one every
citation shows.

## Baseline (M1)

`rag_studio/parse.py`: PyMuPDF 1.28.2, `page.get_text()` with default flags,
exactly as the hand-built `extract.py`.

| Measured (2026-10-15) | Value |
|---|---|
| Pages | iMac 12, GPO 82, GEP 200 (294) |
| Time for all 294 pages | 0.9 s |
| Near-empty pages (≤ 200 non-whitespace characters, skipped) | GPO 11, GEP 23, iMac 0: the same lists as spec §4 |

Each page keeps `text_raw` (what PyMuPDF returned) beside `text` (after
cleaning), so a later step can always show what changed.

**Where a passage sits on its page.** PyMuPDF's word list and the page text
hold the same words in the same order: checked on all 294 pages
(`" ".join(words).split() == text.split()`). So a passage, stored as a character
span of its page, maps exactly to word boxes. That is how the Ask page outlines
a cited passage on the rendered page (`passage_boxes()`).

## Known problems it leaves (spec §4, `FINDINGS.md`)

Tables flattened one cell per line (#1), running headers inside the body (#2),
"Th" broken to "T" in GEP (#7), footnote markers glued to words (#9), hidden
text under a page fill (#13). Set D's question D6 (Thailand's forecast, a
flattened table) fails partly because of #1.

## Next

M2 adds Docling as a second parser beside this one, never a silent
replacement, and the page inspector shows both side by side.
