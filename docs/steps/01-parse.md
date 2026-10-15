# 1. Extract and parse

**What it does:** reads the text out of every PDF page, numbered the way the PDF
counts pages (1-based). That number is the one every citation shows.

There are two parsers, side by side, never one silently replacing the other:

| Parser | Profile | What it is |
|---|---|---|
| **PyMuPDF** | `baseline`, `pymupdf-clean` | The text layer, in the order the PDF draws it. Fast (all 294 pages in about 1 s) |
| **Docling** 2.139 | `docling-clean` | A layout model and a table model over each page: it knows a heading from a footer from a table cell. Its text still comes from the text layer: it does not see better, it organises better |

## PyMuPDF

**Baseline (M1):** `page.get_text()` with default flags, exactly as the
hand-built `extract.py`.

**The layout reader (M2):** `parse.read_layout()` reads the same page as lines
and characters (`get_text("rawdict")`): each character's box, the font size of
its span, whether it is raised (superscript), and whether PyMuPDF *invented* it
(`synthetic`: the spaces it inserts between spread-out letters). Its lines
rebuild `get_text()` **exactly**, on all 294 pages, so "raw" means the same text
in both paths.

**Ligature glyphs, read from the font.** GEP's body font draws "Th" as one
glyph named `T_h`. The font's ToUnicode table forgets it (its code `<01>` has no
entry), so PyMuPDF falls back to the first letter and "The" comes out "Te", 325
times. The font program still carries the glyph's name: `parse.GlyphNames`
reads every embedded font once, finds glyph names in Adobe's ligature form
(`T_h`, `f_f_i`), and the cleaning step restores what the text layer cut short.
No word list is involved; it works on any PDF with the same flaw. One guard
against overcorrecting: GPO maps its `f_f_i` glyph to "f" and then carries "fi"
as text after it, so a repair happens only when the rest is really missing
(without it, "Trafficking" became "Traffiicking").

| Measured (2026-10-15) | Value |
|---|---|
| Pages | iMac 12, GPO 82, GEP 200 (294) |
| Plain text, all pages | 0.9 s |
| Layout reader, all pages, with glyph names | 13.6 s |
| `get_text()` rebuilt exactly from the layout | 294 of 294 pages |
| "Th" glyphs repaired | 325 (GEP), 0 elsewhere |

## Docling

`parse_docling.py`. Options (all recorded in the cache key and the index
manifest): no OCR (every page has a text layer), tables in accurate mode,
heading levels from bookmarks, numbering and font style, 8 threads.

| Measured on this laptop (CPU) | Value |
|---|---|
| iMac, 12 pages | 21 s (0.57 pages/s) |
| GPO, 82 pages | 167 s (0.49 pages/s) |
| GEP, 200 pages | 678 s (0.29 pages/s) |
| All 294 pages | **866 s** (0.34 pages/s) |
| Peak memory | **6.4 GB** (GEP) |
| Models downloaded on first run | about 570 MB (layout, TableFormer), in the Hugging Face cache |
| Install | torch 2.14.1 from PyTorch's CPU index (about 200 MB, not 3 GB of CUDA wheels) |

That cost is why only `./rag ingest` runs Docling, once per PDF; the result is
cached (`data/cache/docling/`) and the API, the inspector and every re-chunk
read the cache.

**Tables become self-describing rows.** A Docling table is written into the
page text one row per line, every value next to its column:
`World: 2023 = 2.8; 2024 = 2.9; 2025e = 2.9; 2026f = 2.5; …`. A fixed-size
passage cut from the middle of a table still says which year each number is.
GEP has 35 tables, GPO 25, iMac 3.

## The 15 known defects, measured (spec §4, `FINDINGS.md`)

"Cleaned" means after the cleaning order (`02-clean.md`) and the guard
(`13-safety.md`).

| # | Defect | PyMuPDF raw | PyMuPDF cleaned | Docling cleaned |
|---|---|---|---|---|
| 1 | Tables flattened, one cell per line | every table | still flattened (lines of one block now share a line) | **fixed**: 63 tables as rows |
| 2 | Running headers and footers in the body | GEP 163 / GPO 73 pages | **moved to metadata**: GEP 174, GPO 75, iMac 11 pages | **moved** (Docling's own furniture layer) |
| 3 | Line-break hyphens | GEP 708 lines | **rejoined or kept by the document's own dictionary**; all 140 attested compounds kept | Docling rejoins every one and merges compounds ("conflictrelated"); **162 restored** from the PyMuPDF decisions |
| 4 | Ligatures | GEP 1,507 | **0** | **0** (Docling expands them) |
| 5 | Three page numbers | - | **printed page saved** before footers go (GEP p96 → "74", GPO p58 → "27") | same labels (from the PyMuPDF path) |
| 6 | Headings and captions out of order | GPO narrative | unchanged (reading order is PyMuPDF's) | **reading order from the layout model** |
| 7 | Dropped "h" in "Th" | GEP 325 | **0** (glyph names) | **0** (docling-parse reads glyph names itself) |
| 8 | Charts and images | iMac p2, GEP charts | unchanged (M7) | pictures found (223 in GEP), their labels kept as text; descriptions are M7 |
| 9 | Footnote markers glued to words | iMac 24 (by search) | **removed by position**: 27 iMac, 101 GEP | not fixed: "recycled content 1" (spaced, not removed) |
| 10 | Contents pages and dot leaders | GPO p33, GEP 7, 9-13 | **leaders collapsed** (231 lines), pages **tagged `toc`** | tagged (`document_index`) |
| 11 | Letter-spaced display text | GPO 37 lines | **fixed**: invented spaces dropped (35 runs) | **worse**: "Apr i l", "PUBL I SH I NG", "Nor th Capi tol" |
| 12 | Footer printed twice | GPO 5 pages | gone with #2 | gone with #2 |
| 13 | Hidden text | GPO p29 (known) | **dropped and flagged**: 9 pages, see below | **dropped** (items over the guard's hidden boxes) |
| 14 | Near-empty pages | GPO 11, GEP 23 | skipped: GPO 12 (p48 is only furniture now), GEP 23 | GPO 12, GEP 21 |
| 15 | Different structure per PDF | - | **section paths**: bookmarks (GEP 18, GPO 56), then headings by font size | section headers from the layout model |

**Hidden text found** (none of it visible when the page is drawn): GPO p1, p29
and p30 (footers under a full-page fill), p5 and p15 (text placed left of the
page edge: p15 carries a copy of p14's "By the Numbers" figures), p28 (an
address block clipped away); GEP p125 (1.6 pt chart labels), p195 (a
white-on-white "Column1" left from a spreadsheet), p200 (one off-page letter).

## Where a passage sits on its page

- **Baseline:** a passage is a span of the page's plain text, and PyMuPDF's
  word list holds the same words in the same order (checked on all 294 pages),
  so the span maps exactly to word boxes.
- **PyMuPDF + cleaning:** cleaning changed the text ("production" was
  "produc-⏎tion"), so the passage's words are aligned with the page's words in
  order (`parse.aligned_boxes`); the outline is the matched stretch, and the
  share of words found in place is reported (97.9% on a GEP p19 passage).
- **Docling:** each layout item knows its box and where its text sits in the
  page text, so the outline is the boxes of the items the passage covers.

## Next

M3 uses the structure found here: section labels on passages, and a cap per
section so a reference list cannot fill the top 5.
