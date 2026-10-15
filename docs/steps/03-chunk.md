# 3. Chunk

**What it does:** cuts each page's clean text into **passages** small enough to
search and to quote.

Every chunker here keeps two promises: a passage never crosses a page (a
citation is one page), and its text is an exact slice of the page's clean
text (`char_start`, `char_end`), so the page preview can outline it.

## Baseline (M1)

`rag_studio/chunk.py`, chunker id `fixed-1000-150`: the hand-built chunker,
line for line.

- 1,000 characters with 150 of overlap; never crosses a page.
- The end moves back to the last space, so a passage never ends mid-word.
- `chunk_id = "{doc_id}:p{page}:c{i}"`.
- **The known defect is kept on purpose:** the next start is `end - overlap`,
  not moved to a space, so passages often start mid-word ("n the upside",
  "agement (LSCM)"). The baseline must keep it to be an honest "before" row.

| Measured | Value |
|---|---|
| Passages | **1,190** (iMac 34, GPO 233, GEP 923) |
| Identical to `advanced-rag/outputs/chunks.jsonl` | yes: every id and every character (sha256 fingerprint pinned in `tests/test_baseline.py`) |

## M3: four chunkers, each a scorecard row

| Chunker (id) | Profile | What it does |
|---|---|---|
| Word-boundary fix (`fixed-1000-150-snap`) | `pymupdf-snap` | The baseline, with each start moved forward to the next word. Nothing else changes |
| **Structure-aware** (`structure-1000`) | `pymupdf-structure`, `docling-structure` | Whole paragraphs (and Docling's whole tables) packed up to 1,000 characters, never across a section, a contents page or reference list, or a table's edge. A section label in front of every passage |
| Semantic (`semantic-1000-p90`) | `pymupdf-semantic` | The same sections, cut where two neighbouring sentences stop being similar. A section label in front of every passage |
| Recursive, sentence | the playground only | Taught on How it works, not indexed |

### Structure-aware passages

1. **Units.** Docling names its items, so a table is one unit and a paragraph
   is one unit. PyMuPDF's cleaned text has one paragraph per line (cleaning
   joined each paragraph's lines), so a line is a unit, except that a break
   inside a sentence ("controlling inflation with / supporting activity", a
   column break) is rejoined: the line ends without punctuation and the next
   starts lowercase. Headings come from the page's own heading list.
2. **Blocks.** Consecutive units of one section and one kind (body, contents
   page, reference list). A table is always a block of its own.
3. **Packing.** Units are joined while the passage stays within 1,000
   characters. A paragraph longer than that is cut between sentences, a table
   between rows, and a sentence longer than that between words.
4. **Headings.** A piece that is only a heading is dropped: its words are in
   the section label of every passage after it. A heading that fits with the
   paragraph after it stays in that passage.
5. **Fragments.** A piece under 100 characters joins the next passage on the
   page (else the previous one), going at most 100 characters over the limit.
   Alone it would match too much or too little: GPO's running line
   "U. S. Government Publishing Office | Financial Year In Review – FY 2018"
   survives cleaning on 10 financial pages, and GEP's charts leave "Percent".
   Tables keep their edges.

### The section label

```
Document: Global Economic Prospects -- June 2026
Section: Executive Summary
Risks to the outlook remain skewed to the downside. A renewed escalation of hostilities ...
```

The first two lines are the **section label**: the document's title (from the
PDF's metadata) and the section path from M2's structure (bookmarks, then
headings by font size, then Docling's headers). Contents pages say "(contents
page)". The label is:

- **embedded** with the passage (so "executive summary" is in the vector of a
  paragraph that never says it, spec §5.2);
- **indexed for BM25** (the word index is over exactly what was embedded);
- **shown to the model** in the prompt, on the lines after `[n] chunk_id`.

It is stored in its own column (`label`), and the content hash covers label and
text, so a renamed section re-embeds its passages. The passage's `text` stays
the page's own words.

### Semantic passages

For every block, the sentences are embedded (nomic-embed-text, cached on disk
by sentence and model digest) and each pair of neighbours gets a cosine
similarity. Where the distance (1 - similarity) is above the 90th percentile of
all distances in the document, a passage ends. Pieces under 200 characters join
the neighbour they are more similar to; pieces over 1,000 are packed by
sentence. Tables stay whole.

On GEP p19 (the executive summary) the playground shows the curve: with the
threshold taken from that page alone, the **80th percentile cuts right before
"Risks to the outlook remain skewed to the downside"**; the 90th does not. An
index takes the threshold from the whole document instead.

## Measured

Passage counts (cut on 2026-10-15):

| Profile | iMac | GPO | GEP | Total |
|---|---|---|---|---|
| Baseline (`fixed-1000-150`) | 34 | 233 | 923 | 1,190 |
| PyMuPDF + cleaning (row 2, fixed) | 34 | 212 | 852 | 1,098 |
| PyMuPDF + structure-aware | 47 | 230 | 937 | 1,214 |
| Docling + structure-aware | 48 | 275 | 1,150 | 1,473 |
| PyMuPDF + word-boundary fix | 34 | 212 | 850 | 1,096 |
| PyMuPDF + semantic (8,248 sentences embedded, 488 s) | 52 | 251 | 1,005 | 1,308 |

Retrieval and answers for each chunker are in [`docs/ladder.md`](../ladder.md)
(row 3). In short: structure-aware passages with section labels win (A6 rank
45 → 1-2, Docling's 20 · 1 · 4 is the first scorecard to beat the baseline on
every count); the word-boundary fix alone is hygiene, not a retrieval gain;
semantic chunking is the weakest, because one document-wide threshold misses
the cut the playground finds on GEP p19.
