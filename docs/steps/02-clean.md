# 2. Clean

**What it does:** repairs what extraction broke, in a fixed order, keeping the
raw text beside the clean text, and records every change so the page inspector
can show it.

## Baseline (M1)

`clean.clean()`: one rule, Unicode **NFKC**. Cleaner id `nfkc`. It stays the
baseline's cleaner, unchanged, so ladder row 1 keeps its meaning.

## The full order (M2): cleaner `clean-v1`

`clean.clean_pdf()` for PyMuPDF, `parse_docling.clean_docling()` for Docling.
The order is the build spec's (§4), and it matters: each step needs what the
one before left.

| Step | Rule | Defect | How | GEP | GPO | iMac |
|---|---|---|---|---|---|---|
| 0 | Hidden text dropped (the guard) | #13 | The page drawn with and without its text: a span whose removal changes no pixel was never visible (`13-safety.md`) | 103 chars | 141 chars (37 spans) | 0 |
| 1 | NFKC | #4 | Unicode compatibility forms: `ﬁ` → `fi` | 1,507 ligatures | 1 ligature (269 characters in all) | 0 |
| 2 | "Th" | #7 | The font's own glyph name (`T_h`) restores what the text layer cut short | 325 | 0 | 0 |
| 3 | Footnote markers | #9 | A raised (superscript), smaller run of digits right after a visible character | 101 | 0 | 27 |
| 4 | Letter spacing | #11 | Inside "A N N U A L", drop the spaces PyMuPDF invented (`synthetic`) and keep the ones the PDF holds | 1 | 34 | 0 |
| 5 | Headers and footers | #2, #5, #12 | Margin lines (top or bottom 10%) whose text repeats on 3+ pages (digits and spaces ignored), or that are only a page number. The printed page number is read first | 591 lines on 174 pages | 147 lines on 75 pages | 21 lines on 11 pages |
| 6 | Dot leaders | #10 | Four or more dots collapse to a space; a page with 5+ such lines is a contents page | 194 lines | 37 lines | 0 |
| 7 | Line-break hyphens | #3 | The document as its own dictionary (below) | 710 breaks: 523 joined, 180 kept | 12: 2 joined, 9 kept | 4 kept |
| 8 | Whitespace | | Runs of spaces collapse; lines of one block join with a space, blocks with a line break. Last, because 5 and 7 need the line breaks | | | |

Text size after cleaning: GEP 725,641 → 679,846 characters, GPO 177,210 →
160,112, iMac 25,573 → 24,807.

**Why "repeats" and "in the margin" both.** Repetition alone is wrong: GEP's
reference lists repeat "Washington, DC: World Bank." in the body, GPO's tables
repeat "Workers' compensation (Note 9)". Position alone is wrong too: a heading
can sit near the top.

**The printed page number** is a number in the page's furniture whose distance
from the PDF page number repeats on neighbouring pages. "CHAPTER 1 7" holds two
numbers; only 7 keeps the same offset (22) as the pages around it. GPO has two
numberings (narrative = PDF − 2, financial = PDF − 31) and both come out right;
GEP's front matter is in roman numerals.

### Hyphens: the document is the dictionary

A line ending in "produc-" followed by "tion": join or keep? In order:

1. **The hyphenated pair appears elsewhere** in the document ("all-time", or
   inside a longer compound: "debt-to-GDP" attests "debt-to") → keep.
2. **The joined word appears elsewhere** ("production") → join.
3. **A longer compound continues** ("back-of-the-" / "envelope") → keep.
4. **A capital or a digit** ("Vance-" / "Cooks", "1,242-" / "page", "U-" /
   "shaped") → keep.
5. **Both halves are words the document uses**, at least 3 letters each
   ("fear-" / "driven", but not "in-" / "stalled") → keep.
6. Otherwise → join.

The halves of a split word never count as words of the document (they would
"prove" that "sugges" and "tions" are words). A line followed by a figure
caption ("invest-" then "FIGURE 1.1") is left alone: in reading order the word
continues after the figure.

**The target, measured:** an independent oracle rebuilt from the raw text (the
FINDINGS "quick check": the hyphenated form appears elsewhere) finds 140
compound line breaks in GEP (FINDINGS counted 127 with a slightly different
tokenizer). **All 140 keep their hyphen** (`tests/test_clean.py`). In GEP,
rules 1-2 decide 626 of the 710 breaks (139 kept, 487 joined). Rules 3-6 decide
the 77 the document cannot settle (11 chains, 11 capitals or digits, 19 pairs of
words, 36 joined by default): the same 77 the oracle calls undecided. These are
the uncertain ones. A read-through finds mistakes both ways: "customer-support"
and "fear-driven" are joined because "customer" and "fear" appear nowhere else,
while prefix splits like "in-stalled" are rightly joined. 2 breaks fall at a
page end (a passage never crosses pages, so they stay split), and 3 are
interrupted by a caption.

### Docling's text through the same rules

Docling already expands ligatures and reads the `T_h` glyph, and moves headers
and footers into its own furniture layer. What still applies:

- **Hidden text:** an item mostly inside the guard's hidden boxes is dropped
  (GPO: 296 characters).
- **Compounds:** Docling rejoins every line-break hyphen and so merges real
  compounds ("conflictrelated", "smallAI", "shortterm"). Where the PyMuPDF path
  kept a compound at a line break on the same page and its merged spelling never
  occurs in PyMuPDF's cleaned text, the hyphen goes back: **162 restored** in
  GEP, 9 in GPO, 3 in iMac.
- **Dot leaders** and **whitespace**, as above.

Not fixed in Docling's text, and shown in the inspector: letter-spaced display
text split into fragments ("Apr i l 14"), and footnote numbers left as separate
words ("recycled content 1").

## Every change is a mark

`clean.Marked` keeps text and marks together: each mark is the span a rule
produced, the rule, and **what the extractor had there** (`before`). Later edits
keep earlier marks aligned; an edit over an earlier mark absorbs it and rebuilds
`before` from the original, so a rejoined "inflation" still quotes "inﬂa-⏎tion".
The raw text carries its own marks (where each defect sits). The page inspector
paints both.

## Cost

The PyMuPDF path takes 25 s for GEP, 11 s for GPO and 5 s for iMac; nearly all
of it is the guard drawing every page twice. The result is cached per file,
parser, cleaner and the cleaning code's hash (`data/cache/clean/`).
