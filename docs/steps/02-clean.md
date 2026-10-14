# 2. Clean

**What it does:** repairs what extraction broke, in a fixed order, keeping the
raw text beside the clean text.

## Baseline (M1)

`rag_studio/clean.py`: one rule, Unicode **NFKC**. It turns the ligature
characters PyMuPDF returns (`ﬁ ﬂ ﬀ ﬃ ﬄ`; GEP has 1,507) into plain letters, so
"financial" can match "ﬁnancial". Cleaner id: `nfkc`.

## Next (M2)

The full order from spec §4, each rule counted before and after:
NFKC → the broken "Th" (15 words, 325 times in GEP) → running headers and
footers (saving the printed page number first) → dot leaders → line-break
hyphens (708 lines, while 127 real compounds survive) → whitespace last.
