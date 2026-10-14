"""Chunk: cut each page's clean text into passages small enough to search and quote.

The baseline is the hand-built fixed-size chunker, reproduced exactly so that
it can be measured as the "before" row:

* 1,000 characters with 150 of overlap, never crossing a page;
* the end is moved back to the last space, so a passage never ends mid-word;
* near-empty pages are skipped;
* `chunk_id = "{doc_id}:p{page}:c{i}"`.

It also keeps the baseline's known defect on purpose: the next passage starts
at `end - overlap`, which is not moved to a space, so passages often start
mid-word ("n the upside"). M3 adds the fixed version as a separate chunker and
the scorecard measures the difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rag_studio.parse import Page


@dataclass(frozen=True)
class ChunkerSpec:
    id: str
    size: int
    overlap: int
    snap_start: bool      # move each start forward to a word boundary (M3 fix)

    def describe(self) -> dict:
        return {"id": self.id, "size": self.size, "overlap": self.overlap,
                "snap_start": self.snap_start}


BASELINE = ChunkerSpec("fixed-1000-150", size=1000, overlap=150, snap_start=False)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    char_start: int       # offsets into the page's clean text, for highlighting
    char_end: int


def split_fixed(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of each piece, after stripping surrounding whitespace.

    The loop is the baseline's `chunk_text`, line for line; only the return
    value changed, from strings to spans, so the page preview can find a
    passage on its page.
    """
    spans = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            trail = len(piece) - len(piece.rstrip())
            # `end` can run past the text on the last piece; the slice clamps
            # it, so the span must too.
            spans.append((start + lead, min(end, len(text)) - trail))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_pages(pages: Iterable[Page], spec: ChunkerSpec = BASELINE) -> list[Passage]:
    if spec.snap_start:
        raise NotImplementedError("The word-boundary chunker arrives in M3.")
    passages = []
    for page in pages:
        if page.is_near_empty:
            continue
        for i, (a, b) in enumerate(split_fixed(page.text, spec.size, spec.overlap)):
            passages.append(Passage(
                chunk_id=f"{page.doc_id}:p{page.page}:c{i}", doc_id=page.doc_id,
                page=page.page, text=page.text[a:b], char_start=a, char_end=b,
            ))
    return passages
