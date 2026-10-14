"""Extract: read the text out of every PDF page.

The baseline parser is PyMuPDF's plain `page.get_text()`, one record per page,
exactly as the hand-built project did it. Docling arrives in M2 as a second
parser beside this one, never as a silent replacement.

Pages are numbered **1-based, as the PDF counts them**. That is the number
every citation shows. The printed page label ("74") is different, and arrives
with the cleaning rules in M2.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from rag_studio import clean

PARSER_ID = "pymupdf"
PARSER_VERSION = pymupdf.VersionBind


@dataclass(frozen=True)
class Page:
    doc_id: str
    page: int            # 1-based PDF page
    text_raw: str        # exactly what the parser returned
    text: str            # after cleaning; passages are cut from this
    chars: int           # non-whitespace characters in text_raw

    @property
    def is_near_empty(self) -> bool:
        return self.chars <= NEAR_EMPTY_CHARS


# A page with this many non-whitespace characters or fewer holds no passage
# worth retrieving: covers, blank backs, section dividers. Measured on the
# three PDFs: GPO 11 pages, GEP 23, iMac none (build spec §4).
NEAR_EMPTY_CHARS = 200


@dataclass(frozen=True)
class Document:
    doc_id: str          # the file stem, everywhere
    path: Path
    sha256: str
    title: str | None    # from the PDF's own metadata; None when it has none
    pages: tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def non_ws_chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def parse_pdf(path: Path) -> Document:
    """Every page of one PDF, raw and cleaned, in page order."""
    path = Path(path).resolve()
    pages = []
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or None
        for p in doc:
            raw = str(p.get_text())
            pages.append(Page(doc_id=path.stem, page=p.number + 1, text_raw=raw,
                              text=clean.clean(raw), chars=non_ws_chars(raw)))
    return Document(doc_id=path.stem, path=path, sha256=file_sha256(path), title=title,
                    pages=tuple(pages))


def pdf_title(path: Path) -> str | None:
    with pymupdf.open(path) as doc:
        return (doc.metadata or {}).get("title") or None


# --------------------------------------------------------------------------- page geometry

def render_page_png(path: Path, page: int, dpi: int) -> bytes:
    """The page as a PNG, for the page preview."""
    with pymupdf.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise IndexError(f"{path.stem} has {doc.page_count} pages, not {page}.")
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")


_TOKEN = re.compile(r"\S+")


def passage_boxes(path: Path, page: int, char_start: int, char_end: int) -> dict[str, Any]:
    """Where a passage sits on its page, as rectangles in 0..1 page coordinates.

    A passage is a span of the page's clean text. PyMuPDF's word list and the
    page text hold the same words in the same order (checked on all 294 pages:
    `" ".join(words).split() == text.split()`), so the k-th whitespace token in
    the text is the k-th word on the page, with a known box. Words of one line
    merge into one rectangle.
    """
    with pymupdf.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise IndexError(f"{path.stem} has {doc.page_count} pages, not {page}.")
        p = doc[page - 1]
        width, height = p.rect.width, p.rect.height
        words = p.get_text("words")
        text = clean.clean(str(p.get_text()))
    spans = [m.span() for m in _TOKEN.finditer(text)]
    tokens = [clean.clean(w[4]) for w in words]
    split = text.split()
    if [text[a:b] for a, b in spans] != split or " ".join(tokens).split() != split:
        return {"width": width, "height": height, "boxes": [], "aligned": False}
    lines: dict[tuple[int, int], list[float]] = {}
    for (a, b), w in zip(spans, words):
        if b <= char_start or a >= char_end:
            continue
        key = (w[5], w[6])  # block, line
        box = lines.get(key)
        lines[key] = ([min(box[0], w[0]), min(box[1], w[1]), max(box[2], w[2]), max(box[3], w[3])]
                      if box else [w[0], w[1], w[2], w[3]])
    boxes = [{"x0": b[0] / width, "y0": b[1] / height, "x1": b[2] / width, "y1": b[3] / height}
             for b in lines.values()]
    return {"width": width, "height": height, "boxes": boxes, "aligned": True}
