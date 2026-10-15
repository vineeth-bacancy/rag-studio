"""Extract: read the text out of every PDF page.

The baseline parser is PyMuPDF's plain `page.get_text()`, one record per page,
exactly as the hand-built project did it. Docling (`parse_docling.py`) is a
second parser beside it, never a silent replacement.

Pages are numbered **1-based, as the PDF counts them**. That is the number
every citation shows. The printed page label ("74") is different: the cleaning
step reads it from the running header or footer before removing them.

**The layout reader** (M2) gives the cleaning step what plain text throws
away: every line with its position and font size, every character with its
box and whether PyMuPDF invented it (a space it inserted between letters), and
the order things were painted in, which the injection guard needs to find text
hidden under a later fill. Its lines rebuild `get_text()` exactly, so "raw" means
the same thing in both paths.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import pymupdf
import pymupdf.mupdf as mupdf

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


# --------------------------------------------------------------------------- layout (M2)

Box = tuple[float, float, float, float]

# The same flags `get_text()` uses, so the layout reader sees the same text.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT


@dataclass(frozen=True)
class Char:
    c: str
    box: Box
    origin: tuple[float, float]
    synthetic: bool      # PyMuPDF inserted it (a space between spread-out letters)
    span: int            # index into the line's spans


@dataclass(frozen=True)
class Span:
    size: float
    flags: int           # PyMuPDF span flags: 1 superscript, 2 italic, 16 bold
    font: str
    color: int           # sRGB as an integer


@dataclass
class RawLine:
    block: int
    box: Box
    chars: list[Char]
    spans: list[Span]

    @property
    def text(self) -> str:
        return "".join(c.c for c in self.chars)

    @property
    def size(self) -> float:
        """The font size most of the line's characters use."""
        counts: dict[float, int] = {}
        for c in self.chars:
            if not c.c.isspace():
                s = round(self.spans[c.span].size, 1)
                counts[s] = counts.get(s, 0) + 1
        return max(counts, key=lambda s: counts[s]) if counts else 0.0

    @property
    def bold(self) -> bool:
        return any(s.flags & 16 for s in self.spans) and all(
            self.spans[c.span].flags & 16 for c in self.chars if not c.c.isspace())


@dataclass
class PageLayout:
    doc_id: str
    page: int                   # 1-based
    width: float
    height: float
    lines: list[RawLine]
    # Characters whose glyph the PDF names as a ligature its text layer cannot
    # decode, keyed by `char_key`: the text the glyph really stands for.
    glyph_fixes: dict[tuple[float, float, str], str] = field(default_factory=dict)

    @property
    def text_raw(self) -> str:
        """Identical to `page.get_text()` (checked on all 294 pages)."""
        return "".join(line_text_raw(ln) for ln in self.lines)


def line_text_raw(line: RawLine) -> str:
    t = line.text
    return t if t.endswith("\n") else t + "\n"


def char_key(origin: tuple[float, float] | list[float], c: str) -> tuple[float, float, str]:
    """Identifies one drawn character across PyMuPDF's views of a page (the
    layout dictionary and the paint-order trace report the same origin)."""
    return (round(float(origin[0]), 2), round(float(origin[1]), 2), c)


def _read_lines(page: pymupdf.Page) -> list[RawLine]:
    raw: Any = page.get_text("rawdict", flags=_TEXT_FLAGS)
    lines = []
    for b in raw["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans, chars = [], []
            for s in ln["spans"]:
                spans.append(Span(size=float(s["size"]), flags=int(s["flags"]), font=str(s["font"]),
                                  color=int(s["color"])))
                for c in s["chars"]:
                    chars.append(Char(c=c["c"], box=tuple(c["bbox"]), origin=tuple(c["origin"]),
                                      synthetic=bool(c.get("synthetic")), span=len(spans) - 1))
            lines.append(RawLine(block=int(b["number"]), box=tuple(ln["bbox"]), chars=chars,
                                 spans=spans))
    return lines


# --------------------------------------------------------------------------- glyph names

# Adobe's glyph naming convention joins the parts of a ligature with "_"
# ("T_h", "f_f_i"). GEP's body font has a "T_h" glyph that its ToUnicode table
# forgets, so PyMuPDF falls back to the first letter and "The" comes out "Te"
# (325 times). The font itself still says what the glyph is.
_LIGATURE_NAME = re.compile(r"^[A-Za-z]+(?:_[A-Za-z]+)+$")


def _decode_glyph_name(name: str) -> str | None:
    name = name.split(".", 1)[0]          # "T_h.alt" is a variant of "T_h"
    if not _LIGATURE_NAME.match(name):
        return None
    out = []
    for part in name.split("_"):
        cp = mupdf.fz_unicode_from_glyph_name(part)
        if cp in (0, 0xFFFD):
            return None
        out.append(chr(cp))
    return "".join(out)


class GlyphNames:
    """Ligature glyphs per embedded font, read from the font program once per document."""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._fonts: dict[int, dict[int, str]] = {}

    def ligatures(self, xref: int) -> dict[int, str]:
        """glyph id -> the letters the glyph's name says it draws."""
        if xref not in self._fonts:
            found: dict[int, str] = {}
            try:
                buf = cast(Any, self.doc.extract_font(xref))[3]
                if buf:
                    font = pymupdf.Font(fontbuffer=buf)
                    for gid in range(font.glyph_count):
                        text = _decode_glyph_name(mupdf.fz_get_glyph_name2(font.this, gid))
                        if text:
                            found[gid] = text
            except Exception:  # noqa: BLE001 - a font we cannot read simply gets no repairs
                found = {}
            self._fonts[xref] = found
        return self._fonts[xref]

    def fixes(self, page: pymupdf.Page) -> dict[tuple[float, float, str], str]:
        """Characters on this page drawn with a ligature glyph whose text came
        out short: "T" for "T_h". Only a truncation is repaired, and only when
        every font on the page with that name agrees on the glyph."""
        by_name: dict[str, list[int]] = {}
        for f in page.get_fonts(full=True):
            base = str(f[3]).split("+", 1)[-1]
            by_name.setdefault(base, []).append(int(f[0]))
        out: dict[tuple[float, float, str], str] = {}
        for span in page.get_texttrace():
            xrefs = by_name.get(str(span["font"]), [])
            tables = [t for t in (self.ligatures(x) for x in xrefs) if t]
            if not tables:
                continue
            for ucs, gid, origin, *_ in span["chars"]:
                meanings = {t.get(gid) for t in tables}
                if len(meanings) != 1:
                    continue
                meaning = meanings.pop()
                if meaning is None:
                    continue
                got = unicodedata.normalize("NFKC", chr(ucs))
                if len(meaning) > len(got) and meaning.startswith(got):
                    out[char_key(origin, chr(ucs))] = meaning
        return out


def read_layout(path: Path) -> Iterator[tuple[pymupdf.Page, PageLayout]]:
    """Every page's lines, characters and glyph repairs, with the open page
    itself (the injection guard reads its paint order)."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        names = GlyphNames(doc)
        for p in doc:
            yield p, PageLayout(doc_id=path.stem, page=p.number + 1 if p.number is not None else 0,
                                width=float(p.rect.width), height=float(p.rect.height),
                                lines=_read_lines(p), glyph_fixes=names.fixes(p))


def bookmarks(path: Path) -> list[tuple[int, str, int]]:
    """The PDF's outline as (level, title, 1-based page). Entries pointing
    outside the document (GPO's first one points to page -1) are dropped."""
    with pymupdf.open(path) as doc:
        n = doc.page_count
        return [(int(lv), str(t).strip(), int(pg)) for lv, t, pg, *_ in doc.get_toc()
                if 1 <= int(pg) <= n]


_NORM = re.compile(r"[^0-9a-z]+")


def _norm_word(w: str) -> str:
    return _NORM.sub("", unicodedata.normalize("NFKC", w).lower())


def aligned_boxes(path: Path, page: int, text: str) -> dict[str, Any]:
    """Where a cleaned passage sits on its page. Cleaning changes the text
    (a rejoined "production", a repaired "The", a removed footer), so a
    character span no longer points into PyMuPDF's words. Instead the
    passage's words are aligned with the page's words in order: the matched
    stretch, gaps included, is the passage. `aligned` is the share of the
    passage's words found in place."""
    import difflib

    with pymupdf.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise IndexError(f"{path.stem} has {doc.page_count} pages, not {page}.")
        p = doc[page - 1]
        width, height = p.rect.width, p.rect.height
        words = p.get_text("words")
    page_tokens = [_norm_word(w[4]) for w in words]
    want = [t for t in (_norm_word(w) for w in text.split()) if t]
    if not want or not page_tokens:
        return {"width": width, "height": height, "boxes": [], "aligned": 0.0}
    sm = difflib.SequenceMatcher(None, page_tokens, want, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    matched = sum(b.size for b in blocks)
    if not blocks:
        return {"width": width, "height": height, "boxes": [], "aligned": 0.0}
    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size
    lines: dict[tuple[int, int], list[float]] = {}
    for w in words[first:last]:
        key = (w[5], w[6])
        box = lines.get(key)
        lines[key] = ([min(box[0], w[0]), min(box[1], w[1]), max(box[2], w[2]), max(box[3], w[3])]
                      if box else [w[0], w[1], w[2], w[3]])
    boxes = [{"x0": b[0] / width, "y0": b[1] / height, "x1": b[2] / width, "y1": b[3] / height}
             for b in lines.values()]
    return {"width": width, "height": height, "boxes": boxes, "aligned": round(matched / len(want), 3)}
