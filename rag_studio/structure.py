"""Structure: where each passage sits in its document.

A passage that knows its section can be labelled ("Executive Summary"),
filtered, and cited better. The three PDFs show their structure three ways
(defect #15), so section paths come from, in order of trust:

1. **Bookmarks.** GEP has 18 (`Ch. 1 Global Outlook` → p23); GPO's financial
   half has 56 (its first one points to page -1 and is dropped).
2. **Headings, by font size**, where a PDF has no bookmarks (GPO's narrative
   half, the iMac report) and inside a bookmark's pages: a short line set
   clearly larger than the body text.
3. **Docling's section headers**, for the Docling parser.

Two kinds of page content are **tagged**, not removed:

* `toc`: contents pages (many dot-leader lines). They match every query and
  answer none.
* `references`: a reference list, from a "References" heading to the next
  section. GEP's Chapter 3 bibliography outranked the chapter's own summary
  for "What does Chapter 3 say about sovereign debt?" (spec §5.3).

M2 records these; M3 uses them (section labels, a per-section cap).

**Where a box ends (M3).** GEP bookmarks where each box starts ("Box 3.1 The
costs of hidden debt" → p116) but not where it ends, so every page after a box
used to stay inside it until the next bookmark: Chapter 3's own references
read "Box 3.1 > References". Every page of a GEP box prints the box's number in
capitals ("BOX 3.1 The costs of hidden debt (continued)"), and the chapter's
pages never do (they cite "Box 3.1"). So a box ends at the first page that does
not name it, and the section returns to the chapter heading that was current
when the box began. Once passages carry section labels (M3), a wrong section
is a wrong fact in every passage after the box.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_studio.clean import CleanPage, Line

HEADING_RATIO = 1.15      # a heading is set at least 15% larger than the body text
HEADING_MAX_CHARS = 120
HEADING_MAX_WORDS = 12
_REFERENCES = re.compile(r"^(?:references|bibliography|works cited|reference list)$", re.I)
# GEP writes its regional sub-bookmarks as " -- South Asia", sometimes after a
# line break inside the title ("Middle East, North Africa,\r\n -- Afghanistan…").
_CHILD = re.compile(r"(?:^|[\r\n])\s*--\s*")
_NUMBERED = re.compile(r"^(?:Box|Table|Figure|Annex)\s+(\d+)\.\d", re.I)
_CHAPTER = re.compile(r"^(?:Ch\.|Chapter)\s*(\d+)\b", re.I)
_BOX = re.compile(r"^Box\s+(\d+\.\d+)\b", re.I)


def body_size(pages: list[list["Line"]]) -> float:
    """The font size most body characters use, across the document."""
    sizes: Counter[float] = Counter()
    for lines in pages:
        for ln in lines:
            if ln.role == "body":
                sizes[round(ln.size, 1)] += len(ln.text.text.strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _large(ln: "Line", body: float) -> bool:
    return ln.role == "body" and body > 0 and ln.size >= body * HEADING_RATIO and bool(ln.text.text.strip())


def _reads_as_heading(t: str) -> bool:
    """Short, mostly letters, and not a sentence. Large type that ends like a
    sentence is a call-out (the iMac report's "iMac contains more than 15
    percent recycled content."), and large digits are chart labels ("346 kg")."""
    letters = sum(c.isalpha() for c in t)
    visible = sum(not c.isspace() for c in t)
    return (2 <= len(t) <= HEADING_MAX_CHARS and letters >= 2 and letters >= 0.6 * visible
            and len(t.split()) <= HEADING_MAX_WORDS and not t.endswith((".", "!", "?", ",", ";")))


def headings(lines: list["Line"], starts: list[int], body: float) -> list[dict[str, Any]]:
    """Headings on one page, with where each starts in the page text. The
    large lines of one block are judged together: a title set on two lines is
    one heading, and a call-out sentence spread over three lines is none."""
    runs: list[list[int]] = []
    for i, ln in enumerate(lines):
        if starts[i] < 0 or not _large(ln, body):
            continue
        last = runs[-1][-1] if runs else None
        if (last is not None and lines[last].block == ln.block
                and abs(lines[last].size - ln.size) < 0.5):
            runs[-1].append(i)
        else:
            runs.append([i])
    out: list[dict[str, Any]] = []
    for run in runs:
        t = " ".join(" ".join(lines[i].text.text.split()) for i in run)
        if _reads_as_heading(t):
            out.append({"start": starts[run[0]], "text": t, "size": round(lines[run[0]].size, 1)})
    return out


def outline(bookmarks: list[tuple[int, str, int]]) -> list[dict[str, Any]]:
    """Bookmarks as section paths, in page order. A bookmark nests under an
    earlier one when its level is deeper, when it is written as a sub-entry
    (GEP's "-- South Asia" belongs to "Ch. 2 Regional Outlooks"), or when its
    number says so ("Box 3.1" belongs to "Ch. 3")."""
    if not bookmarks:
        return []
    top = min(lv for lv, _, _ in bookmarks)
    stack: list[tuple[int, str]] = []
    parent: tuple[int, str] | None = None       # the last entry that is not a sub-entry
    out = []
    for level, raw, page in bookmarks:
        title = " ".join(_CHILD.sub(" ", raw).split())
        level = level - top + 1
        if _CHILD.search(raw) and parent is not None:
            level = parent[0] + 1
            while stack and stack[-1] != parent:
                stack.pop()
        else:
            num = _NUMBERED.match(title)
            chapter = next((e for e in stack if num and _chapter_number(e[1]) == num.group(1)), None)
            if chapter is not None:
                level = chapter[0] + 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            if chapter is None:                 # a box is not the parent of the regions after it
                parent = (level, title)
        stack.append((level, title))
        out.append({"page": page, "path": [t for _, t in stack]})
    return sorted(out, key=lambda e: e["page"])


def _chapter_number(title: str) -> str | None:
    m = _CHAPTER.match(title)
    return m.group(1) if m else None


def kinds(page: "CleanPage", toc: bool) -> list[dict[str, Any]]:
    """A first pass per page; `assign_sections` carries references across pages."""
    return [{"start": 0, "kind": "toc" if toc else "body"}]


def assign_sections(pages: list["CleanPage"], entries: list[dict[str, Any]]) -> None:
    """Walk the document in order and give every page its section path and
    kind change points. A heading names the section until the next heading or
    bookmark; a reference list runs until the next heading or bookmark."""
    path: list[str] = []
    heading: str | None = None
    refs = False
    ref_size = 0.0
    e = 0
    box: tuple[str, list[str], str | None] | None = None   # marker, parent path, heading before it
    for page in pages:
        started = False
        while e < len(entries) and entries[e]["page"] <= page.page:
            new = entries[e]["path"]
            m = _BOX.match(new[-1]) if new else None
            box = (f"BOX {m.group(1)}", new[:-1], heading if path == new[:-1] else None) if m else None
            path = new
            heading, refs, started = None, False, True
            e += 1
        if box and not started and box[0] not in page.text and box[0] not in page.text_raw:
            path, heading, box = box[1], box[2], None
        toc = page.kinds and page.kinds[0]["kind"] == "toc"

        def current() -> list[str]:
            same = heading and path and heading.lower().startswith(path[-1].lower()[:20])
            return path + ([heading] if heading and not same else [])

        sections = [{"start": 0, "path": current()}]
        kind_points = [{"start": 0, "kind": "toc" if toc else ("references" if refs and not started else "body")}]
        for h in page.headings:
            if _REFERENCES.match(h["text"]):
                refs, ref_size = True, h["size"]
                heading = h["text"]
                kind_points.append({"start": h["start"], "kind": "references"})
            else:
                if refs and h["size"] >= ref_size:
                    refs = False
                    kind_points.append({"start": h["start"], "kind": "body"})
                heading = h["text"]
            sections.append({"start": h["start"], "path": current()})
        if toc:
            kind_points = [{"start": 0, "kind": "toc"}]
        page.sections = _dedupe(sections, "path")
        page.kinds = _dedupe(kind_points, "kind")


def _dedupe(points: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in points:
        if out and out[-1]["start"] == p["start"]:
            out[-1] = p
        elif not out or out[-1][key] != p[key]:
            out.append(p)
    return out
