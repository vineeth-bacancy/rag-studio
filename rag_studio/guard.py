"""Guard: text from a PDF is data, never instructions.

A PDF can carry text a reader never sees, and text written to steer a model.
Both reach the prompt unless something stops them, so the guard has three
defences (spec §2.8):

1. **Hidden text is dropped** before any other step reads the page. The test
   is the rendered page itself: draw the page, draw it again with its text
   removed, and a span whose removal changes no pixel was never visible.
   Paint order then names the reason: painted over by a later fill or image
   (GPO p29 draws its footer, then a full-page colour fill on top), drawn in
   invisible mode, drawn in the colour of what is beneath it (white on white),
   or clipped away. Text under 2 points is dropped too: nobody can read it.
   Each dropped span is kept with its reason and box for the page inspector.

   Rules alone are not enough, and were measured: "white text" and "a later
   fill covers it" flagged all of iMac p6 (a transparent overlay) and white
   labels on coloured table headers, all of them visible. The pixels decide;
   the rules only explain.
2. **Instruction-like text is flagged**, in passages and in questions: "ignore
   the previous instructions", "you are now…", "reveal your system prompt".
   A flag never deletes text (a report may quote such a sentence); it is
   shown with the passage and recorded in the trace.
3. **Passages are wrapped as data** in the prompt (`prompt.py`, version v3).

The detectors are generic: nothing here names a document or a page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pymupdf

from rag_studio.parse import Box, char_key

# Below this a glyph is under 0.7 mm tall in print: nobody reads it.
TINY_PT = 2.0
# Colours closer than this (0-1 RGB, per channel) count as the same.
SAME_COLOUR = 0.06

REASONS = {
    "covered": "Painted over by a later fill",
    "invisible": "Drawn invisible",
    "same_colour": "Same colour as its background",
    "clipped": "Not drawn on the page",
    "off_page": "Placed outside the page",
    "tiny": "Too small to read",
}
# Resolution of the visibility test, and how much a channel must change for a
# pixel to count as changed (anti-aliasing noise stays well below it).
VISIBILITY_DPI = 96
PIXEL_CHANGE = 16
# Redaction modes for the text-removed copy (exported by PyMuPDF at runtime).
_KEEP_IMAGES: int = getattr(pymupdf, "PDF_REDACT_IMAGE_NONE")
_KEEP_GRAPHICS: int = getattr(pymupdf, "PDF_REDACT_LINE_ART_NONE")
_REMOVE_TEXT: int = getattr(pymupdf, "PDF_REDACT_TEXT_REMOVE")


@dataclass(frozen=True)
class Hidden:
    reason: str                      # a key of REASONS
    text: str
    box: Box
    keys: frozenset[tuple[float, float, str]]   # the characters, by `char_key`

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "label": REASONS[self.reason], "text": self.text,
                "box": list(self.box)}


def _rgb(color: Any, space: int) -> tuple[float, float, float] | None:
    """A texttrace colour (in its own colour space) as 0-1 RGB."""
    if color is None:
        return None
    c = [float(v) for v in color]
    if space == 1 and len(c) >= 1:
        return (c[0], c[0], c[0])
    if space == 3 and len(c) >= 3:
        return (c[0], c[1], c[2])
    if space == 4 and len(c) >= 4:
        k = c[3]
        return ((1 - c[0]) * (1 - k), (1 - c[1]) * (1 - k), (1 - c[2]) * (1 - k))
    return None


def _same(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> bool:
    return a is not None and b is not None and all(abs(x - y) <= SAME_COLOUR for x, y in zip(a, b))


def _is_rect(d: dict[str, Any]) -> bool:
    """Only rectangles count as covers: a curved or ring-shaped path whose
    bounding box happens to contain the text does not hide it."""
    items = d.get("items") or []
    if items and all(it[0] in ("re", "qu") for it in items):
        return all(it[0] == "re" or it[1].is_rectangular for it in items)
    if items and all(it[0] == "l" for it in items):
        xs = {round(p.x, 1) for it in items for p in it[1:3]}
        ys = {round(p.y, 1) for it in items for p in it[1:3]}
        return len(xs) == 2 and len(ys) == 2
    return False


def _covers(outer: pymupdf.Rect, inner: pymupdf.Rect, slack: float = 1.0) -> bool:
    return (outer.x0 - slack <= inner.x0 and outer.y0 - slack <= inner.y0
            and outer.x1 + slack >= inner.x1 and outer.y1 + slack >= inner.y1)


def _pixels(page: pymupdf.Page) -> Any:
    pix = page.get_pixmap(dpi=VISIBILITY_DPI, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n).astype(np.int16)


def _visibility(page: pymupdf.Page, scratch: pymupdf.Page) -> tuple[Any, list[pymupdf.Rect]]:
    """Per-pixel change when the page's text is removed, and the boxes of any
    text the removal could not reach (that text must count as visible).

    `scratch` is the same page opened a second time; its text is removed in
    place, so the caller's document is never touched."""
    before = _pixels(page)
    # fill=False: remove the text without painting the area white.
    scratch.add_redact_annot(scratch.rect, fill=cast(Any, False))
    scratch.apply_redactions(images=_KEEP_IMAGES, graphics=_KEEP_GRAPHICS, text=_REMOVE_TEXT)
    after = _pixels(scratch)
    left = [pymupdf.Rect(s["bbox"]) for s in scratch.get_texttrace()
            if "".join(chr(c[0]) for c in s["chars"]).strip()]
    return np.abs(before - after).max(axis=2), left


def hidden_text(page: pymupdf.Page, scratch: pymupdf.Page) -> list[Hidden]:
    """Every text span on the page that a reader cannot see, with the reason.
    `scratch` is the same page from a second open of the file (see `_visibility`)."""
    change, unreachable = _visibility(page, scratch)
    scale = VISIBILITY_DPI / 72
    h, w = change.shape

    page_box = page.rect

    def drawn(box: pymupdf.Rect) -> bool:
        box = box & page_box
        if box.is_empty:
            return False
        if any(box.intersects(r) for r in unreachable):
            return True
        x0, y0 = max(int(box.x0 * scale), 0), max(int(box.y0 * scale), 0)
        x1, y1 = min(int(box.x1 * scale) + 1, w), min(int(box.y1 * scale) + 1, h)
        return x1 > x0 and y1 > y0 and bool(change[y0:y1, x0:x1].max() > PIXEL_CHANGE)

    # Opaque covers in paint order: (seqno, rect, rgb or None for an image).
    covers: list[tuple[int, pymupdf.Rect, tuple[float, float, float] | None]] = []
    for d in page.get_drawings():
        fill = d.get("fill")
        opacity = d.get("fill_opacity")
        if fill is None or (opacity is not None and opacity < 0.99) or not _is_rect(d):
            continue
        covers.append((int(d["seqno"]), pymupdf.Rect(d["rect"]), tuple(float(v) for v in fill)))  # type: ignore[arg-type]
    # The paint log's index is the same sequence number the trace and the
    # drawings report (checked), so images join the same ordering.
    for i, (kind, box) in enumerate(page.get_bboxlog()):
        if kind == "fill-image":
            covers.append((i, pymupdf.Rect(box), None))

    out: list[Hidden] = []
    for s in page.get_texttrace():
        text = "".join(chr(c[0]) for c in s["chars"])
        if not text.strip():
            continue
        box = pymupdf.Rect(s["bbox"])
        reason = None
        if float(s["size"]) < TINY_PT:
            reason = "tiny"
        elif (box & page_box).is_empty:
            reason = "off_page"     # GPO p15 carries p14's figures left of its edge
        elif not drawn(box):
            beneath = [c for c in covers if c[0] < s["seqno"] and _covers(c[1], box)]
            # Nothing painted beneath means the white page; an image beneath
            # could be any colour, so it never counts as a match.
            bg = max(beneath, key=lambda c: c[0])[2] if beneath else (1.0, 1.0, 1.0)
            if s.get("type") == 3 or float(s.get("opacity", 1) or 0) == 0:
                reason = "invisible"
            elif any(seq > s["seqno"] and _covers(r, box) for seq, r, _ in covers):
                reason = "covered"
            elif _same(_rgb(s.get("color"), int(s.get("colorspace") or 0)), bg):
                reason = "same_colour"
            else:
                reason = "clipped"
        if reason:
            out.append(Hidden(reason=reason, text=text, box=(box.x0, box.y0, box.x1, box.y1),
                              keys=frozenset(char_key(c[2], chr(c[0])) for c in s["chars"])))
    return out


# --------------------------------------------------------------------------- instructions

# Phrases written to steer a language model rather than to inform a reader.
# Each one names the model's own rules or role, which report prose never does.
# Kept narrow on purpose: the three reports raise no flag (checked on every
# passage), and the crafted test PDF raises each one.
_RULE_WORDS = r"(?:instructions?|rules?|prompts?|directions?|guidelines?|guardrails?)"
INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?\b(?:previous|prior|above|earlier|"
        r"preceding|all|any|your|these|those|the)\b[^.\n]{0,30}?\b" + _RULE_WORDS + r"\b", re.I)),
    ("own_knowledge", re.compile(r"\b(?:answer|respond|reply)\b[^.\n]{0,30}\bfrom your own knowledge\b", re.I)),
    ("role", re.compile(r"\byou are now\b|\bfrom now on,? you\b|\bact as (?:an? )?(?:ai|assistant|chatbot|model)\b", re.I)),
    ("new_rules", re.compile(r"\b(?:new|updated|real|actual)\s+" + _RULE_WORDS + r"\s*:", re.I)),
    ("reveal", re.compile(r"\b(?:reveal|print|repeat|show)\b[^.\n]{0,20}\b(?:system prompt|your " + _RULE_WORDS + r")\b", re.I)),
    ("chat_markup", re.compile(r"<\|?(?:im_start|im_end|system|endoftext)\|?>|^\s*(?:system|assistant)\s*:", re.I | re.M)),
    ("ai_addressed", re.compile(r"\b(?:AI|assistant|language model|LLM|chatbot)s?\b[^.\n]{0,20}\b(?:must|should)\s+"
                                r"(?:ignore|say|answer|reply|respond|state|claim)\b", re.I)),
)

INSTRUCTION_LABELS = {
    "override": "Tells the reader to ignore its rules",
    "own_knowledge": "Asks for an answer from outside the documents",
    "role": "Tries to give the model a new role",
    "new_rules": "Announces new rules",
    "reveal": "Asks for the model's instructions",
    "chat_markup": "Contains chat-message markup",
    "ai_addressed": "Gives orders to an AI",
}


def instruction_flags(text: str) -> list[dict[str, Any]]:
    """Instruction-like phrases in a passage or question, with what matched."""
    out = []
    for kind, pattern in INSTRUCTION_PATTERNS:
        m = pattern.search(text)
        if m:
            out.append({"kind": kind, "label": INSTRUCTION_LABELS[kind], "match": m.group(0)[:120],
                        "start": m.start(), "end": m.end()})
    return out
