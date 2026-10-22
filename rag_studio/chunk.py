"""Chunk: cut each page's clean text into passages small enough to search and quote.

The baseline is the hand-built fixed-size chunker, reproduced exactly so that
it can be measured as the "before" row:

* 1,000 characters with 150 of overlap, never crossing a page;
* the end is moved back to the last space, so a passage never ends mid-word;
* near-empty pages are skipped;
* `chunk_id = "{doc_id}:p{page}:c{i}"`.

It also keeps the baseline's known defect on purpose: the next passage starts
at `end - overlap`, which is not moved to a space, so passages often start
mid-word ("n the upside"). M3 adds the fixed version (`snap_start`) as a
separate chunker and the scorecard measures the difference.

M3 adds two chunkers that read the cleaned page's structure:

* **structure**: paragraphs (and Docling's tables) packed up to `size`
  characters, never across a section, a contents page or reference list, or a
  table's edge. A paragraph longer than `size` is cut between sentences.
* **semantic**: the same sections, cut where two neighbouring sentences stop
  being about the same thing (their embeddings' cosine similarity drops).

Both can put a **section label** in front of every passage ("Document: ... /
Section: ..."). The label is embedded with the passage, indexed for BM25 and
shown to the model, so a passage that never says "executive summary" still
carries it (spec §5.2). The passage's own text stays an exact slice of the
page, so the page preview can still outline it.

`playground()` also offers recursive and sentence chunking, for the How it
works page only: they are taught, not indexed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from rag_studio.parse import Page

if TYPE_CHECKING:
    from rag_studio.clean import CleanPage

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class ChunkerSpec:
    id: str
    size: int
    overlap: int
    snap_start: bool      # move each start forward to a word boundary (M3 fix)
    method: str = "fixed"           # fixed | structure | semantic
    labels: bool = False            # put the section label in front of each passage
    percentile: float | None = None  # semantic: cut above this percentile of distances
    min_chars: int = 0              # semantic: merge pieces shorter than this

    def describe(self) -> dict:
        # The M1-M2 keys only, unless a later field is set: a profile's
        # recorded configuration must not change for builds that already exist.
        d: dict[str, Any] = {"id": self.id, "size": self.size, "overlap": self.overlap,
                             "snap_start": self.snap_start}
        if self.method != "fixed":
            d["method"] = self.method
        if self.labels:
            d["labels"] = LABEL_FORMAT
        if self.percentile is not None:
            d["percentile"] = self.percentile
        if self.min_chars:
            d["min_chars"] = self.min_chars
        return d


BASELINE = ChunkerSpec("fixed-1000-150", size=1000, overlap=150, snap_start=False)
FIXED_SNAP = ChunkerSpec("fixed-1000-150-snap", size=1000, overlap=150, snap_start=True)
STRUCTURE = ChunkerSpec("structure-1000", size=1000, overlap=0, snap_start=True,
                        method="structure", labels=True)
SEMANTIC = ChunkerSpec("semantic-1000-p90", size=1000, overlap=0, snap_start=True,
                       method="semantic", labels=True, percentile=90.0, min_chars=200)

# Recorded with every labelled build: changing it means re-embedding.
LABEL_FORMAT = "Document: {title}\\nSection: {section}"


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    char_start: int       # offsets into the page's clean text, for highlighting
    char_end: int
    label: str | None = None                 # the section label, when the chunker adds one
    meta: dict[str, Any] = field(default_factory=dict)   # section_path, kind, type (structure chunkers)
    context: str | None = None               # a line that places the passage in its document (M5)

    @property
    def embed_text(self) -> str:
        """Exactly what is embedded and indexed for BM25: the section label, the
        context line (when the profile has them), then the passage. The answer
        model is shown the passage (`text`) with its source, **not** this: the
        context line only helps search find the passage (row 7 is a
        retrieval-only change)."""
        parts = [x for x in (self.label, self.context) if x]
        return "\n".join([*parts, self.text])


def section_label(title: str | None, path: Sequence[str] | None, kind: str = "body") -> str:
    """"Document: Global Economic Prospects -- June 2026\\nSection: Executive
    Summary". Contents pages and reference lists say so, since that is what a
    question about them would ask for."""
    section = " > ".join(path or []) or "-"
    if kind == "toc":
        section = f"{section} (contents page)" if path else "Contents page"
    return LABEL_FORMAT.replace("\\n", "\n").format(title=" ".join((title or "-").split()), section=section)


# --------------------------------------------------------------------------- fixed-size

def split_fixed(text: str, size: int, overlap: int, snap_start: bool = False) -> list[tuple[int, int]]:
    """Character spans of each piece, after stripping surrounding whitespace.

    The loop is the baseline's `chunk_text`, line for line; only the return
    value changed, from strings to spans, so the page preview can find a
    passage on its page. `snap_start` is the M3 fix: the next start moves
    forward to the start of a word, so no passage begins mid-word.
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
        if snap_start:
            start = _word_start(text, start, end)
    return spans


def _word_start(text: str, i: int, limit: int) -> int:
    """`i` if a word starts there, else the start of the next word (not past
    `limit`, the previous piece's end, so the loop always moves on)."""
    if i == 0 or text[i - 1].isspace() or text[i].isspace():
        return i
    j = i
    while j < limit and not text[j].isspace():
        j += 1
    return j if j < limit else i


def chunk_pages(pages: Iterable[Page], spec: ChunkerSpec = BASELINE) -> list[Passage]:
    """Fixed-size passages from parsed or cleaned pages (M1-M2 profiles)."""
    if spec.method != "fixed":
        raise ValueError(f"The {spec.method} chunker needs the cleaned document: use chunk_document().")
    passages = []
    for page in pages:
        if page.is_near_empty:
            continue
        for i, (a, b) in enumerate(split_fixed(page.text, spec.size, spec.overlap, spec.snap_start)):
            passages.append(Passage(
                chunk_id=f"{page.doc_id}:p{page.page}:c{i}", doc_id=page.doc_id,
                page=page.page, text=page.text[a:b], char_start=a, char_end=b,
            ))
    return passages


# --------------------------------------------------------------------------- units and blocks

@dataclass
class Unit:
    """A paragraph, a heading or a table on one page: the smallest piece a
    structure-aware chunker moves around."""
    start: int
    end: int
    role: str                   # text | heading | table
    path: tuple[str, ...]
    kind: str                   # body | toc | references


_TABLE_LABELS = ("table", "document_index")
_HEADING_LABELS = ("section_header", "title", "page_header")


def _trimmed(text: str, a: int, b: int) -> tuple[int, int] | None:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return (a, b) if b > a else None


def units(page: "CleanPage") -> list[Unit]:
    """The page as paragraphs, headings and tables, in reading order.

    Docling names its items, so a table is one unit. PyMuPDF's cleaned text has
    one paragraph per line (cleaning joined each paragraph's lines), so each
    line is one unit. Headings come from the page's own heading list, for both.
    """
    text = page.text
    spans: list[tuple[int, int, str]] = []
    covered = 0
    items = sorted((i for i in page.items if i.get("start") is not None), key=lambda i: i["start"])
    for it in items:
        a, b = max(it["start"], covered), it["end"]
        if a < covered or b <= a:
            continue
        spans += _lines(text, covered, a)
        role = "table" if it["label"] in _TABLE_LABELS else "heading" if it["label"] in _HEADING_LABELS else "text"
        if role == "table":
            spans.append((a, b, role))
        else:
            spans += [(x, y, role) for x, y, _ in _lines(text, a, b)]
        covered = b
    spans += _lines(text, covered, len(text))
    heads = {h["start"] for h in page.headings}
    out: list[Unit] = []
    for a, b, role in spans:
        t = _trimmed(text, a, b)
        if t is None:
            continue
        if role == "text" and t[0] in heads:
            role = "heading"
        path = tuple(page.at(page.sections, t[0], "path") or ())
        kind = page.at(page.kinds, t[0], "kind") or "body"
        prev = out[-1] if out else None
        # A line break left inside a sentence (a column break: "controlling
        # inflation with\nsupporting activity") does not end the paragraph.
        if (prev and role == prev.role == "text" and (path, kind) == (prev.path, prev.kind)
                and text[prev.end - 1] not in ".!?:;" and text[t[0]].islower()):
            prev.end = t[1]
            continue
        out.append(Unit(t[0], t[1], role, path, kind))
    return out


def _lines(text: str, a: int, b: int) -> list[tuple[int, int, str]]:
    out, i = [], a
    while i < b:
        j = text.find("\n", i, b)
        j = b if j < 0 else j
        if j > i:
            out.append((i, j, "text"))
        i = j + 1
    return out


def blocks(us: list[Unit]) -> list[list[Unit]]:
    """Runs of units a passage may span: one section, one kind, and a table
    always on its own."""
    out: list[list[Unit]] = []
    for u in us:
        prev = out[-1][-1] if out else None
        if (prev is None or u.path != prev.path or u.kind != prev.kind
                or u.role == "table" or prev.role == "table"):
            out.append([u])
        else:
            out[-1].append(u)
    return out


# --------------------------------------------------------------------------- sentences

# A sentence ends at . ! ? (optionally closed by a quote or bracket) followed by
# whitespace and something that can start a sentence. "U.S. Government" splits
# too; a passage boundary there is rare and harmless.
_SENT_END = re.compile(r"(?<=[.!?])[\"”’)\]]?\s+(?=[\"“‘(\[]?[A-Z0-9•])")


def _paragraphs(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """Lines, rejoined where a break falls inside a sentence (the line ends
    without punctuation and the next starts lowercase)."""
    out: list[tuple[int, int]] = []
    for la, lb, _ in _lines(text, a, b):
        t = _trimmed(text, la, lb)
        if t is None:
            continue
        if out and text[out[-1][1] - 1] not in ".!?:;" and text[t[0]].islower():
            out[-1] = (out[-1][0], t[1])
        else:
            out.append(t)
    return out


def sentences(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """Sentence spans inside text[a:b], each trimmed. Line breaks also end a
    sentence: in cleaned text they separate paragraphs and list items."""
    out = []
    for la, lb in _paragraphs(text, a, b):
        start = la
        for m in _SENT_END.finditer(text, la, lb):
            if (t := _trimmed(text, start, m.start() + len(m.group(0).rstrip()))):
                out.append(t)
            start = m.end()
        if (t := _trimmed(text, start, lb)):
            out.append(t)
    return out


def _split_long(text: str, a: int, b: int, size: int, table: bool) -> list[tuple[int, int]]:
    """A unit longer than `size`, cut into pieces of at most `size`: tables
    between rows (lines), prose between sentences, and a sentence that is
    itself too long between words."""
    parts = [(x, y) for x, y, _ in _lines(text, a, b)] if table else sentences(text, a, b)
    fine: list[tuple[int, int]] = []
    for x, y in parts:
        if y - x <= size:
            fine.append((x, y))
        else:
            fine += [(x + p, x + q) for p, q in split_fixed(text[x:y], size, 0, snap_start=True)]
    return _pack(fine, size)


def _pack(spans: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Greedily join neighbouring spans while the joined span fits in `size`."""
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if out and b - out[-1][0] <= size:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


# --------------------------------------------------------------------------- structure-aware

# A piece shorter than this joins a neighbour on its page, even across a
# section edge: alone it is noise that matches too much (GPO's running line
# "U. S. Government Publishing Office | Financial Year In Review – FY 2018"
# survives cleaning on 10 financial pages) or too little ("Percent").
MIN_FRAGMENT = 100


def structure_spans(page: "CleanPage", size: int,
                    min_chars: int = MIN_FRAGMENT) -> list[tuple[int, int, Unit]]:
    """Passage spans on one page, each with the first unit of its block (for
    its section, kind and type). A piece that is only headings is dropped: its
    words are in the section label of every passage that follows it."""
    out: list[list[Any]] = []
    for block in blocks(units(page)):
        pieces: list[tuple[int, int, bool]] = []          # start, end, is a heading
        for u in block:
            if u.end - u.start > size:
                pieces += [(a, b, False) for a, b in _split_long(page.text, u.start, u.end, size,
                                                                  u.role == "table")]
            else:
                pieces.append((u.start, u.end, u.role == "heading"))
        packed: list[tuple[int, int, bool]] = []
        for a, b, head in pieces:
            if packed and block[0].role != "table" and b - packed[-1][0] <= size:
                packed[-1] = (packed[-1][0], b, packed[-1][2] and head)
            else:
                packed.append((a, b, head))
        out += [[a, b, block[0]] for a, b, head in packed if not head]
    # Fragments: join the next passage on the page, else the previous one,
    # going at most `min_chars` over `size`. Tables keep their edges.
    limit = size + min_chars
    i = 0
    while i < len(out):
        a, b, u = out[i]
        if b - a < min_chars and u.role != "table":
            nxt = out[i + 1] if i + 1 < len(out) else None
            prv = out[i - 1] if i > 0 else None
            if nxt and nxt[2].role != "table" and nxt[1] - a <= limit:
                nxt[0] = a
                del out[i]
                continue
            if prv and prv[2].role != "table" and b - prv[0] <= limit:
                prv[1] = b
                del out[i]
                continue
        i += 1
    return [(a, b, u) for a, b, u in out]


# --------------------------------------------------------------------------- semantic

def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    xs = sorted(values)
    if not xs:
        return math.inf
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class SemanticBlock:
    unit: Unit
    sents: list[tuple[int, int]]
    sims: list[float]           # sims[i]: similarity of sentence i and i + 1


def semantic_blocks(page: "CleanPage", embed: EmbedFn) -> list[SemanticBlock]:
    """Every non-heading block on a page as sentences with their neighbour
    similarities. Tables are kept whole (no similarities)."""
    out = []
    for block in blocks(units(page)):
        if all(u.role == "heading" for u in block):
            continue
        a, b = block[0].start, block[-1].end
        if block[0].role == "table":
            out.append(SemanticBlock(block[0], [(a, b)], []))
            continue
        sents = sentences(page.text, a, b)
        vecs = embed([page.text[x:y] for x, y in sents]) if len(sents) > 1 else []
        sims = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        out.append(SemanticBlock(block[0], sents, sims))
    return out


def semantic_cuts(sb: SemanticBlock, threshold: float) -> list[int]:
    """Indices i where a passage ends after sentence i: the neighbours'
    distance (1 - similarity) is above the threshold."""
    return [i for i, s in enumerate(sb.sims) if 1 - s > threshold]


def semantic_spans(sb: SemanticBlock, threshold: float, size: int, min_chars: int) -> list[tuple[int, int]]:
    if not sb.sims:                                           # a table, or one sentence
        a, b = sb.sents[0][0], sb.sents[-1][1]
        return [(a, b)] if b - a <= size else [(x, y) for x, y in _pack(sb.sents, size)]
    cuts = set(semantic_cuts(sb, threshold))
    groups: list[list[int]] = [[0]]
    for i in range(1, len(sb.sents)):
        if i - 1 in cuts:
            groups.append([i])
        else:
            groups[-1].append(i)
    # A group shorter than min_chars joins the neighbour it is more similar to.
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for gi, g in enumerate(groups):
            a, b = sb.sents[g[0]][0], sb.sents[g[-1]][1]
            if b - a >= min_chars:
                continue
            left = sb.sims[g[0] - 1] if gi > 0 else -2.0
            right = sb.sims[g[-1]] if gi < len(groups) - 1 else -2.0
            if left >= right:
                groups[gi - 1] += g
            else:
                groups[gi + 1] = g + groups[gi + 1]
            del groups[gi]
            merged = True
            break
    out: list[tuple[int, int]] = []
    for g in groups:
        span = [sb.sents[i] for i in g]
        a, b = span[0][0], span[-1][1]
        out += [(a, b)] if b - a <= size else _pack(span, size)
    return out


# --------------------------------------------------------------------------- whole documents

def chunk_document(pages: Sequence["CleanPage"], spec: ChunkerSpec, *, title: str | None = None,
                   embed: EmbedFn | None = None) -> list[Passage]:
    """Passages from a cleaned document, for every chunker. Structure-aware
    and semantic passages know their section, kind and type, and carry a
    section label when the spec asks for one."""
    if spec.method == "fixed":
        return chunk_pages(pages, spec)  # type: ignore[arg-type]
    live = [p for p in pages if not p.is_near_empty]
    per_page: list[tuple["CleanPage", list[tuple[int, int, Unit]]]] = []
    if spec.method == "structure":
        per_page = [(p, structure_spans(p, spec.size)) for p in live]
    elif spec.method == "semantic":
        if embed is None:
            raise ValueError("Semantic chunking needs an embedding function.")
        sbs = [(p, semantic_blocks(p, embed)) for p in live]
        threshold = percentile([1 - s for _, bl in sbs for sb in bl for s in sb.sims],
                               spec.percentile or 90.0)
        per_page = [(p, [(a, b, sb.unit) for sb in bl
                         for a, b in semantic_spans(sb, threshold, spec.size, spec.min_chars)])
                    for p, bl in sbs]
    else:
        raise ValueError(f"Unknown chunker method {spec.method!r}.")
    out = []
    for page, spans in per_page:
        for i, (a, b, u) in enumerate(spans):
            out.append(Passage(
                chunk_id=f"{page.doc_id}:p{page.page}:c{i}", doc_id=page.doc_id, page=page.page,
                text=page.text[a:b], char_start=a, char_end=b,
                label=section_label(title, u.path, u.kind) if spec.labels else None,
                meta={"section_path": " > ".join(u.path) or None, "kind": u.kind,
                      "type": "table" if u.role == "table" else "text"},
            ))
    return out


# --------------------------------------------------------------------------- the playground (How it works)

PLAYGROUND_METHODS = ("fixed", "recursive", "sentence", "structure", "semantic")
_RECURSIVE_SEPS = ("\n\n", "\n", ". ", " ")


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """LangChain-style recursive splitting: try paragraph breaks, then line
    breaks, then sentence ends, then spaces, until every piece fits; then
    pack neighbours up to `size`, each new piece starting `overlap` back."""
    def split(a: int, b: int, seps: Sequence[str]) -> list[tuple[int, int]]:
        if b - a <= size or not seps:
            return [(a, b)] if not seps or b - a <= size else [(a + x, a + y) for x, y in split_fixed(text[a:b], size, 0, True)]
        sep, rest = seps[0], seps[1:]
        pieces, i = [], a
        while i < b:
            j = text.find(sep, i, b)
            j = b if j < 0 else j + len(sep)
            pieces.append((i, j))
            i = j
        if len(pieces) == 1:
            return split(a, b, rest)
        out = []
        for x, y in pieces:
            out += split(x, y, rest)
        return out

    fine = [t for x, y in split(0, len(text), _RECURSIVE_SEPS) if (t := _trimmed(text, x, y))]
    return _with_overlap(fine, size, overlap)


def _with_overlap(fine: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Pack small spans up to `size`; each new passage starts with the last
    small spans of the previous one that fit in `overlap` characters."""
    out: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []
    for s in fine:
        if cur and s[1] - cur[0][0] > size:
            out.append((cur[0][0], cur[-1][1]))
            keep: list[tuple[int, int]] = []
            for t in reversed(cur):
                if cur[-1][1] - t[0] > overlap:
                    break
                keep.insert(0, t)
            cur = [t for t in keep if s[1] - t[0] <= size]
        cur.append(s)
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return out


def sentence_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Whole sentences packed up to `size`, overlapping by whole sentences."""
    return _with_overlap(sentences(text, 0, len(text)), size, overlap)


def playground(page: "CleanPage", method: str, size: int, overlap: int, *,
               embed: EmbedFn | None = None, percentile_p: float = 90.0) -> dict[str, Any]:
    """One page cut by any chunker, for the How it works chunk playground.
    The semantic result also returns the similarity curve and the threshold;
    the threshold comes from this page alone (an index uses the whole document)."""
    text = page.text
    if method == "fixed":
        return {"spans": split_fixed(text, size, overlap, snap_start=True), "baseline_spans": split_fixed(text, size, overlap)}
    if method == "recursive":
        return {"spans": recursive_spans(text, size, overlap)}
    if method == "sentence":
        return {"spans": sentence_spans(text, size, overlap)}
    if method == "structure":
        return {"spans": [(a, b) for a, b, _ in structure_spans(page, size)],
                "sections": [{"start": a, "end": b, "path": list(u.path), "kind": u.kind, "role": u.role}
                             for a, b, u in structure_spans(page, size)]}
    if method == "semantic":
        if embed is None:
            raise ValueError("Semantic chunking needs an embedding function.")
        sbs = semantic_blocks(page, embed)
        dists = [1 - s for sb in sbs for s in sb.sims]
        threshold = percentile(dists, percentile_p)
        curve = [{"after": sb.sents[i][1], "next": sb.sents[i + 1][0], "similarity": round(s, 4),
                  "cut": 1 - s > threshold}
                 for sb in sbs for i, s in enumerate(sb.sims)]
        spans = [s for sb in sbs for s in semantic_spans(sb, threshold, size, 0)]
        return {"spans": spans, "curve": curve, "threshold": round(1 - threshold, 4),
                "sentences": [s for sb in sbs for s in sb.sents]}
    raise ValueError(f"Unknown method {method!r}. Known: {', '.join(PLAYGROUND_METHODS)}.")
