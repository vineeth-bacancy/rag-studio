"""Clean: repair what extraction broke, keeping the raw text beside the result.

**The baseline** (`nfkc`, ladder row 1) applies one rule, Unicode NFKC, which
turns the ligature characters PyMuPDF returns ("ﬁ" "ﬂ" "ﬀ") into plain letters.
GEP alone has 1,507 of them, and a search for "financial" cannot match
"ﬁnancial". `clean()` is that rule, unchanged.

**The full order** (`clean-v1`, M2) follows the build spec §4, each rule
pointing back to a defect in `FINDINGS.md`:

    hidden text dropped (the guard, #13), before anything reads the page
    1 NFKC: ligatures and compatibility characters                       #4
    2 "Th": glyphs the text layer cannot decode, read from the font      #7
    3 footnote markers glued to words, by superscript position           #9
    4 letter spacing: the spaces PyMuPDF inserted between letters        #11
    5 running headers and footers, after saving the printed page number  #2, #5, #12
    6 dot leaders                                                         #10
    7 line-break hyphens, with the document as its own dictionary        #3
    8 whitespace, last: steps 5 and 7 need the line breaks

Rules 1-4 work on characters while each line is assembled; 5-8 need the whole
document (what repeats, which words exist). Every change is recorded as a
**mark**, so the page inspector can show what each rule did and where, and
every rule is counted. Nothing names a document or a word: the "Th" repair
reads the font's own glyph names, and the hyphen rule asks the document
whether it writes "all-time" or "alltime".
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from rag_studio.parse import PageLayout, RawLine

CLEANER_ID = "nfkc"          # the baseline: NFKC only
FULL_CLEANER_ID = "clean-v1"  # the full order (M2)


def clean(text: str) -> str:
    """The baseline cleaner: NFKC, nothing else."""
    return unicodedata.normalize("NFKC", text)


@dataclass(frozen=True)
class Rule:
    id: str
    defect: int | None      # number in FINDINGS.md / spec §4
    label: str
    explain: str


RULES: tuple[Rule, ...] = (
    Rule("hidden", 13, "Hidden text",
         "Text a reader cannot see: painted over, invisible, white on white, clipped or tiny. Dropped by the guard."),
    Rule("nfkc", 4, "Ligatures",
         "Ligature characters like \"ﬁ\" become plain letters (Unicode NFKC), so \"financial\" matches \"ﬁnancial\"."),
    Rule("th", 7, "\"Th\" glyphs",
         "The font draws \"Th\" as one glyph its text table forgets, so \"The\" came out \"Te\". The glyph's own name says \"T_h\"."),
    Rule("footnote", 9, "Footnote markers",
         "Raised footnote numbers glued to words (\"content1\") are removed by position, not by pattern."),
    Rule("spacing", 11, "Letter spacing",
         "Spread-out display text (\"A N N U A L\") loses the spaces the extractor invented between letters."),
    Rule("furniture", 2, "Headers and footers",
         "Lines that repeat in the page margins move out of the text into metadata, after the printed page number is saved."),
    Rule("leaders", 10, "Dot leaders",
         "Rows of dots in tables of contents are collapsed."),
    Rule("hyphen", 3, "Line-break hyphens",
         "A word split at a line end is rejoined when the document writes it joined, kept when it writes it hyphenated."),
    Rule("whitespace", None, "Whitespace",
         "Runs of spaces collapse to one, and lines of one paragraph join. Last, because the rules above need the line breaks."),
)
RULE_IDS = [r.id for r in RULES]


# --------------------------------------------------------------------------- marked text

@dataclass
class Mark:
    start: int
    end: int
    rule: str
    before: str


class Marked:
    """Text plus the marks that say what changed where. Edits keep marks
    aligned; an edit over an earlier mark absorbs it, and its `before` is
    rebuilt from the original text, so `before` always quotes the extractor."""

    __slots__ = ("text", "marks")

    def __init__(self, text: str = "", marks: list[Mark] | None = None):
        self.text = text
        self.marks = marks or []

    def append(self, s: str, rule: str | None = None, before: str | None = None) -> None:
        if rule is not None:
            self.marks.append(Mark(len(self.text), len(self.text) + len(s), rule, before or ""))
        self.text += s

    def original(self, start: int, end: int) -> str:
        """The text of [start, end) as it was before any marked edit."""
        out, pos = [], start
        for m in sorted(self.marks, key=lambda m: (m.start, m.end)):
            if m.start >= start and m.end <= end and not (m.start == m.end and m.start in (start, end)):
                out.append(self.text[pos:m.start])
                out.append(m.before)
                pos = m.end
        out.append(self.text[pos:end])
        return "".join(out)

    def replace(self, start: int, end: int, new: str, rule: str | None,
                before: str | None = None) -> None:
        """Replace [start, end) with `new`. With a rule, the result is marked;
        without one (whitespace) only the existing marks move."""
        inside = [m for m in self.marks if m.end > start and m.start < end]
        if inside and rule is not None:
            start = min(start, *(m.start for m in inside))
            end = max(end, *(m.end for m in inside))
            if before is None:
                before = self.original(start, end)
        elif before is None:
            before = self.text[start:end]
        delta = len(new) - (end - start)
        kept = []
        for m in self.marks:
            if rule is not None and m in inside:
                continue
            if m.start >= end and not (m.start == m.end == start == end):
                kept.append(Mark(m.start + delta, m.end + delta, m.rule, m.before))
            elif m.end <= start:
                kept.append(m)
            else:   # unmarked edit inside a mark: the mark stretches or shrinks
                kept.append(Mark(m.start, max(m.start, m.end + delta), m.rule, m.before))
        if rule is not None:
            kept.append(Mark(start, start + len(new), rule, before))
        self.marks = sorted(kept, key=lambda m: (m.start, m.end))
        self.text = self.text[:start] + new + self.text[end:]

    def sub(self, pattern: re.Pattern[str], repl: str, rule: str | None) -> int:
        """Regex replace, right to left so earlier offsets stay valid."""
        matches = list(pattern.finditer(self.text))
        for m in reversed(matches):
            self.replace(m.start(), m.end(), m.expand(repl), rule)
        return len(matches)

    @staticmethod
    def join(parts: Iterable["Marked"], seps: Iterable[str]) -> "Marked":
        out = Marked()
        for part, sep in zip(parts, [""] + list(seps)):
            out.text += sep
            base = len(out.text)
            out.marks += [Mark(m.start + base, m.end + base, m.rule, m.before) for m in part.marks]
            out.text += part.text
        return out


# --------------------------------------------------------------------------- lines

@dataclass
class Line:
    page: int
    block: int
    box: tuple[float, float, float, float]
    size: float
    bold: bool
    raw: str               # the line as extracted, without its line break
    raw_start: int         # where `raw` starts in the page's text_raw
    text: Marked           # the line as cleaning has it so far
    role: str = "body"     # body | furniture


@dataclass
class RawMark:
    start: int
    end: int
    rule: str


@dataclass
class CleanPage:
    doc_id: str
    page: int                       # 1-based PDF page: the citation number
    width: float
    height: float
    text_raw: str                   # what the parser returned
    text: str                       # after every rule
    marks: list[dict[str, Any]]     # in `text`: start, end, rule, before
    raw_marks: list[dict[str, Any]]  # in `text_raw`: start, end, rule
    label: str | None = None        # the printed page number, if the page prints one
    furniture: list[dict[str, Any]] = field(default_factory=list)   # text, box
    hidden: list[dict[str, Any]] = field(default_factory=list)      # guard.Hidden.as_dict()
    headings: list[dict[str, Any]] = field(default_factory=list)    # start, text
    kinds: list[dict[str, Any]] = field(default_factory=list)       # start, kind (body | toc | references)
    sections: list[dict[str, Any]] = field(default_factory=list)    # start, path
    items: list[dict[str, Any]] = field(default_factory=list)       # Docling: label, box, start, end

    @property
    def chars(self) -> int:
        return sum(1 for c in self.text if not c.isspace())

    @property
    def is_near_empty(self) -> bool:
        from rag_studio.parse import NEAR_EMPTY_CHARS
        return self.chars <= NEAR_EMPTY_CHARS

    def at(self, series: list[dict[str, Any]], offset: int, key: str) -> Any:
        """The value of a change-point series (kinds, sections) at `offset`."""
        value = None
        for point in series:
            if point["start"] > offset:
                break
            value = point[key]
        return value


@dataclass
class CleanDoc:
    doc_id: str
    sha256: str
    parser: str
    cleaner: str
    pages: list[CleanPage]
    counts: dict[str, Any]          # per rule, for the inspector and docs
    info: dict[str, Any]            # parser version, seconds, body font size, ...

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CleanDoc":
        pages = [CleanPage(**p) for p in d["pages"]]
        return CleanDoc(**{**d, "pages": pages})


# --------------------------------------------------------------------------- rules 1-4: characters

_MARKER = re.compile(r"^[0-9,*†‡]{1,4}$")
_SPACED_RUN = re.compile(r"(?<!\S)(?:\S )+\S(?!\S)")
_NFKC_CACHE: dict[str, str] = {}


def _nfkc(c: str) -> str:
    out = _NFKC_CACHE.get(c)
    if out is None:
        out = _NFKC_CACHE[c] = unicodedata.normalize("NFKC", c)
    return out


def build_line(rl: "RawLine", raw_start: int, hidden: set[tuple[float, float, str]],
               glyph_fixes: dict[tuple[float, float, str], str], page: int,
               raw_marks: list[RawMark], counts: Counter[str]) -> Line:
    """One extracted line through rules 1-4 (and the hidden-text drop)."""
    from rag_studio.parse import char_key

    chars = rl.chars[:-1] if rl.chars and rl.chars[-1].c == "\n" else rl.chars
    size = rl.size
    # Footnote markers: a run of raised, smaller digits right after a visible
    # character. "content¹" loses the 1; "CO2" (a subscript) and "1st" stay.
    drop_span: set[int] = set()
    for i, s in enumerate(rl.spans):
        if not (s.flags & 1) or s.size >= 0.85 * size:
            continue
        idx = [k for k, c in enumerate(chars) if c.span == i]
        text = "".join(chars[k].c for k in idx).strip()
        if idx and _MARKER.match(text) and idx[0] > 0 and not chars[idx[0] - 1].c.isspace():
            drop_span.add(i)

    out = Marked()
    pieces: list[tuple[int, int, bool]] = []   # (out start, out end, synthetic) per kept char
    k = 0
    while k < len(chars):
        ch = chars[k]
        raw_at = raw_start + k
        key = char_key(ch.origin, ch.c)
        if key in hidden:
            j = k
            while j < len(chars) and char_key(chars[j].origin, chars[j].c) in hidden:
                j += 1
            raw_marks.append(RawMark(raw_at, raw_start + j, "hidden"))
            counts["hidden"] += sum(1 for c in chars[k:j] if not c.c.isspace())
            k = j
            continue
        if ch.span in drop_span:
            j = k
            while j < len(chars) and chars[j].span == ch.span:
                j += 1
            marker = "".join(c.c for c in chars[k:j])
            out.append("", "footnote", marker)
            raw_marks.append(RawMark(raw_at, raw_start + j, "footnote"))
            counts["footnote"] += 1
            k = j
            continue
        start = len(out.text)
        fix = glyph_fixes.get(key)
        if fix is not None:
            # Some fonts map a ligature glyph to its first letter and then carry
            # the rest as text after it (GPO's "f_f_i" in "Trafficking"):
            # nothing is missing, so nothing is added.
            rest = fix[len(_nfkc(ch.c)):]
            if "".join(_nfkc(c.c) for c in chars[k + 1:k + 1 + len(rest)]) == rest:
                fix = None
        if fix is not None:
            out.append(fix, "th", ch.c)
            raw_marks.append(RawMark(raw_at, raw_at + 1, "th"))
            counts["th"] += 1
        else:
            n = _nfkc(ch.c)
            if n != ch.c and not ch.c.isspace():
                out.append(n, "nfkc", ch.c)
                raw_marks.append(RawMark(raw_at, raw_at + 1, "nfkc"))
                counts["nfkc"] += 1
                if "ﬀ" <= ch.c <= "ﬆ":
                    counts["ligatures"] += 1
            else:
                out.append(n)
        pieces.append((start, len(out.text), ch.synthetic and ch.c == " "))
        k += 1

    _unspace(out, pieces, raw_start, chars, raw_marks, counts)
    return Line(page=page, block=rl.block, box=rl.box, size=size, bold=rl.bold,
                raw="".join(c.c for c in chars), raw_start=raw_start, text=out)


def _unspace(out: Marked, pieces: list[tuple[int, int, bool]], raw_start: int,
             chars: list[Any], raw_marks: list[RawMark], counts: Counter[str]) -> None:
    """Rule 4: inside a run of single letters separated by single spaces
    ("A N N U A L R E P O R T"), drop the spaces PyMuPDF invented and keep the
    ones the PDF holds. Text whose spaces are all real is left alone."""
    synthetic_at = {start: syn for start, _end, syn in pieces}
    for m in reversed(list(_SPACED_RUN.finditer(out.text))):
        run = m.group(0)
        if len(run.replace(" ", "")) < 4:
            continue
        spaces = [m.start() + i for i, c in enumerate(run) if c == " "]
        invented = [p for p in spaces if synthetic_at.get(p)]
        if not invented:
            continue
        joined = "".join(c for i, c in enumerate(run) if m.start() + i not in set(invented))
        out.replace(m.start(), m.end(), joined, "spacing")
        counts["spacing"] += 1
    # The raw text is not changed by this rule; the runs are marked on it too.
    raw = "".join(c.c for c in chars)
    for m in _SPACED_RUN.finditer(raw):
        if len(m.group(0).replace(" ", "")) >= 4 and any(
                chars[m.start() + i].synthetic for i, c in enumerate(m.group(0)) if c == " "):
            raw_marks.append(RawMark(raw_start + m.start(), raw_start + m.end(), "spacing"))


# --------------------------------------------------------------------------- rule 5: furniture

MARGIN = 0.10          # top and bottom 10% of the page height
MIN_REPEATS = 3        # a margin line on at least this many pages is furniture
_DIGITS = re.compile(r"\d+")
_PAGE_NUMBER = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)
_ROMAN = re.compile(r"^[ivxlcdm]{1,7}$", re.I)


def _furniture_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    return _DIGITS.sub("#", "".join(t.split()))


def _in_margin(line: Line, height: float) -> bool:
    return line.box[3] <= MARGIN * height or line.box[1] >= (1 - MARGIN) * height


def find_furniture(pages: list[list[Line]], heights: list[float], counts: Counter[str]) -> None:
    """Mark running headers, footers and page numbers: margin lines whose text
    (digits ignored, spaces ignored) repeats on several pages, or that are only
    a page number. Repetition alone is wrong ("Washington, DC: World Bank."
    repeats in every reference list), so both are required."""
    seen: Counter[str] = Counter()
    for lines, h in zip(pages, heights):
        keys = {_furniture_key(ln.text.text) for ln in lines if _in_margin(ln, h) and ln.text.text.strip()}
        seen.update(keys)
    for lines, h in zip(pages, heights):
        for ln in lines:
            t = ln.text.text.strip()
            if not t or not _in_margin(ln, h):
                continue
            if seen[_furniture_key(t)] >= MIN_REPEATS or _PAGE_NUMBER.match(t):
                ln.role = "furniture"
                counts["furniture"] += 1


def _label_candidates(lines: list[Line]) -> list[str]:
    out = []
    for ln in lines:
        if ln.role != "furniture":
            continue
        t = ln.text.text.strip()
        out += _DIGITS.findall(t)
        if _ROMAN.match(t):
            out.append(t.lower())
    return out


def printed_labels(pages: list[list[Line]], numbers: list[int]) -> list[str | None]:
    """The page number printed on each page, read from its furniture before it
    is removed. A furniture line can hold several numbers ("CHAPTER 1  7",
    "2018 ANNUAL REPORT 26"); the one that keeps the same distance from the PDF
    page number as its neighbours is the page number. Roman numerals count
    when they are the only candidate."""
    cands = [_label_candidates(lines) for lines in pages]
    # Every numeric candidate proposes an offset (PDF page - printed number).
    # The page number's offset repeats on neighbouring pages; a chapter number
    # or a year does not.
    offsets = [{n - int(x) for x in c if x.isdigit()} for n, c in zip(numbers, cands)]
    out: list[str | None] = []
    for i, (n, c) in enumerate(zip(numbers, cands)):
        window: Counter[int] = Counter()
        for o in offsets[max(0, i - 6): i + 7]:
            window.update(o)
        nums = [x for x in c if x.isdigit()]
        best = [x for x in nums if window[n - int(x)] >= 2]
        pick = max(best, key=lambda x: window[n - int(x)]) if best else None
        if pick is None and not nums:
            romans = [x for x in c if not x.isdigit()]
            pick = romans[0] if len(romans) == 1 else None
        out.append(pick)
    return out


# --------------------------------------------------------------------------- rule 6: dot leaders

_LEADERS = re.compile(r"[ \t]*(?:\.[ \t]?){4,}[ \t]*")
TOC_MIN_LEADER_LINES = 5


def collapse_leaders(lines: list[Line], counts: Counter[str], raw_marks: list[RawMark]) -> int:
    """Rule 6. Returns how many lines on the page had leaders (a contents page
    has many)."""
    n = 0
    for ln in lines:
        if ln.role != "body":
            continue
        if ln.text.sub(_LEADERS, " ", "leaders"):
            n += 1
            counts["leaders"] += 1
            for m in _LEADERS.finditer(ln.raw):
                raw_marks.append(RawMark(ln.raw_start + m.start(), ln.raw_start + m.end(), "leaders"))
    return n


# --------------------------------------------------------------------------- rule 7: hyphens

_HYPHENS = "-‐­"
_WORD = re.compile(r"[A-Za-z]+(?:[-‐][A-Za-z]+)*")
_TAIL = re.compile(r"([A-Za-z0-9]+)[" + _HYPHENS + r"]\s*$")
_HEAD = re.compile(r"^\s*([A-Za-z]+)")
_DASH_END = re.compile(r"[–—]\s*$")


@dataclass
class HyphenCall:
    page: int
    line: int               # index of the line that ends with the hyphen
    next_line: int
    left: str
    right: str
    decision: str           # joined | kept
    reason: str             # hyphenated_form | joined_form | chain | capital_or_digit | both_words | default


def dictionary(pages: list[list[Line]]) -> tuple[Counter[str], set[str]]:
    """Every word the document writes in one piece, and every pair of words it
    joins with a hyphen, lower-cased. A word split at a line end counts as
    neither."""
    words: Counter[str] = Counter()
    compounds: set[str] = set()
    for lines in pages:
        after_break = False
        for ln in lines:
            if ln.role != "body" or not ln.text.text.strip():
                continue
            toks = _WORD.findall(ln.text.text)
            # The halves of a word split at a line end ("sugges-" / "tions")
            # are not words of the document.
            breaks = bool(_TAIL.search(ln.text.text))
            if breaks and toks:
                toks = toks[:-1]
            if after_break and toks:
                toks = toks[1:]
            after_break = breaks
            for tok in toks:
                tok = tok.lower().replace("‐", "-")
                if "-" in tok:
                    parts = tok.split("-")
                    # "debt-to-GDP" also attests "debt-to" and "to-gdp".
                    compounds.update(f"{x}-{y}" for x, y in zip(parts, parts[1:]))
                    words.update(parts)
                else:
                    words[tok] += 1
    return words, compounds


# A fallback "both halves are words" only trusts halves the document uses
# often, and long enough not to be a prefix ("in-stalled", "pre-pares").
BOTH_WORDS_MIN_LEN = 3


def decide_hyphen(left: str, right: str, words: Counter[str], compounds: set[str],
                  chain: bool = False) -> tuple[str, str]:
    """Join "produc-" + "tion", keep "all-" + "time": ask the document first,
    then fall back to how the halves look."""
    a, b = left.lower(), right.lower()
    if f"{a}-{b}" in compounds:
        return "kept", "hyphenated_form"
    if words[a + b]:
        return "joined", "joined_form"
    if chain:
        return "kept", "chain"                  # "back-of-the-" / "envelope"
    if left[-1].isdigit() or right[0].isupper() or (left.isupper() and len(left) <= 4):
        return "kept", "capital_or_digit"       # "1,242-" / "page", "Vance-" / "Cooks", "U-" / "shaped"
    if len(a) >= BOTH_WORDS_MIN_LEN and len(b) >= BOTH_WORDS_MIN_LEN and words[a] and words[b]:
        return "kept", "both_words"             # "fear-" / "driven", but not "in-" / "stalled"
    return "joined", "default"


def hyphen_calls(pages: list[list[Line]], counts: Counter[str]) -> list[HyphenCall]:
    words, compounds = dictionary(pages)
    calls = []
    for lines in pages:
        body = [i for i, ln in enumerate(lines) if ln.role == "body" and ln.text.text.strip()]
        for pos, i in enumerate(body):
            m = _TAIL.search(lines[i].text.text)
            if not m:
                continue
            counts["hyphen_lines"] += 1
            if pos + 1 >= len(body):
                counts["hyphen_page_end"] += 1      # the word continues on the next page
                continue
            j = body[pos + 1]
            h = _HEAD.match(lines[j].text.text)
            if not h:
                continue
            text = lines[i].text.text.rstrip()
            chain = m.start(1) > 0 and text[m.start(1) - 1] in _HYPHENS
            decision, reason = decide_hyphen(m.group(1), h.group(1), words, compounds, chain)
            if (reason not in ("hyphenated_form", "joined_form") and h.group(1).isupper()
                    and len(h.group(1)) >= 3 and not m.group(1).isupper()):
                # "invest-" then "FIGURE 1.1": a caption interrupts the word in
                # reading order. Joining would glue the word to the caption.
                counts["hyphen_interrupted"] += 1
                continue
            calls.append(HyphenCall(page=lines[i].page, line=i, next_line=j, left=m.group(1),
                                    right=h.group(1), decision=decision, reason=reason))
            counts[f"hyphen_{decision}"] += 1
            counts[f"hyphen_{reason}"] += 1
    return calls


# --------------------------------------------------------------------------- assembly and rule 8

_SPACES = re.compile(r"[ \t  -​　]{2,}|[  -​　]")
_SPACE_NL = re.compile(r" ?\n ?")
_BLANK_LINES = re.compile(r"\n{2,}")


def assemble(lines: list[Line], calls: list[HyphenCall], raw_marks: list[RawMark],
             counts: Counter[str]) -> tuple[Marked, list[int]]:
    """Body lines into page text. Lines of one block join with a space, blocks
    with a line break, and a hyphen call glues two lines into one word.
    Returns the text and where each line starts in it (-1 for furniture)."""
    by_line = {c.line: c for c in calls}
    body = [i for i, ln in enumerate(lines) if ln.role == "body" and ln.text.text.strip()]
    parts: list[Marked] = []
    seps: list[str] = []
    starts = [-1] * len(lines)
    glue = False
    offset = 0
    for pos, i in enumerate(body):
        ln = lines[i]
        text = Marked(ln.text.text, list(ln.text.marks))
        trail = len(text.text) - len(text.text.rstrip())
        if trail:
            text.replace(len(text.text) - trail, len(text.text), "", None)
        lead = len(text.text) - len(text.text.lstrip())
        if lead:
            text.replace(0, lead, "", None)
        if pos:
            prev = lines[body[pos - 1]]
            sep = "" if glue or _DASH_END.search(prev.text.text) else (
                " " if prev.block == ln.block else "\n")
            seps.append(sep)
            offset += len(sep)
        starts[i] = offset
        call = by_line.get(i)
        if call is not None and call.decision == "joined":
            text.replace(len(text.text) - 1, len(text.text), "", None)   # the hyphen
        parts.append(text)
        offset += len(text.text)
        glue = call is not None
    page = Marked.join(parts, seps)
    # Mark each repaired word across the old line break: "produc-⏎tion" ->
    # "production" (joined), "all-⏎time" -> "all-time" (kept).
    for c in sorted(calls, key=lambda c: c.line, reverse=True):
        b = starts[c.next_line]
        if starts[c.line] < 0 or b < 0:
            continue
        joined = c.decision == "joined"
        left_end = b if joined else b - 1          # a kept hyphen sits just before the right half
        left_start = left_end - len(c.left)
        right_end = b + len(c.right)
        before = page.original(left_start, left_end) + "-\n" + page.original(b, right_end)
        page.replace(left_start, right_end, page.text[left_start:right_end],
                     "hyphen" if joined else "compound", before)
        ln = lines[c.line]
        tail = _TAIL.search(ln.raw)
        if tail:
            raw_marks.append(RawMark(ln.raw_start + tail.start(), ln.raw_start + len(ln.raw.rstrip()),
                                     "hyphen" if joined else "compound"))
    # Rule 8: whitespace, unmarked (it changes almost every line).
    page.sub(_SPACES, " ", None)
    page.sub(_SPACE_NL, "\n", None)
    page.sub(_BLANK_LINES, "\n", None)
    stripped = page.text.strip()
    if stripped != page.text:
        lead = len(page.text) - len(page.text.lstrip())
        page.replace(0, lead, "", None)
        page.replace(len(stripped), len(page.text), "", None)
    counts["whitespace_pages"] += 1
    return page, starts


# --------------------------------------------------------------------------- the PyMuPDF path

def clean_pdf(path: Path) -> CleanDoc:
    """PyMuPDF layout + the guard + every rule, for one PDF."""
    import pymupdf

    from rag_studio import guard, parse, structure

    t0 = time.perf_counter()
    path = Path(path)
    counts: Counter[str] = Counter()
    page_lines: list[list[Line]] = []
    layouts: list["PageLayout"] = []
    raw_marks: list[list[RawMark]] = []
    hidden: list[list[guard.Hidden]] = []
    with pymupdf.open(path) as scratch:
        for pg, layout in parse.read_layout(path):
            spans = guard.hidden_text(pg, scratch[pg.number or 0])
            keys: set[tuple[float, float, str]] = set().union(*(h.keys for h in spans)) if spans else set()
            marks: list[RawMark] = []
            lines, offset = [], 0
            for rl in layout.lines:
                lines.append(build_line(rl, offset, keys, layout.glyph_fixes, layout.page, marks, counts))
                offset += len(parse.line_text_raw(rl))
            page_lines.append(lines)
            layouts.append(layout)
            raw_marks.append(marks)
            hidden.append(spans)
            counts["hidden_spans"] += len(spans)
    t_chars = time.perf_counter() - t0

    find_furniture(page_lines, [lay.height for lay in layouts], counts)
    labels = printed_labels(page_lines, [lay.page for lay in layouts])
    leader_lines = [collapse_leaders(lines, counts, marks) for lines, marks in zip(page_lines, raw_marks)]
    calls = hyphen_calls(page_lines, counts)
    by_page: dict[int, list[HyphenCall]] = {}
    for c in calls:
        by_page.setdefault(c.page, []).append(c)

    pages: list[CleanPage] = []
    body_size = structure.body_size(page_lines)
    outline = structure.outline(parse.bookmarks(path))
    for lines, layout, marks, spans, label, leaders in zip(page_lines, layouts, raw_marks, hidden,
                                                           labels, leader_lines):
        for ln in lines:
            if ln.role == "furniture":
                marks.append(RawMark(ln.raw_start, ln.raw_start + len(ln.raw), "furniture"))
        text, starts = assemble(lines, by_page.get(layout.page, []), marks, counts)
        pages.append(CleanPage(
            doc_id=layout.doc_id, page=layout.page, width=layout.width, height=layout.height,
            text_raw=layout.text_raw, text=text.text,
            marks=[asdict(m) for m in text.marks],
            raw_marks=[asdict(m) for m in sorted(marks, key=lambda m: m.start)],
            label=label,
            furniture=[{"text": ln.text.text.strip(), "box": list(ln.box)} for ln in lines
                       if ln.role == "furniture" and ln.text.text.strip()],
            hidden=[h.as_dict() for h in spans],
            headings=structure.headings(lines, starts, body_size),
        ))
        pages[-1].kinds = structure.kinds(pages[-1], toc=leaders >= TOC_MIN_LEADER_LINES)
    structure.assign_sections(pages, outline)
    counts["hyphen_compounds_kept"] = sum(1 for c in calls if c.decision == "kept")

    from rag_studio.parse import PARSER_ID, PARSER_VERSION, file_sha256
    return CleanDoc(
        doc_id=path.stem, sha256=file_sha256(path), parser=PARSER_ID, cleaner=FULL_CLEANER_ID,
        pages=pages, counts=dict(counts),
        info={"parser_version": PARSER_VERSION, "seconds": round(time.perf_counter() - t0, 2),
              "char_rules_seconds": round(t_chars, 2), "body_size": body_size,
              "bookmarks": len(outline),
              "hyphen_calls": [asdict(c) for c in calls]},
    )

