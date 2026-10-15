"""Parse with Docling: layout, reading order and tables, beside PyMuPDF.

Docling (IBM, MIT licence) runs a layout model over each page and a table
model (TableFormer) over each table it finds, so it knows which text is a
heading, a footer, a caption or a table cell. Its text still comes from the
PDF's text layer: it does not see better, it organises better.

Measured on this laptop (2026-10-15, 8 threads, CPU, no OCR, tables in
accurate mode): 294 pages in 866 s (0.34 pages/s), peak memory 6.4 GB. That is
why only `./rag ingest` converts, once per PDF, and every later step (the API,
the page inspector, a re-chunk) reads the cached result.

`clean_docling()` then applies the cleaning order to Docling's text, reusing
what the PyMuPDF path learnt about the same pages: which text is hidden (the
guard's pixel test), the printed page numbers, and which line-break hyphens
are real compounds, because Docling's own rejoining drops the hyphen in
"conflict-related" and "small-AI".
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_studio import paths
from rag_studio.clean import (_LEADERS, RULE_IDS, CleanDoc, CleanPage, Marked, RawMark,
                              FULL_CLEANER_ID, clean_pdf)
from rag_studio.files import read_json, write_json_atomic
from rag_studio.parse import file_sha256

PARSER_ID = "docling"

# Every option that changes the output is recorded, and is part of the cache key.
OPTIONS: dict[str, Any] = {
    "do_ocr": False,                 # every page here has a text layer
    "table_mode": "accurate",
    "heading_hierarchy": True,       # heading levels from bookmarks, numbering, font style
    "generate_parsed_pages": True,   # font style for the heading levels
    "threads": 8,
}


def version() -> str:
    from importlib.metadata import version as v
    return v("docling")


def _cache_key(sha: str) -> str:
    import hashlib
    import json
    opts = hashlib.sha256(json.dumps({**OPTIONS, "docling": version()}, sort_keys=True).encode()).hexdigest()
    return f"{sha[:16]}-{opts[:8]}"


def cache_path(path: Path, sha: str | None = None) -> Path:
    return paths.CACHE / "docling" / f"{_cache_key(sha or file_sha256(path))}.json"


def is_converted(path: Path) -> bool:
    return cache_path(path).is_file()


def convert(path: Path, *, progress: Any = None) -> tuple[Any, dict[str, Any]]:
    """The DoclingDocument for a PDF, from the cache or by running Docling.
    Returns it with its metadata (seconds, version, options)."""
    from docling_core.types.doc.document import DoclingDocument

    path = Path(path)
    sha = file_sha256(path)
    out = cache_path(path, sha)
    meta_path = out.with_suffix(".meta.json")
    if out.is_file():
        return DoclingDocument.load_from_json(out), read_json(meta_path) or {}

    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (HeadingHierarchyOptions, PdfPipelineOptions,
                                                    TableFormerMode)
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        do_ocr=OPTIONS["do_ocr"], do_table_structure=True,
        generate_parsed_pages=OPTIONS["generate_parsed_pages"],
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=OPTIONS["heading_hierarchy"]),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU,
                                               num_threads=OPTIONS["threads"]))
    setattr(opts.table_structure_options, "mode", TableFormerMode.ACCURATE)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    if progress:
        progress("docling_start", {"doc_id": path.stem})
    t = time.perf_counter()
    doc = conv.convert(path).document
    seconds = time.perf_counter() - t
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    doc.save_as_json(tmp)
    tmp.replace(out)
    meta = {"doc_id": path.stem, "sha256": sha, "docling": version(), "options": OPTIONS,
            "seconds": round(seconds, 1), "pages": len(doc.pages)}
    write_json_atomic(meta_path, meta)
    if progress:
        progress("docling_done", {"doc_id": path.stem, "seconds": seconds, "pages": len(doc.pages)})
    return doc, meta


# --------------------------------------------------------------------------- Docling text

def _box(prov: Any, height: float) -> list[float]:
    """Docling boxes have a bottom-left origin; everything else here is top-left."""
    b = prov.bbox
    return [float(b.l), float(height - b.t), float(b.r), float(height - b.b)]


def _cell(text: str) -> str:
    return " ".join(str(text).split())


def table_text(item: Any, doc: Any) -> str:
    """A table as self-contained rows: "World: 2023 = 2.8; 2024 = 2.9; ...".
    A fixed-size passage cut from the middle of a table still says which
    column each number belongs to (defect #1)."""
    grid = item.data.grid
    if not grid:
        return ""
    head_rows = 0
    while head_rows < len(grid) and all(c.column_header for c in grid[head_rows] if _cell(c.text)):
        head_rows += 1
    head_rows = min(head_rows, len(grid) - 1) if len(grid) > 1 else 0
    ncols = max(len(r) for r in grid)
    names = []
    for col in range(ncols):
        parts: list[str] = []
        for r in grid[:head_rows]:
            t = _cell(r[col].text) if col < len(r) else ""
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        names.append(" ".join(parts))
    lines = []
    caption = _cell(item.caption_text(doc)) if hasattr(item, "caption_text") else ""
    if caption:
        lines.append(caption)
    for row in grid[head_rows:]:
        cells = [_cell(c.text) for c in row]
        if not any(cells):
            continue
        label, rest = cells[0], cells[1:]
        pairs = [f"{names[k + 1]} = {v}" if names[k + 1] else v
                 for k, v in enumerate(rest) if v and k + 1 < len(names)]
        lines.append(f"{label}: {'; '.join(pairs)}" if label and pairs else (label or "; ".join(pairs)))
    return "\n".join(lines)


_SPACE = re.compile(r"[ \t ]{2,}| ")


def clean_docling(path: Path, reference: CleanDoc | None = None, *, progress: Any = None) -> CleanDoc:
    """Docling's body text, page by page, through the cleaning rules that
    apply to it. `reference` is the PyMuPDF path's result for the same PDF
    (computed if not given): hidden text, page labels and hyphen decisions."""
    import unicodedata

    from docling_core.types.doc.common.content_layer import ContentLayer

    from rag_studio import structure

    t0 = time.perf_counter()
    path = Path(path)
    ref = reference or clean_pdf(path)
    doc, meta = convert(path, progress=progress)
    counts: dict[str, int] = {r: 0 for r in RULE_IDS}
    kept_by_page: dict[int, list[dict[str, Any]]] = {}
    for c in ref.info.get("hyphen_calls", []):
        if c["decision"] == "kept":
            kept_by_page.setdefault(c["page"], []).append(c)

    by_page: dict[int, list[tuple[Any, int]]] = {}
    # traverse_pictures: Docling files a chart's labels under the picture; PyMuPDF keeps them too.
    for item, level in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                                         traverse_pictures=True):
        if not getattr(item, "prov", None):
            continue
        by_page.setdefault(int(item.prov[0].page_no), []).append((item, level))

    pages: list[CleanPage] = []
    for rp in ref.pages:
        n = rp.page
        height = float(doc.pages[n].size.height) if n in doc.pages else rp.height
        hidden_boxes = [h["box"] for h in rp.hidden]
        # Compounds the PyMuPDF path kept at a line break on this page, whose
        # merged spelling never occurs in its cleaned text: wherever Docling
        # writes them merged ("shortterm" beside a "short-term"), it merged them.
        py_words = set(re.findall(r"[A-Za-z]+", rp.text))
        restore = {(c["left"], c["right"]) for c in kept_by_page.get(n, [])
                   if c["left"] + c["right"] not in py_words}
        parts: list[Marked] = []
        items: list[dict[str, Any]] = []
        furniture: list[dict[str, Any]] = []
        headings: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        raw_marks: list[RawMark] = []
        toc = False
        offset = 0
        raw_offset = 0
        for item, level in by_page.get(n, []):
            label = str(getattr(item.label, "value", item.label))
            box = _box(item.prov[0], height)
            layer = str(getattr(item.content_layer, "value", item.content_layer))
            if label in ("table", "document_index"):
                text = table_text(item, doc)
            elif label == "picture":
                text = ""
            else:
                text = _cell(getattr(item, "text", "") or "")
            raw_parts.append(text)
            raw_start = raw_offset
            raw_offset += len(text) + 1
            if layer == "furniture" or label in ("page_header", "page_footer"):
                if text:
                    furniture.append({"text": text, "box": box})
                    raw_marks.append(RawMark(raw_start, raw_start + len(text), "furniture"))
                    counts["furniture"] += 1
                continue
            if label == "picture":
                items.append({"label": label, "box": box, "start": None, "end": None})
                continue
            if text and _covered(box, hidden_boxes):
                raw_marks.append(RawMark(raw_start, raw_start + len(text), "hidden"))
                counts["hidden"] += sum(1 for c in text if not c.isspace())
                items.append({"label": label, "box": box, "start": None, "end": None, "hidden": True})
                continue
            if not text:
                continue
            m = Marked()
            for ch in text:
                nk = unicodedata.normalize("NFKC", ch)
                if nk != ch and not ch.isspace():
                    m.append(nk, "nfkc", ch)
                    counts["nfkc"] += 1
                else:
                    m.append(nk)
            if m.sub(_LEADERS, " ", "leaders"):
                counts["leaders"] += 1
            for left, right in restore:
                merged = re.compile(rf"\b{re.escape(left + right)}\b")
                counts["hyphen"] += m.sub(merged, f"{left}-{right}", "compound")
            m.sub(_SPACE, " ", None)
            if label == "document_index":
                toc = True
            if label in ("section_header", "title"):
                headings.append({"start": offset, "text": m.text,
                                 "size": 20.0 - float(getattr(item, "level", 1) or 1)})
            items.append({"label": label, "box": box, "start": offset, "end": offset + len(m.text)})
            parts.append(m)
            offset += len(m.text) + 1
        text = Marked.join(parts, ["\n"] * max(len(parts) - 1, 0))
        page = CleanPage(
            doc_id=rp.doc_id, page=n, width=rp.width, height=rp.height,
            text_raw="\n".join(raw_parts), text=text.text,
            marks=[asdict(mk) for mk in text.marks],
            raw_marks=[asdict(mk) for mk in raw_marks],
            label=rp.label, furniture=furniture, hidden=rp.hidden, headings=headings,
            items=items,
        )
        page.kinds = [{"start": 0, "kind": "toc" if toc else "body"}]
        pages.append(page)
    from rag_studio import parse
    structure.assign_sections(pages, structure.outline(parse.bookmarks(path)))
    counts["tables"] = sum(1 for p in pages for i in p.items if i["label"] == "table")
    counts["pictures"] = sum(1 for p in pages for i in p.items if i["label"] == "picture")
    return CleanDoc(
        doc_id=path.stem, sha256=ref.sha256, parser=PARSER_ID, cleaner=FULL_CLEANER_ID,
        pages=pages, counts=counts,
        info={"parser_version": meta.get("docling") or version(), "options": OPTIONS,
              "convert_seconds": meta.get("seconds"), "seconds": round(time.perf_counter() - t0, 2)},
    )


def _covered(box: list[float], hidden: list[list[float]], share: float = 0.6) -> bool:
    """An item mostly inside text the guard found hidden (GPO p29's footer)."""
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    inside = 0.0
    for h in hidden:
        w = min(box[2], h[2]) - max(box[0], h[0])
        ht = min(box[3], h[3]) - max(box[1], h[1])
        if w > 0 and ht > 0:
            inside += w * ht
    return inside / area >= share

