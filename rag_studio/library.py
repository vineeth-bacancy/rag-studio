"""The document library: which PDFs exist, and what the active index holds of each.

`doc_id` is the file stem everywhere. A request names a document only by
`doc_id`, and `pdf_path()` resolves it against the files actually on disk, so
no path a browser sends is ever opened.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_studio import paths, profiles, store
from rag_studio.parse import pdf_title


def pdf_files() -> dict[str, Path]:
    if not paths.PDFS.is_dir():
        return {}
    return {p.stem: p for p in sorted(paths.PDFS.glob("*.pdf"))}


def pdf_path(doc_id: str) -> Path:
    path = pdf_files().get(doc_id)
    if path is None:
        raise KeyError(f"No document {doc_id!r} in the library.")
    return path


def documents(profile_id: str = profiles.DEFAULT) -> list[dict[str, Any]]:
    """Every PDF, with its title and, once indexed, its page and passage counts.
    Counts are None (shown as "-") until an index includes the document."""
    manifest = store.active_manifest(profile_id) or {}
    indexed = {d["doc_id"]: d for d in manifest.get("documents", [])}
    out = []
    for doc_id, path in pdf_files().items():
        row = indexed.get(doc_id) or {}
        try:
            title = row.get("title") or pdf_title(path)
        except Exception:  # noqa: BLE001 - an unreadable PDF still gets listed
            title = None
        out.append({
            "doc_id": doc_id, "file": path.name, "bytes": path.stat().st_size, "title": title,
            "pages": row.get("pages"), "passages": row.get("passages"),
            "near_empty_pages": row.get("near_empty_pages"),
            "indexed": bool(row),
        })
    return out


# --------------------------------------------------------------------------- the page inspector (M2)

_clean_cache: dict[tuple[str, str], tuple[float, Any]] = {}


def clean_doc(doc_id: str, parser: str) -> Any | None:
    """The cleaned document for one parser, from the disk cache. PyMuPDF is
    computed on first use (seconds); Docling only exists once `./rag ingest
    --profile docling-clean` has converted the PDF (minutes), so it is None
    until then. Kept in memory while its cache file is unchanged."""
    from rag_studio import ingest

    path = pdf_path(doc_id)
    cache = ingest.clean_cache_path(path, parser)
    mtime = cache.stat().st_mtime if cache.is_file() else 0.0
    hit = _clean_cache.get((doc_id, parser))
    if hit and hit[0] == mtime and mtime:
        return hit[1]
    doc = ingest.cleaned_document(path, parser, cached_only=(parser == "docling"))
    if doc is not None:
        _clean_cache[(doc_id, parser)] = (cache.stat().st_mtime if cache.is_file() else 0.0, doc)
    return doc


def _page_counts(page: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in page.raw_marks:
        counts[m["rule"]] = counts.get(m["rule"], 0) + 1
    for m in page.marks:
        if m["rule"] in ("hyphen", "compound") and not any(r["rule"] == m["rule"] for r in page.raw_marks):
            counts[m["rule"]] = counts.get(m["rule"], 0) + 1
    return counts


def _parser_view(page: Any) -> dict[str, Any]:
    from rag_studio import guard

    return {
        "text_raw": page.text_raw, "raw_marks": page.raw_marks,
        "text": page.text, "marks": page.marks,
        "label": page.label, "furniture": page.furniture, "hidden": page.hidden,
        "headings": page.headings, "sections": page.sections, "kinds": page.kinds,
        "items": page.items, "near_empty": page.is_near_empty,
        "flags": guard.instruction_flags(page.text),
        "counts": _page_counts(page),
    }


def inspect_page(doc_id: str, page: int) -> dict[str, Any]:
    """One page through both parsers, for the Documents page inspector."""
    from rag_studio.clean import RULES

    py = clean_doc(doc_id, "pymupdf")
    if py is None or not 1 <= page <= len(py.pages):
        raise IndexError(f"{doc_id} has {len(py.pages) if py else 0} pages, not {page}.")
    dl = clean_doc(doc_id, "docling")
    p = py.pages[page - 1]
    return {
        "doc_id": doc_id, "page": page, "page_count": len(py.pages),
        "width": p.width, "height": p.height, "label": p.label,
        "rules": [{"id": r.id, "defect": r.defect, "label": r.label, "explain": r.explain} for r in RULES],
        "pymupdf": _parser_view(p),
        "docling": _parser_view(dl.pages[page - 1]) if dl is not None else None,
    }


def cleaning_summary(doc_id: str) -> dict[str, Any]:
    """Per-document counts for each parser, and which pages have something to see."""
    out: dict[str, Any] = {"doc_id": doc_id}
    for parser in ("pymupdf", "docling"):
        d = clean_doc(doc_id, parser)
        if d is None:
            out[parser] = None
            continue
        pages = []
        for pg in d.pages:
            c = _page_counts(pg)
            if c or pg.hidden:
                pages.append({"page": pg.page, "label": pg.label, "counts": c})
        out[parser] = {"counts": d.counts, "info": {k: v for k, v in d.info.items() if k != "hyphen_calls"},
                       "pages": pages}
    return out
