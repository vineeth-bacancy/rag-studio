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
