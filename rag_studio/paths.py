"""Where everything lives on disk.

One module owns the paths so a test can point the whole engine at a temporary
directory by setting `RAG_STUDIO_DATA` before import.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get("RAG_STUDIO_DATA", ROOT / "data"))
PDFS = DATA / "pdfs"          # committed: the source documents
GOLDEN = DATA / "golden"      # committed: verified test questions
LANCEDB = DATA / "lancedb"    # rebuilt: one LanceDB database, a table per index build
INDEXES = DATA / "indexes"    # rebuilt: each profile's active-build pointer and build manifests
CACHE = DATA / "cache"        # rebuilt: parsed pages, context lines, figure descriptions
RUNS = DATA / "runs"          # rebuilt: evaluation runs
CHAT = DATA / "chat"          # conversations, one JSON file each
CONTEXT = DATA / "context"    # exported context lines, one JSONL file per profile (hours of GPU time; may be committed)
FIGURES = DATA / "figures"    # exported figure descriptions, one JSONL file per figures profile (GPU time; may be committed)
UPLOADS = DATA / "uploads"    # PDFs added through the browser

# Session state that must never be committed (the Colab token, the connected
# endpoint). Outside the repository on purpose.
STATE = Path(os.environ.get("RAG_STUDIO_STATE", Path.home() / ".local" / "state" / "rag-studio"))

DOCS = ROOT / "docs"
