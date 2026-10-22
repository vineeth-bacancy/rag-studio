"""Vector store: LanceDB, embedded (a folder on disk).

**One table per build.** Rebuilding an index never touches the table answers
are being served from: a new table is written, checked, and only then does the
profile's pointer file switch to it (written atomically, rule 4). The previous
build is kept for rollback. LanceDB OSS has no `rename_table`, which is why the
pointer is ours.

**Exact search.** At about 1,200 passages, comparing the question with every
vector takes a few milliseconds (measured 7-10 ms for 1,000 × 768), so there is
no approximate index to lose recall to. The metric is always set to cosine:
LanceDB's default is squared L2.

**A word index beside the vectors (M3).** Every build gets LanceDB's native
full-text index for BM25 ("by words"), over exactly the text that was embedded
(`search_text`: the section label and the passage). Builds made before M3 have
no `search_text`; their index is over `text`. Stop words are kept: "What is the
global growth" must still match "global", and BM25's IDF already gives common
words almost no weight. Adding the word index to an existing build changes no
row and no vector, so its manifest and earlier runs stay valid.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import lancedb
import pyarrow as pa

from lancedb.index import FTS

from rag_studio import paths
from rag_studio.files import read_json, write_json_atomic
from rag_studio.profiles import IndexProfile

KEEP_BUILDS = 2

# What the UI and the prompt read back. `vector` and `search_text` stay in the
# table. Builds made before M2 lack kind, page_label and flags, and builds
# made before M3 lack label; a read fills them with their defaults.
COLUMNS = ["chunk_id", "doc_id", "page", "section_path", "type", "text", "context",
           "content_hash", "parser", "chunker", "char_start", "char_end",
           "kind", "page_label", "flags", "label"]
DEFAULTS: dict[str, Any] = {"kind": "body", "page_label": None, "flags": [], "label": None}

# The word index (research C §2.1, D §10): simple tokenizer, English stemming,
# stop words kept, positions on (phrase queries), lower case, accents folded.
FTS_SETTINGS: dict[str, Any] = {"base_tokenizer": "simple", "language": "English", "stem": True,
                                "remove_stop_words": False, "lower_case": True,
                                "ascii_folding": True, "with_position": True}


def schema(dimensions: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("page", pa.int32(), nullable=False),          # 1-based PDF page
        pa.field("section_path", pa.string()),                 # null until M2-M3
        pa.field("type", pa.string(), nullable=False),         # text | table | figure
        pa.field("text", pa.string(), nullable=False),
        pa.field("context", pa.string()),                      # context line (M5)
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), dimensions), nullable=False),
        pa.field("parser", pa.string(), nullable=False),
        pa.field("chunker", pa.string(), nullable=False),
        pa.field("char_start", pa.int32(), nullable=False),
        pa.field("char_end", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),         # body | toc | references
        pa.field("page_label", pa.string()),                   # the printed page number
        pa.field("flags", pa.list_(pa.string()), nullable=False),  # instruction-like phrases (guard)
        pa.field("label", pa.string()),                        # section label (M3), or null
        pa.field("search_text", pa.string(), nullable=False),  # exactly what was embedded and indexed
    ])


def columns(table: Any) -> list[str]:
    """The readable columns this build actually has."""
    names = set(table.schema.names)
    return [c for c in COLUMNS if c in names]


def rerank_columns(table: Any) -> list[str]:
    """What search reads beyond `columns`: the text a reranker reads (exactly
    what was embedded). It rides on hits inside the engine and is never
    serialised for the UI."""
    return ["search_text"] if "search_text" in table.schema.names else []


def connect() -> Any:
    paths.LANCEDB.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(paths.LANCEDB))


def pointer_path(profile_id: str):
    return paths.INDEXES / f"{profile_id}.json"


def new_table_name(profile_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{profile_id}__{stamp}"


def write_table(name: str, rows: list[dict[str, Any]], profile: IndexProfile) -> Any:
    table = pa.Table.from_pylist(rows, schema=schema(profile.embedding.dimensions))
    return connect().create_table(name, data=table, mode="create")


def activate(manifest: dict[str, Any]) -> None:
    """Point the profile at a finished build. Atomic: readers see the old
    manifest or the new one. The manifest is copied into the build history."""
    write_json_atomic(paths.INDEXES / "builds" / f"{manifest['table']}.json", manifest)
    write_json_atomic(pointer_path(manifest["profile"]), manifest)


def active_manifest(profile_id: str) -> dict[str, Any] | None:
    return read_json(pointer_path(profile_id))


def open_table(name: str) -> Any:
    return connect().open_table(name)


def builds(profile_id: str) -> list[dict[str, Any]]:
    """Every finished build of a profile, newest first."""
    folder = paths.INDEXES / "builds"
    if not folder.is_dir():
        return []
    out = [m for p in folder.glob(f"{profile_id}__*.json") if (m := read_json(p))]
    return sorted(out, key=lambda m: m["built_at"], reverse=True)


def prune(profile_id: str, keep: int = KEEP_BUILDS) -> list[str]:
    """Drop all but the newest `keep` builds, never the active one."""
    active = (active_manifest(profile_id) or {}).get("table")
    db = connect()
    pattern = re.compile(rf"^{re.escape(profile_id)}__\d{{8}}T\d+Z$")
    names = sorted((t for t in db.list_tables().tables if pattern.match(t)), reverse=True)
    dropped = []
    for name in names[keep:]:
        if name == active:
            continue
        db.drop_table(name)
        (paths.INDEXES / "builds" / f"{name}.json").unlink(missing_ok=True)
        dropped.append(name)
    return dropped


def vectors_by_hash(table: Any) -> dict[str, list[float]]:
    """content_hash -> vector of an existing build, for incremental rebuilds."""
    t = table.to_arrow().select(["content_hash", "vector"])
    return dict(zip(t["content_hash"].to_pylist(), t["vector"].to_pylist()))


# --------------------------------------------------------------------------- the word index

class WordIndexMissing(RuntimeError):
    """This build has no full-text index yet."""


def fts_column(table: Any) -> str:
    """The column BM25 searches: what was embedded, or the text for builds
    made before M3."""
    return "search_text" if "search_text" in table.schema.names else "text"


def fts_index(table: Any) -> dict[str, Any] | None:
    """The build's full-text index, or None."""
    for idx in table.list_indices():
        if idx.index_type == "FTS":
            return {"name": idx.name, "column": idx.columns[0]}
    return None


def ensure_fts(table: Any) -> dict[str, Any]:
    """Create the word index if the build has none. Idempotent. Returns what
    a manifest records about it."""
    found = fts_index(table)
    created = False
    if found is None:
        column = fts_column(table)
        table.create_index(column, config=FTS(**FTS_SETTINGS))
        found = fts_index(table)
        if found is None:
            raise RuntimeError("The word index was not created. Rebuild the index with ./rag ingest --force.")
        table.wait_for_index([found["name"]])
        created = True
    return {**found, "settings": FTS_SETTINGS, "created": created,
            "lancedb_version": lancedb.__version__, "table_version": table.version}


def doc_filter(doc_id: str, known: set[str] | list[str]) -> str:
    """A `where` expression for one document, built ONLY from a document id the index knows. Free text
    (a model's output, a request field) never reaches a `where` string: an unknown id is refused here."""
    if doc_id not in set(known):
        raise ValueError(f"{doc_id!r} is not a document in this index.")
    return "doc_id = '" + doc_id.replace("'", "''") + "'"


def search_bm25(table: Any, query: str, k: int, where: str | None = None) -> list[dict[str, Any]]:
    """The k best passages by BM25 over the word index. `score` is LanceDB's
    BM25 score: comparable within one question only."""
    if fts_index(table) is None:
        raise WordIndexMissing("This index has no word index yet, so BM25 and hybrid search are "
                               "unavailable. Add it with ./rag ingest --profile <profile>.")
    if not query.strip():
        return []
    q = table.search(query, query_type="fts", fts_columns=fts_column(table))
    if where:
        q = q.where(where, prefilter=True)             # filter first, then take the k best: not k best, then filter
    rows = q.limit(k).select([*columns(table), *rerank_columns(table), "_score"]).to_list()
    return [{c: r.get(c, DEFAULTS.get(c)) for c in COLUMNS} | {"search_text": r.get("search_text") or r["text"]}
            | {"rank": rank, "score": float(r["_score"])}
            for rank, r in enumerate(rows, start=1)]


# --------------------------------------------------------------------------- dense search

def search_dense(table: Any, vector: list[float], k: int, where: str | None = None) -> list[dict[str, Any]]:
    """The k nearest passages by cosine similarity, exactly (a flat scan).

    `score` is the cosine similarity (1 - LanceDB's cosine distance). Scores on
    this corpus bunch between 0.6 and 0.9, so the rank says more than the
    number (spec §6).
    """
    q = table.search(vector).distance_type("cosine").bypass_vector_index()
    if where:
        q = q.where(where, prefilter=True)
    rows = q.limit(k).select([*columns(table), *rerank_columns(table), "_distance"]).to_list()
    out = []
    for rank, r in enumerate(rows, start=1):
        r = ({c: r.get(c, DEFAULTS.get(c)) for c in COLUMNS} | {"search_text": r.get("search_text") or r["text"]}
             | {"rank": rank, "score": 1.0 - float(r["_distance"])})
        out.append(r)
    return out


def get_passage(table: Any, chunk_id: str) -> dict[str, Any] | None:
    """One passage by id, or None."""
    safe = chunk_id.replace("'", "''")
    rows = table.search().where(f"chunk_id = '{safe}'").select(columns(table)).limit(1).to_list()
    return {c: rows[0].get(c, DEFAULTS.get(c)) for c in COLUMNS} if rows else None
