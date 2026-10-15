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
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import lancedb
import pyarrow as pa

from rag_studio import paths
from rag_studio.files import read_json, write_json_atomic
from rag_studio.profiles import IndexProfile

KEEP_BUILDS = 2

# What the UI and the prompt read back. `vector` stays in the table. Builds
# made before M2 lack the last three; a read fills them with their defaults.
COLUMNS = ["chunk_id", "doc_id", "page", "section_path", "type", "text", "context",
           "content_hash", "parser", "chunker", "char_start", "char_end",
           "kind", "page_label", "flags"]
DEFAULTS: dict[str, Any] = {"kind": "body", "page_label": None, "flags": []}


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
    ])


def columns(table: Any) -> list[str]:
    """The readable columns this build actually has."""
    names = set(table.schema.names)
    return [c for c in COLUMNS if c in names]


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


def search_dense(table: Any, vector: list[float], k: int) -> list[dict[str, Any]]:
    """The k nearest passages by cosine similarity, exactly (a flat scan).

    `score` is the cosine similarity (1 - LanceDB's cosine distance). Scores on
    this corpus bunch between 0.6 and 0.9, so the rank says more than the
    number (spec §6).
    """
    rows = (table.search(vector).distance_type("cosine").bypass_vector_index()
            .limit(k).select([*columns(table), "_distance"]).to_list())
    out = []
    for rank, r in enumerate(rows, start=1):
        r = {c: r.get(c, DEFAULTS.get(c)) for c in COLUMNS} | {"rank": rank, "score": 1.0 - float(r["_distance"])}
        out.append(r)
    return out


def get_passage(table: Any, chunk_id: str) -> dict[str, Any] | None:
    """One passage by id, or None."""
    safe = chunk_id.replace("'", "''")
    rows = table.search().where(f"chunk_id = '{safe}'").select(columns(table)).limit(1).to_list()
    return {c: rows[0].get(c, DEFAULTS.get(c)) for c in COLUMNS} if rows else None
