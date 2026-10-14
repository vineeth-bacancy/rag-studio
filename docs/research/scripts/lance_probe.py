"""Small LanceDB probe for RAG Studio research (2026-10-14).

Usage: python lance_probe.py <data_dir>
"""

import inspect
import shutil
import statistics
import sys
import time
import traceback
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa

import lancedb

DATA = Path(sys.argv[1])
if DATA.exists():
    shutil.rmtree(DATA)
DATA.mkdir(parents=True)

DIM = 768
N = 1000
rng = np.random.default_rng(42)


def section(name):
    print(f"\n===== {name}")


def safe(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print("ERROR:", type(exc).__name__, str(exc)[:400])


section("versions")
for pkg in ("lancedb", "pylance", "pyarrow", "numpy", "pandas", "tantivy"):
    try:
        print(pkg, version(pkg))
    except Exception:
        print(pkg, "NOT INSTALLED")

section("signatures")
from lancedb.table import LanceTable  # noqa: E402

for name in ("create_fts_index", "create_index", "search", "optimize",
             "cleanup_old_versions", "list_versions", "checkout", "restore",
             "merge_insert", "add"):
    fn = getattr(LanceTable, name, None)
    print(name, inspect.signature(fn) if fn else "MISSING")
DBConn = type(lancedb.connect(str(DATA)))
for name in ("create_table", "rename_table", "drop_table", "table_names", "list_tables", "open_table"):
    fn = getattr(DBConn, name, None)
    print("db.", name, inspect.signature(fn) if fn else "MISSING")
from lancedb.rerankers import RRFReranker  # noqa: E402
print("RRFReranker", inspect.signature(RRFReranker.__init__))

# ---------------------------------------------------------------- data
WORDS = ("revenue growth margin inflation forecast policy bank credit debt "
         "fiscal emerging market outlook risk investment trade energy "
         "commodity exchange rate poverty labour productivity").split()


def norm(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


vecs = norm(rng.standard_normal((N, DIM)))
texts = [" ".join(rng.choice(WORDS, size=12)) for _ in range(N)]
texts[7] = "Global growth is projected at 2.5% in FY 2018 according to the outlook"
texts[8] = "The theory of growing economies; running costs rose in FY2018"
texts[9] = "Inflation fell to 2.5 percent while the old man and the sea sold well"
schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])


def make_tbl(start, n, vv, tt):
    return pa.table({
        "chunk_id": [f"c{start + i}" for i in range(n)],
        "page": pa.array([(start + i) % 294 + 1 for i in range(n)], pa.int32()),
        "text": tt,
        "content_hash": [f"h{start + i}" for i in range(n)],
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vv.ravel(), pa.float32()), DIM),
    }, schema=schema)


db = lancedb.connect(str(DATA))
tbl = db.create_table("chunks", data=make_tbl(0, N, vecs, texts), schema=schema, mode="overwrite")
print("rows", tbl.count_rows(), "version", tbl.version, "schema vector type", tbl.schema.field("vector").type)

# ---------------------------------------------------------------- exact search
section("exact (flat) search, no index")
queries = norm(rng.standard_normal((20, DIM)))


def exact_ids(q, k=10, metric="cosine"):
    return tbl.search(q).distance_type(metric).bypass_vector_index().limit(k).select(["chunk_id"]).to_arrow()


def timed(fn, reps):
    ts = []
    for r in range(reps):
        t0 = time.perf_counter()
        fn(r)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), max(ts)


res = tbl.search(queries[0]).distance_type("cosine").limit(3).to_arrow()
print("columns", res.column_names)
print("first distances cosine", [round(x, 4) for x in res["_distance"].to_pylist()])
for metric in ("l2", "cosine", "dot"):
    r = tbl.search(queries[0]).distance_type(metric).limit(3).select(["chunk_id"]).to_arrow()
    print(metric, list(zip(r["chunk_id"].to_pylist(), [round(x, 4) for x in r["_distance"].to_pylist()])))
r = tbl.search(queries[0]).limit(3).select(["chunk_id"]).to_arrow()
print("default metric (none given)", [round(x, 4) for x in r["_distance"].to_pylist()])
med, mx = timed(lambda i: exact_ids(queries[i % 20]), 40)
print(f"exact top-10 cosine over {N}x{DIM}: median {med:.2f} ms, max {mx:.2f} ms")
# self-query sanity: a stored vector must come back first with distance ~0
r = tbl.search(vecs[123]).distance_type("cosine").limit(1).select(["chunk_id"]).to_arrow()
print("self query ->", r["chunk_id"][0], round(r["_distance"][0].as_py(), 6))

# ---------------------------------------------------------------- FTS
section("full-text search")


def fts_create():
    t0 = time.perf_counter()
    tbl.create_fts_index("text", with_position=True, remove_stop_words=False,
                         stem=True, language="English", lower_case=True,
                         ascii_folding=True, replace=True)
    print(f"create_fts_index took {(time.perf_counter() - t0) * 1000:.0f} ms")
    print("indices", tbl.list_indices())


safe(fts_create)
safe(lambda: tbl.create_fts_index("text", use_tantivy=True, replace=True))
safe(lambda: tbl.create_fts_index("text", use_tantivy=False, replace=True, with_position=True, remove_stop_words=False))


def fts_q(q, k=5, **kw):
    r = tbl.search(q, query_type="fts", **kw).limit(k).select(["chunk_id", "text"]).to_arrow()
    return list(zip(r["chunk_id"].to_pylist(), [round(x, 3) for x in r["_score"].to_pylist()]))


for q in ["inflation", "growing", "grow", "2.5%", "2.5", "\"FY 2018\"", "FY2018", "2018",
          "\"old man and the sea\"", "theory", "the"]:
    try:
        r = tbl.search(q, query_type="fts").limit(3).select(["chunk_id", "text"]).to_arrow()
        print(repr(q), "->", list(zip(r["chunk_id"].to_pylist(), [round(x, 3) for x in r["_score"].to_pylist()])),
              "cols", r.column_names)
    except Exception as exc:  # noqa: BLE001
        print(repr(q), "ERROR", type(exc).__name__, str(exc)[:200])


def phrase_obj():
    from lancedb.query import MatchQuery, PhraseQuery
    r = tbl.search(PhraseQuery("old man and the sea", "text")).limit(3).select(["chunk_id"]).to_arrow()
    print("PhraseQuery ->", r["chunk_id"].to_pylist(), r.column_names)
    r = tbl.search(MatchQuery("inflaton", "text", fuzziness=1)).limit(3).select(["chunk_id"]).to_arrow()
    print("MatchQuery fuzzy ->", r["chunk_id"].to_pylist())


safe(phrase_obj)
med, mx = timed(lambda i: tbl.search("inflation risk", query_type="fts").limit(10).to_arrow(), 30)
print(f"fts query median {med:.2f} ms max {mx:.2f} ms")

# ---------------------------------------------------------------- hybrid
section("hybrid search")


def hybrid():
    qv = vecs[7] + 0.05 * rng.standard_normal(DIM).astype(np.float32)
    qv = norm(qv)
    r = (tbl.search(query_type="hybrid").vector(qv).text("growth outlook 2018")
         .distance_type("cosine").limit(5).select(["chunk_id"]).to_arrow())
    print("hybrid default cols", r.column_names)
    print(list(zip(r["chunk_id"].to_pylist(), [round(x, 4) for x in r["_relevance_score"].to_pylist()])))
    rr = RRFReranker(return_score="all")
    r = (tbl.search(query_type="hybrid").vector(qv).text("growth outlook 2018")
         .distance_type("cosine").rerank(rr).limit(5).with_row_id(True).to_arrow())
    print("hybrid return_score=all cols", [c for c in r.column_names if c != "vector"])
    keep = [c for c in ("chunk_id", "_distance", "_score", "_relevance_score", "_rowid") if c in r.column_names]
    for row in r.select(keep).to_pylist():
        print("  ", row)
    # separate lists for the UI
    dense = tbl.search(qv).distance_type("cosine").limit(5).select(["chunk_id"]).with_row_id(True).to_arrow()
    sparse = tbl.search("growth outlook 2018", query_type="fts").limit(5).select(["chunk_id"]).with_row_id(True).to_arrow()
    print("dense list", dense["chunk_id"].to_pylist())
    print("bm25 list", sparse["chunk_id"].to_pylist(), [round(x, 3) for x in sparse["_score"].to_pylist()])
    # manual RRF with k=60 must reproduce LanceDB's order if semantics match
    fused = {}
    for lst in (dense["_rowid"].to_pylist(), sparse["_rowid"].to_pylist()):
        for rank, rid in enumerate(lst):
            fused[rid] = fused.get(rid, 0) + 1 / (rank + 60)
    print("manual RRF (k=60, 0-based rank) top", sorted(fused.items(), key=lambda kv: -kv[1])[:5])
    med, mx = timed(lambda i: tbl.search(query_type="hybrid").vector(qv).text("growth outlook").limit(10).to_arrow(), 20)
    print(f"hybrid median {med:.2f} ms max {mx:.2f} ms")


safe(hybrid)


def hybrid_rrf_source():
    src = inspect.getsource(RRFReranker)
    for line in src.splitlines():
        if "def __init__" in line or "1 / (" in line or "rank +" in line or "self.K" in line:
            print("  RRF src:", line.strip())


safe(hybrid_rrf_source)

# ---------------------------------------------------------------- filters
section("filters")
safe(lambda: print("where page<=10 prefilter:", tbl.search(queries[1]).distance_type("cosine")
                   .where("page <= 10", prefilter=True).limit(5).select(["chunk_id", "page"]).to_arrow()["page"].to_pylist()))
safe(lambda: print("where postfilter:", tbl.search(queries[1]).distance_type("cosine")
                   .where("page <= 10", prefilter=False).limit(5).select(["chunk_id", "page"]).to_arrow()["page"].to_pylist()))
safe(lambda: print("fts + where:", tbl.search("inflation", query_type="fts").where("page <= 50")
                   .limit(5).select(["chunk_id", "page"]).to_arrow()["page"].to_pylist()))

# ---------------------------------------------------------------- versions / FTS after append
section("versions and appends")
print("version before add", tbl.version)
newv = norm(rng.standard_normal((5, DIM)))
tbl.add(make_tbl(N, 5, newv, ["zebracorn appears in the appended rows"] * 5))
print("version after add", tbl.version)
print("fts finds appended rows without reindex:", fts_q("zebracorn"))
safe(lambda: print("index_stats", tbl.index_stats("text_idx")))
safe(lambda: print("list_versions", [(v["version"], str(v["timestamp"])[:19]) for v in tbl.list_versions()]))


def tags_and_checkout():
    tbl.tags.create("before-upsert", tbl.version)
    print("tags", tbl.tags.list())
    tbl.checkout(1)
    print("checked out v1 rows", tbl.count_rows(), "version", tbl.version)
    tbl.checkout_latest()
    print("latest rows", tbl.count_rows(), "version", tbl.version)


safe(tags_and_checkout)

# ---------------------------------------------------------------- merge_insert
section("merge_insert upsert by content hash")


def upsert():
    upd = make_tbl(0, 3, vecs[:3], ["changed text alpha", "changed text beta", "changed text gamma"])
    upd = upd.set_column(upd.schema.get_field_index("content_hash"), "content_hash",
                         pa.array(["hNEW0", "hNEW1", "hNEW2"]))
    res = (tbl.merge_insert("chunk_id").when_matched_update_all(where="target.content_hash != source.content_hash")
           .when_not_matched_insert_all().execute(upd))
    print("merge_insert result:", res)
    print("version", tbl.version, "rows", tbl.count_rows())
    print("fts sees updated text:", fts_q("gamma"))
    # full sync: delete rows not present in source
    src = tbl.to_arrow().slice(0, 900)
    res = (tbl.merge_insert("chunk_id").when_matched_update_all(where="target.content_hash != source.content_hash")
           .when_not_matched_insert_all().when_not_matched_by_source_delete().execute(src))
    print("full-sync merge result:", res, "rows", tbl.count_rows())
    print("indices after merges", [(i.name, i.index_type) for i in tbl.list_indices()])


safe(upsert)


def restore_test():
    v = tbl.tags.get_version("before-upsert")
    tbl.restore(v)
    print("restored to tag version", v, "-> new version", tbl.version, "rows", tbl.count_rows())


safe(restore_test)


def optimize_test():
    n0 = len(tbl.list_versions())
    stats = tbl.optimize(cleanup_older_than=timedelta(seconds=0), delete_unverified=True)
    print("optimize stats:", stats)
    print("versions before/after", n0, len(tbl.list_versions()), "tags kept", tbl.tags.list())
    safe(lambda: print("index_stats after optimize", tbl.index_stats("text_idx")))


safe(optimize_test)

# ---------------------------------------------------------------- swap patterns
section("swap patterns")
safe(lambda: (db.rename_table("chunks", "chunks_renamed"), print("rename_table OK", db.table_names())))


def overwrite_keeps_index():
    t2 = db.create_table("swap_demo", data=make_tbl(0, 300, vecs[:300], texts[:300]), schema=schema)
    t2.create_fts_index("text", replace=True)
    print("swap_demo indices before overwrite", [i.name for i in t2.list_indices()])
    t2 = db.create_table("swap_demo", data=make_tbl(0, 300, vecs[:300], texts[:300]), schema=schema, mode="overwrite")
    print("after create_table(mode=overwrite): indices", [i.name for i in t2.list_indices()], "version", t2.version)
    t3 = db.open_table("swap_demo")
    t3.create_fts_index("text", replace=True)
    t3.add(make_tbl(0, 300, vecs[:300], texts[:300]), mode="overwrite")
    print("after add(mode=overwrite): indices", [i.name for i in t3.list_indices()], "version", t3.version)


safe(overwrite_keeps_index)

# ---------------------------------------------------------------- ANN index + recall
section("ANN indexes and recall@10 vs exact")
tname = "ann_demo"
ta = db.create_table(tname, data=make_tbl(0, N, vecs, texts), schema=schema, mode="overwrite")
gt = [set(ta.search(q).distance_type("cosine").bypass_vector_index().limit(10)
          .select(["chunk_id"]).to_arrow()["chunk_id"].to_pylist()) for q in queries]
# second, more realistic query set: perturbed copies of stored vectors (true neighbour exists)
pq = norm(vecs[:20] + 0.3 * rng.standard_normal((20, DIM)).astype(np.float32) / np.sqrt(DIM) * 10)
gt2 = [set(ta.search(q).distance_type("cosine").bypass_vector_index().limit(10)
           .select(["chunk_id"]).to_arrow()["chunk_id"].to_pylist()) for q in pq]


def recall(qs, truth, **kw):
    hits = 0
    ts = []
    for q, g in zip(qs, truth):
        b = ta.search(q).distance_type("cosine").limit(10).select(["chunk_id"])
        if "nprobes" in kw:
            b = b.nprobes(kw["nprobes"])
        if "refine" in kw:
            b = b.refine_factor(kw["refine"])
        t0 = time.perf_counter()
        ids = b.to_arrow()["chunk_id"].to_pylist()
        ts.append((time.perf_counter() - t0) * 1000)
        hits += len(g & set(ids))
    return hits / (10 * len(qs)), statistics.median(ts)


def build(index_type, **params):
    t0 = time.perf_counter()
    ta.create_index(metric="cosine", vector_column_name="vector", index_type=index_type, replace=True, **params)
    print(f"{index_type} {params} built in {(time.perf_counter() - t0):.2f}s; indices",
          [(i.name, i.index_type) for i in ta.list_indices()])
    safe(lambda: print("   index_stats", ta.index_stats("vector_idx")))
    for kw in ({}, {"nprobes": 4}, {"nprobes": 20}, {"refine": 5}, {"nprobes": 64, "refine": 10}):
        r1, t1 = recall(queries, gt, **kw)
        r2, t2 = recall(pq, gt2, **kw)
        print(f"   {kw or 'defaults'}: recall@10 random-queries={r1:.3f} ({t1:.2f} ms)  near-dup-queries={r2:.3f} ({t2:.2f} ms)")


safe(lambda: build("IVF_PQ"))
safe(lambda: build("IVF_PQ", num_partitions=8, num_sub_vectors=96))
safe(lambda: build("IVF_HNSW_SQ", num_partitions=1))
safe(lambda: build("IVF_FLAT", num_partitions=8))

section("minimum rows for IVF_PQ")
for n in (100, 255, 256, 300):
    def small(n=n):
        ts = db.create_table(f"small_{n}", data=make_tbl(0, n, vecs[:n], texts[:n]), schema=schema, mode="overwrite")
        ts.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=1, num_sub_vectors=48)
        print(n, "rows: IVF_PQ OK")
    safe(small)

print("\nDONE; data dir size MB:", round(sum(f.stat().st_size for f in DATA.rglob('*') if f.is_file()) / 1e6, 1))
