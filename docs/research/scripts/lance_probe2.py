"""Second LanceDB probe: new config API, tokenizer edge cases, recall with real near-duplicates."""

import inspect
import shutil
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa

import lancedb
from lancedb import index as lidx

DATA = Path(sys.argv[1])
if DATA.exists():
    shutil.rmtree(DATA)
DATA.mkdir(parents=True)
DIM, N = 768, 1000
rng = np.random.default_rng(7)


def safe(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print("ERROR:", type(exc).__name__, str(exc)[:300])


def norm(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


print("index config classes:", [n for n in dir(lidx) if n[0].isupper()])
for n in ("FTS", "IvfPq", "IvfHnswSq", "HnswSq", "IvfFlat"):
    c = getattr(lidx, n, None)
    print(n, inspect.signature(c) if c else "MISSING")

schema = pa.schema([pa.field("chunk_id", pa.string()), pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), DIM))])
vecs = norm(rng.standard_normal((N, DIM)))
W = "growth inflation policy debt market risk trade energy outlook credit".split()
texts = [" ".join(rng.choice(W, 10)) for _ in range(N)]
special = ["GDP grew 2.5% in FY2018/19", "inflation of 2.5 per cent", "COVID-19 shock hit US$1.2bn exports",
           "FY 2018 results", "e-commerce and post-pandemic recovery", "Thailand and the Philippines",
           "e eory of e firm"]  # last one mimics the dropped 'Th' defect
for i, s in enumerate(special):
    texts[i] = s
data = pa.table({"chunk_id": [f"c{i}" for i in range(N)], "text": texts,
                 "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel(), pa.float32()), DIM)}, schema=schema)
db = lancedb.connect(str(DATA))
t = db.create_table("chunks", data=data, schema=schema)


def fts_new_api():
    t.create_index("text", config=lidx.FTS(with_position=True, remove_stop_words=False, stem=True), replace=True)
    print("new API indices:", [(i.name, i.index_type) for i in t.list_indices()])


safe(fts_new_api)
for q in ["2.5%", "\"2.5%\"", "FY2018", "FY2018/19", "\"FY 2018\"", "2018", "19", "COVID-19", "\"COVID-19\"",
          "covid", "US$1.2bn", "1.2bn", "ecommerce", "e-commerce", "pandemic", "Thailand", "theory", "eory"]:
    try:
        r = t.search(q, query_type="fts").limit(4).select(["chunk_id"]).to_arrow()
        print(f"{q!r:16} -> {r['chunk_id'].to_pylist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"{q!r:16} -> ERROR {str(exc)[:150]}")

print("\n-- phrase query on index without positions")
t2 = db.create_table("nopos", data=data, schema=schema)
t2.create_index("text", config=lidx.FTS(), replace=True)
safe(lambda: print("phrase w/o positions:", t2.search("\"FY 2018\"", query_type="fts").limit(3).to_arrow()["chunk_id"].to_pylist()))
safe(lambda: print("'the' with default stop words removed:", t2.search("the", query_type="fts").limit(3).to_arrow()["chunk_id"].to_pylist()))
print("default FTS() index_details:", {k: v for k, v in t2.list_indices()[0].index_details.items()
                                      if k in ("stem", "remove_stop_words", "with_position", "ascii_folding", "lower_case")})

print("\n-- hybrid candidate depth")
qv = norm(vecs[3] + 0.02 * rng.standard_normal(DIM))
from lancedb.rerankers import RRFReranker  # noqa: E402
for lim in (3, 10):
    r = (t.search(query_type="hybrid").vector(qv).text("inflation growth").limit(lim)
         .rerank(RRFReranker(return_score="all")).to_arrow())
    print(f"hybrid limit={lim}: rows={r.num_rows} with_distance={sum(x is not None for x in r['_distance'].to_pylist())}"
          f" with_score={sum(x is not None for x in r['_score'].to_pylist())}")
safe(lambda: print("fast_search exists:", hasattr(t.search(qv), "fast_search")))

print("\n-- merge_insert vs index freshness")
upd = data.slice(0, 50).set_column(1, "text", pa.array(["updated row text"] * 50))
print(t.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(upd))
print("fts index_stats after update:", t.index_stats("text_idx"))
print("hits for 'updated':", t.search("updated", query_type="fts").limit(100).to_arrow().num_rows)
t.optimize()
print("after optimize():", t.index_stats("text_idx"))

print("\n-- recall with near-duplicate queries (noise norm ~0.3)")
ta = db.create_table("ann", data=data, schema=schema)
queries = norm(vecs[100:120] + 0.3 * rng.standard_normal((20, DIM)) / np.sqrt(DIM))
gt = [set(ta.search(q).distance_type("cosine").bypass_vector_index().limit(10).select(["chunk_id"])
          .to_arrow()["chunk_id"].to_pylist()) for q in queries]
print("true NN is the source row for", sum(f"c{100 + i}" in g for i, g in enumerate(gt)), "of 20")


def rec(**kw):
    hits, ts = 0, []
    for q, g in zip(queries, gt):
        b = ta.search(q).distance_type("cosine").limit(10).select(["chunk_id"])
        if "nprobes" in kw:
            b = b.nprobes(kw["nprobes"])
        if "refine" in kw:
            b = b.refine_factor(kw["refine"])
        if "ef" in kw:
            b = b.ef(kw["ef"])
        t0 = time.perf_counter()
        ids = b.to_arrow()["chunk_id"].to_pylist()
        ts.append((time.perf_counter() - t0) * 1000)
        hits += len(g & set(ids))
    return f"{hits / 200:.3f} ({statistics.median(ts):.1f} ms)"


for name, cfg in [("IVF_PQ default", lidx.IvfPq(distance_type="cosine")),
                  ("IVF_PQ np=2 sv=96", lidx.IvfPq(distance_type="cosine", num_partitions=2, num_sub_vectors=96)),
                  ("IVF_HNSW_SQ default", lidx.HnswSq(distance_type="cosine")),
                  ("IVF_FLAT np=4", lidx.IvfFlat(distance_type="cosine", num_partitions=4))]:
    def run(name=name, cfg=cfg):
        t0 = time.perf_counter()
        ta.create_index("vector", config=cfg, replace=True)
        print(f"{name}: build {time.perf_counter() - t0:.2f}s | default {rec()} | refine10 {rec(refine=10)}"
              f" | np20+refine10 {rec(nprobes=20, refine=10)}")
    safe(run)
med = statistics.median([(lambda t0: (ta.search(q).distance_type("cosine").bypass_vector_index().limit(10).to_arrow(),
                                       (time.perf_counter() - t0) * 1000)[1])(time.perf_counter()) for q in queries])
print(f"exact flat median {med:.1f} ms")

print("\n-- tags vs cleanup")
t.tags.create("keep", 1)
safe(lambda: t.optimize(cleanup_older_than=timedelta(seconds=0)))
safe(lambda: print("cleanup_old_versions w/ tag:", t.cleanup_old_versions(timedelta(seconds=0))))
t.tags.delete("keep")
print("versions before", len(t.list_versions()))
safe(lambda: print("cleanup stats:", t.cleanup_old_versions(timedelta(seconds=0), delete_unverified=True)))
print("versions after", len(t.list_versions()))
