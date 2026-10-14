"""Third probe: optimize-based cleanup, swap via overwrite while a reader holds the old version."""

import shutil
import sys
from datetime import timedelta
from pathlib import Path

import pyarrow as pa

import lancedb
from lancedb.index import FTS

DATA = Path(sys.argv[1])
if DATA.exists():
    shutil.rmtree(DATA)
db = lancedb.connect(str(DATA))
mk = lambda tag, n: pa.table({"chunk_id": [f"{tag}{i}" for i in range(n)], "text": [f"{tag} text {i}" for i in range(n)]})  # noqa: E731

t = db.create_table("idx_fixed", data=mk("old", 100))
t.create_index("text", config=FTS(), replace=True)
for i in range(3):
    t.add(mk(f"a{i}", 10))
t.tags.create("keep", 2)
print("versions", [v["version"] for v in t.list_versions()], "tags", list(t.tags.list()))
try:
    t.optimize(cleanup_older_than=timedelta(seconds=0))
except Exception as exc:  # noqa: BLE001
    print("optimize with tagged old version ->", str(exc)[:160])
t.tags.delete("keep")
st = t.optimize(cleanup_older_than=timedelta(seconds=0))
print("optimize stats:", st)
print("versions after cleanup", [v["version"] for v in t.list_versions()])

# reader pinned to an old version while a writer overwrites
reader = db.open_table("idx_fixed")
pinned_version = reader.version
reader.checkout(pinned_version)
writer = db.open_table("idx_fixed")
writer.add(mk("new", 50), mode="overwrite")
writer.create_index("text", config=FTS(), replace=True)
print("writer version", writer.version, "rows", writer.count_rows(), "indices", [i.name for i in writer.list_indices()])
print("pinned reader version", reader.version, "rows", reader.count_rows(),
      "hit old:", reader.search("old", query_type="fts").limit(3).to_arrow().num_rows)
fresh = db.open_table("idx_fixed")
print("fresh open version", fresh.version, "hit old:", fresh.search("old", query_type="fts").limit(3).to_arrow().num_rows,
      "hit new:", fresh.search("new", query_type="fts").limit(3).to_arrow().num_rows)
print("tables", list(db.table_names()))
db.drop_table("idx_fixed")
print("after drop", list(db.table_names()))
