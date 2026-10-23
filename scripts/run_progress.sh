#!/usr/bin/env bash
# A live view of the evaluation run in progress: a bar, the ETA, outcomes so far, free memory.
# Reads data/runs/*.json every 5 s (no model, no API). Ctrl-C to leave; the run is not affected.
# Usage: scripts/run_progress.sh
cd "$(dirname "$0")/.." || exit 1
while true; do
    printf '\033[H\033[2J'
    .venv/bin/python - <<'PY'
import glob, json, os, time
from datetime import datetime, timezone
files = sorted(glob.glob("data/runs/*.json"), key=os.path.getmtime)
run = None
for p in reversed(files[-5:]):
    try:
        r = json.load(open(p))
    except (OSError, ValueError):
        continue
    if r.get("status") == "running" or run is None:
        run = r
        if r.get("status") == "running":
            break
if not run:
    print("No evaluation run yet.")
    raise SystemExit
prog = run.get("progress") or {}
done, total = prog.get("done", 0), prog.get("total") or 0
rows = [x for x in run.get("results", []) if x.get("outcome")]
c = {o: sum(1 for x in rows if x["outcome"] == o) for o in ("PASS", "REFUSED", "WRONG")}
width = 40
fill = int(width * done / total) if total else 0
started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
elapsed = (datetime.now(timezone.utc) - started).total_seconds()
answered = [x for x in rows if (x.get("timings") or {}).get("generate_ms")]
per = (sum(x["timings"]["generate_ms"] for x in answered) / len(answered) / 1000) if answered else None
left = (total - done) * per if per and total else None
avail = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
print(f"{run.get('label') or run['id']}   [{run['status']}]")
print(f"[{'#' * fill}{'.' * (width - fill)}] {done}/{total}")
print(f"elapsed {elapsed / 60:.1f} min" + (f" | ~{per:.0f} s per answer | about {left / 60:.0f} min left" if left else ""))
print(f"Pass {c['PASS']} · Refused {c['REFUSED']} · Wrong {c['WRONG']}")
print("last: " + "  ".join(f"{x['id']} {x['outcome']}" for x in rows[-6:]))
print(f"free memory {avail:,} MB" + ("   (the watchdog stops the run below 800 MB)" if avail < 3000 else ""))
PY
    sleep 5
done
