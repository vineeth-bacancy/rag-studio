#!/usr/bin/env bash
# Run one ./rag eval under a memory watchdog and record peak memory. The model
# runs as `llama-server` under Ollama 0.32.6 (its RSS includes the mapped weights).
# Usage: scripts/eval_watchdog.sh <name> <eval args...>
# Writes data/cache/eval-watch/<name>.mem.json; stops the eval if MemAvailable < 800 MB.
set -uo pipefail
SCRATCH=/home/bacancy/Dev/RAG/rag-studio/data/cache/eval-watch; mkdir -p "$SCRATCH"
name="$1"; shift
cd /home/bacancy/Dev/RAG/rag-studio || exit 1
./rag eval "$@" > "$SCRATCH/$name.log" 2>&1 &
pid=$!
min_avail=999999; peak_runner=0; peak_eval=0; killed=0
while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    (( avail < min_avail )) && min_avail=$avail
    runner=$(ps -eo rss,args | awk '/llama-server|ollama runner/ && !/awk/ {s+=$1} END {print int(s/1024)}')
    (( runner > peak_runner )) && peak_runner=$runner
    evr=$(ps -o rss= --ppid "$pid" -p "$pid" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
    (( evr > peak_eval )) && peak_eval=$evr
    if (( avail < 800 )); then
        echo "WATCHDOG: MemAvailable ${avail} MB, stopping the eval" >> "$SCRATCH/$name.log"
        pkill -INT -P "$pid" 2>/dev/null; kill -INT "$pid" 2>/dev/null
        killed=1; sleep 5; kill -9 "$pid" 2>/dev/null
        break
    fi
    sleep 3
done
wait "$pid" 2>/dev/null; rc=$?
printf '{"name": "%s", "rc": %s, "killed": %s, "min_mem_available_mb": %s, "peak_ollama_runner_mb": %s, "peak_eval_process_mb": %s, "finished": "%s"}\n' \
    "$name" "$rc" "$killed" "$min_avail" "$peak_runner" "$peak_eval" "$(date -Is)" > "$SCRATCH/$name.mem.json"
cat "$SCRATCH/$name.mem.json"
