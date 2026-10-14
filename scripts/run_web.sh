#!/usr/bin/env bash
# =============================================================================
# Start the app: the Python API and the Next.js UI, both on 127.0.0.1 only.
#
#   ./rag web           dev mode (hot reload)
#   ./rag web --prod    build once, then serve the built app
#
# Nothing here is reachable from another machine, which is why there is no
# login. Ctrl-C stops both.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
UI="$ROOT/web/ui"
API_PORT="${RAG_STUDIO_API_PORT:-8775}"
UI_PORT="${RAG_STUDIO_UI_PORT:-3100}"
# Exported so the API's request guard trusts exactly these two origins.
export RAG_STUDIO_API_PORT="$API_PORT" RAG_STUDIO_UI_PORT="$UI_PORT"
MODE="dev"
[[ "${1:-}" == "--prod" ]] && MODE="prod"

if ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Error: the web dependencies are not installed." >&2
    echo "Run: cd $ROOT && uv sync" >&2
    exit 1
fi
if [[ ! -d "$UI/node_modules" ]]; then
    echo "Installing frontend dependencies (first run only)..."
    (cd "$UI" && pnpm install) || { echo "Error: pnpm install failed." >&2; exit 1; }
fi

for port in "$API_PORT" "$UI_PORT"; do
    if curl -s -o /dev/null "http://127.0.0.1:$port" 2>/dev/null; then
        echo "Error: port $port is already in use. Is RAG Studio already running?" >&2
        exit 1
    fi
done

cleanup() {
    echo ""
    echo "Stopping..."
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null
    [[ -n "${UI_PID:-}"  ]] && kill "$UI_PID"  2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# Dev mode reloads the API when the engine or the routes change, as Next does
# for the UI. Only those two folders are watched: data/ changes during ingest.
RELOAD=()
[[ "$MODE" == "dev" ]] && RELOAD=(--reload --reload-dir "$ROOT/rag_studio" --reload-dir "$ROOT/web/api")

echo "Starting the app service on 127.0.0.1:$API_PORT..."
"$PY" -m uvicorn main:app --app-dir "$ROOT/web/api" \
      --host 127.0.0.1 --port "$API_PORT" --log-level warning "${RELOAD[@]}" &
API_PID=$!

# Wait for the API rather than racing it: the UI's first paint calls /api/status.
for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$API_PORT/api/status" >/dev/null 2>&1 && break
    sleep 0.25
done
if ! curl -sf "http://127.0.0.1:$API_PORT/api/status" >/dev/null 2>&1; then
    echo "Error: the app service did not start. Run it directly to see why:" >&2
    echo "  $PY -m uvicorn main:app --app-dir $ROOT/web/api --port $API_PORT" >&2
    exit 1
fi

cd "$UI" || exit 1
export RAG_STUDIO_API_URL="http://127.0.0.1:$API_PORT"

# --hostname is not optional. Next binds 0.0.0.0 by default, and next.config.ts
# rewrites /api/* to the API, so a UI on a routable interface would be an open
# proxy to an API that deliberately binds loopback only.
if [[ "$MODE" == "prod" ]]; then
    echo "Building the UI (about a minute)..."
    pnpm build || { echo "Error: the UI build failed." >&2; exit 1; }
    pnpm exec next start --hostname 127.0.0.1 --port "$UI_PORT" &
else
    pnpm exec next dev --hostname 127.0.0.1 --port "$UI_PORT" &
fi
UI_PID=$!

echo ""
echo "  RAG Studio  ->  http://localhost:$UI_PORT"
echo "  UI 127.0.0.1:$UI_PORT · API 127.0.0.1:$API_PORT · $MODE mode · Ctrl-C stops both"
echo ""

wait
