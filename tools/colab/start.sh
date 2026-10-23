#!/usr/bin/env bash
# =============================================================================
# Bring up a disposable GPU model server for RAG Studio, in Google Colab.
#
#   ollama (127.0.0.1:11434)  <-  server.py (0.0.0.0:8000)  <-  cloudflared
#
# Prints ONE line to run on the laptop: `./rag llm connect <url>` when the
# laptop supplied the token (RAG_STUDIO_TOKEN, which `./rag llm colab` embeds
# in the cell), or a connect code when this runtime made its own token. The
# code is never printed into a notebook's output (Colab saves output to
# Drive): it goes to .rag_code and the setup cell shows it once.
#
#   bash start.sh                                   start everything (default)
#   bash start.sh --models "qwen3-embedding:8b"     serve other models
#   bash start.sh --models "" --worker qwen3-4b     a rerank-only runtime: no Ollama at all
#   bash start.sh --worker qwen3-0.6b-gpu,qwen3-4b  also run the worker (worker.py), these rerankers
#   bash start.sh status                            is it up, and on what URL
#   bash start.sh code                              the connect code again, no restart
#   bash start.sh logs                              the last lines of every log
#   bash start.sh stop                              stop everything
#
# Nothing is copied from the laptop: no documents, no index, no repository.
# Only models, and an HTTP door with a lock on it. Passages arrive one batch
# at a time to be embedded, and nothing keeps them.
#
# Every step checks itself instead of assuming the previous one worked. Two of
# those checks come from real failures on a fresh T4 (found by the trading app,
# 2026-09-10): Ollama's installer needs `zstd`, which Colab lacks, and a
# trycloudflare hostname is not routable the moment cloudflared prints it.
# =============================================================================
set -uo pipefail

MODELS="${MODELS-qwen3-embedding:8b qwen3-embedding:0.6b}"
# Rerankers the worker downloads and serves (comma-separated ids from worker.py; the first is loaded).
WORKER="${WORKER:-}"
WORKER_PORT="${WORKER_PORT:-8001}"
# The laptop's library versions, so the worker computes with the same code:
# "sentence-transformers==X transformers==Y". Empty means whatever pip picks.
PINS="${PINS:-}"
PORT="${PORT:-8000}"
WORK="${WORK:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The laptop's Ollama version, passed by the setup cell, so both sides run the
# same runtime and the local-vs-Colab vector check compares like with like.
# Empty means the latest release.
OLLAMA_PIN="${OLLAMA_PIN:-}"

OLLAMA_LOG="$WORK/ollama.log"
GATEWAY_LOG="$WORK/gateway.log"
WORKER_LOG="$WORK/worker.log"
TUNNEL_LOG="$WORK/cloudflared.log"
INSTALL_LOG="$WORK/install.log"
TOKEN_FILE="$WORK/.rag_token"
URL_FILE="$WORK/.rag_url"
CODE_FILE="$WORK/.rag_code"

OLLAMA_URL="http://127.0.0.1:11434"

say()  { printf "\033[1m%s\033[0m\n" "$*"; }
dim()  { printf "\033[2m%s\033[0m\n" "$*"; }
warn() { printf "\033[33m     %s\033[0m\n" "$*"; }
bad()  { printf "\033[31m     %s\033[0m\n" "$*"; }
good() { printf "\033[32m     %s\033[0m\n" "$*"; }
die()  { printf "\033[31mFATAL: %s\033[0m\n" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

daemon() {
    # Start a long-running service in its OWN session. The notebook's stop
    # button interrupts the cell's whole process group; a service in that group
    # dies with it (nohup only ignores hang-ups, not interrupts). A service in
    # its own session survives the stop, which is what lets a keep-alive cell
    # be stopped without taking the runtime down. Falls back to nohup.
    if have setsid; then setsid nohup "$@"; else nohup "$@"; fi
}


# --- helpers, each testable on its own (tests/test_colab_start.py) ------------------
as_root() {
    # Colab runs as root; a laptop running the same script does not.
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then "$@"
    elif have sudo; then sudo "$@"
    else return 127
    fi
}

ensure_pkg() {
    # ensure_pkg <binary> <apt package>: true if the binary exists afterwards.
    # Install is tried before `apt-get update`, which alone costs ~20 s.
    have "$1" && return 0
    have apt-get || return 1
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$2" \
        >>"$INSTALL_LOG" 2>&1 && have "$1" && return 0
    as_root apt-get update -qq >>"$INSTALL_LOG" 2>&1
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$2" \
        >>"$INSTALL_LOG" 2>&1
    have "$1"
}

http_code() {
    # The status of one request, or 000 when no connection completed. Never
    # fails. curl prints 000 AND exits non-zero on a refused connection, so
    # `curl ... || printf 000` would print 000000; the output is checked.
    local url="$1"; shift
    local out
    out="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$@" "$url" 2>/dev/null)"
    if [[ "$out" =~ ^[0-9]{3}$ ]]; then printf '%s' "$out"; else printf '000'; fi
}

wait_for_http() {
    # wait_for_http <url> <seconds> [acceptable codes...]: polls until one of
    # the codes (or, with none given, any real answer) or the deadline, and
    # echoes the last code seen, so "never connected" (000) differs from
    # "answered, wrong code".
    local url="$1" seconds="$2"; shift 2
    local want=("$@") code="000" deadline=$(( SECONDS + seconds ))
    while :; do
        code="$(http_code "$url")"
        if (( ${#want[@]} == 0 )); then
            [[ "$code" != "000" ]] && break
        else
            local w
            for w in "${want[@]}"; do [[ "$code" == "$w" ]] && break 2; done
        fi
        (( SECONDS >= deadline )) && break
        sleep 2
    done
    printf '%s' "$code"
}

tunnel_verdict() {
    # What an UNAUTHENTICATED probe of the public endpoint means. 000 is
    # "could not verify yet" and survivable; 200 is "open to the internet".
    case "${1:-000}" in
        401|403) printf 'locked';             return 0 ;;
        200)     printf 'OPEN';               return 2 ;;
        000)     printf 'unreachable';        return 1 ;;
        *)       printf 'unexpected:%s' "$1"; return 1 ;;
    esac
}

ollama_up()  { [[ "$(http_code "$OLLAMA_URL/api/version")" == "200" ]]; }
ollama_ver() { curl -s --max-time 5 "$OLLAMA_URL/api/version" 2>/dev/null \
                 | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null; }

model_present() {
    # Is $1 served, under the exact name the laptop will ask for?
    curl -s --max-time 15 "$OLLAMA_URL/api/tags" 2>/dev/null | python3 -c '
import json, sys
try:
    names = [m.get("name", "") for m in (json.load(sys.stdin).get("models") or [])]
except Exception:
    sys.exit(1)
want = sys.argv[1]
sys.exit(0 if want in names or f"{want}:latest" in names else 1)
' "$1"
}

is_embedding_model() {
    # True when Ollama says the model makes embeddings (it then has no chat).
    python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1]}))' "$1" \
        | curl -s --max-time 30 -H 'Content-Type: application/json' -d @- "$OLLAMA_URL/api/show" 2>/dev/null \
        | python3 -c '
import json, sys
try:
    caps = json.load(sys.stdin).get("capabilities") or []
except Exception:
    sys.exit(1)
sys.exit(0 if "embedding" in caps else 1)
'
}

connect_code() {
    # connect_code <url> <token> "<models>" <gateway>: one pasteable string.
    # Mirrors rag_studio.remote.encode_connect_code; tests/test_remote.py runs
    # THIS function and decodes the result with THAT one, so the two cannot
    # drift. It is not encryption: whoever holds the code holds the token.
    python3 -c '
import base64, json, sys, time
url, token, models, gateway = (sys.argv + ["", "", "", ""])[1:5]
payload = {"u": url.rstrip("/")}
if token:   payload["t"] = token
if models.split(): payload["m"] = models.split()
if gateway: payload["g"] = gateway
payload["ts"] = int(time.time())
print("rsc1." + base64.urlsafe_b64encode(
    json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("="))
' "$1" "$2" "$3" "$4"
}

gateway_version() {
    # Read from server.py, so the two cannot disagree.
    sed -n 's/^VERSION = "\(.*\)"/\1/p' "$HERE/server.py" 2>/dev/null | head -1
}

stash_code() {
    # stash_code <connect code> -> $CODE_FILE, readable by this user only.
    ( umask 077 && printf '%s\n' "$1" > "$CODE_FILE" )
}

handoff_banner() {
    # handoff_banner <url> <token> "<models>" <token source: laptop|reused|generated>
    #
    # The one place a credential could reach this script's output, so the
    # rule lives here: the raw token is never printed, and the connect code
    # (which carries it) is printed only to a terminal. A cell's output is
    # saved with the notebook in Drive; there the code goes to $CODE_FILE and
    # the setup cell shows it in an output it blanks once you press Enter.
    local url="$1" token="$2" models="$3" source="$4" code
    cat <<HEAD

===============================================================================
  Endpoint   $url
  Models     $models
  Token      kept in $TOKEN_FILE on this runtime, never printed

  ON THE LAPTOP, copy this one line:

HEAD
    if [[ "$source" == "laptop" ]]; then
        rm -f "$CODE_FILE"
        cat <<LAPTOP
      ./rag llm connect $url

  (The laptop already holds this runtime's token, so only the URL is new.
  In the app: Settings > Model backend > Connect, and paste the same URL.)
LAPTOP
    else
        code="$(connect_code "$url" "$token" "$models" "$(gateway_version)")"
        if [[ -t 1 ]]; then
            cat <<SHOWN
      ./rag llm connect $code

  (One string with the URL and the token this runtime made. In the app,
  paste just the code: Settings > Model backend > Connect.)
SHOWN
        else
            stash_code "$code"
            cat <<STASHED
      ./rag llm connect <connect code>

  The connect code carries this runtime's token, so it is not printed here:
  this output is saved with the notebook. It is in $CODE_FILE (mode 600).
  The setup cell shows it below and blanks it once you press Enter;
  show_connect_code() shows it again the same way.
STASHED
        fi
    fi
    cat <<TAIL

  Then, on the laptop:   ./rag llm test
  When you are done:     ./rag llm disconnect

  This runtime is disposable. Its URL dies with it (Colab reclaims an idle
  runtime after about 90 minutes); re-running this cell is the whole
  reconnect. A build that was cut off resumes where it stopped.
===============================================================================
TAIL
}

gpu_residency() {
    # "<vram bytes> <total bytes>" for one loaded model, matched by its exact
    # name (qwen3-embedding:0.6b and :8b share a family), or nothing. Always
    # exits 0, so a caller's $(...) never inherits curl's failure.
    local body
    body="$(curl -s --max-time 10 "$OLLAMA_URL/api/ps" 2>/dev/null || true)"
    [[ -n "$body" ]] || return 0
    printf '%s' "$body" | python3 -c '
import json, sys
try:
    models = json.load(sys.stdin).get("models") or []
except Exception:
    sys.exit(0)
want = sys.argv[1] if ":" in sys.argv[1] else sys.argv[1] + ":latest"
for m in models:
    name = m.get("name", "")
    if (name if ":" in name else name + ":latest") == want:
        print(m.get("size_vram", 0), m.get("size", 0))
        break
' "$1" || true
}

load_model() {
    # One tiny request, so the weights leave the disk for wherever they are
    # going to live, and the first real request does not pay the load.
    if is_embedding_model "$1"; then
        python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "input": ["hi"], "keep_alive": "30m"}))' "$1" \
            | curl -s --max-time 600 -o /dev/null -H 'Content-Type: application/json' -d @- "$OLLAMA_URL/api/embed"
    else
        python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "stream": False, "think": False, "keep_alive": "30m",
                  "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 1}}))
' "$1" | curl -s --max-time 600 -o /dev/null -H 'Content-Type: application/json' -d @- "$OLLAMA_URL/api/chat"
    fi
}

release_model() {
    # Free a model's GPU memory now instead of after the 30-minute keep-alive. The startup load is
    # a check, not a warm-up: two vision models left resident (6.2 + 8.0 GB) left the worker 60 MB
    # of a T4, and it died loading a 1.2 GB reranker (2026-10-23). The first real request reloads it.
    ollama stop "$1" >/dev/null 2>&1 || true
}

release_all_models() {
    local names
    names="$(curl -s --max-time 10 "$OLLAMA_URL/api/ps" 2>/dev/null | python3 -c '
import json, sys
try:
    print(" ".join(m.get("name", "") for m in json.load(sys.stdin).get("models") or []))
except Exception:
    pass
' || true)"
    for n in $names; do release_model "$n"; done
}

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
}

stop_all() {
    say "Stopping"
    pkill -f cloudflared 2>/dev/null && dim "  stopped cloudflared"
    pkill -f "$HERE/server.py" 2>/dev/null && dim "  stopped the gateway"
    pkill -f "$HERE/worker.py" 2>/dev/null && dim "  stopped the worker"
    pkill -f "ollama serve" 2>/dev/null && dim "  stopped ollama"
    rm -f "$URL_FILE"
    dim "  the token in $TOKEN_FILE is kept; delete it to make a new one"
}


# Sourced for its functions (the tests do this), not run.
if [[ -n "${RAG_COLAB_LIB:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi


# --- subcommands -------------------------------------------------------------------------
cmd="start"
while (( $# )); do
    case "$1" in
        start|stop|status|logs|code) cmd="$1"; shift ;;
        --models) MODELS="$2"; shift 2 ;;
        --worker) WORKER="$2"; shift 2 ;;
        --pins)   PINS="$2";   shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        --ollama-version) OLLAMA_PIN="$2"; shift 2 ;;
        --help|-h)
            sed -n '/^# =====/,/^# =====/p' "${BASH_SOURCE[0]}" | sed '1d;$d;s/^# \?//'
            exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

case "$cmd" in
    stop) stop_all; exit 0 ;;
    code)
        [[ -f "$URL_FILE" ]]   || die "no tunnel URL on record. Run: bash start.sh"
        [[ -f "$TOKEN_FILE" ]] || die "no token on record. Run: bash start.sh"
        CONNECT="$(connect_code "$(cat "$URL_FILE")" "$(cat "$TOKEN_FILE")" "$MODELS" "$(gateway_version)")"
        if [[ -t 1 ]]; then
            printf './rag llm connect %s\n' "$CONNECT"
        else
            stash_code "$CONNECT"
            printf 'The connect code is in %s (mode 600), not printed: this output\n' "$CODE_FILE"
            printf 'would be saved with the notebook. Run show_connect_code() in the\n'
            printf 'notebook that started this runtime, or run this in a terminal.\n'
        fi
        exit 0 ;;
    logs) tail -n 40 "$OLLAMA_LOG" "$WORKER_LOG" "$GATEWAY_LOG" "$TUNNEL_LOG" 2>/dev/null; exit 0 ;;
    status)
        say "Status"
        pgrep -f "ollama serve"     >/dev/null && dim "  ollama      running" || dim "  ollama      STOPPED"
        pgrep -f "$HERE/server.py"  >/dev/null && dim "  gateway     running" || dim "  gateway     STOPPED"
        if [[ -n "$WORKER" ]]; then
            pgrep -f "$HERE/worker.py" >/dev/null && dim "  worker      running" || dim "  worker      STOPPED"
            curl -s --max-time 5 "http://127.0.0.1:$WORKER_PORT/worker" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print("  worker      loaded:", d.get("loaded"), "| serves:", ", ".join(d.get("served") or []), "| vram:", d.get("vram"))
except Exception:
    print("  worker      not answering")' 2>/dev/null
        fi
        pgrep -f "cloudflared"      >/dev/null && dim "  cloudflared running" || dim "  cloudflared STOPPED"
        for M in $MODELS; do
            RES="$(gpu_residency "$M")"
            if [[ -n "$RES" ]]; then
                read -r VRAM TOTAL <<< "$RES"
                if (( VRAM > 0 )); then dim "  $M: loaded, $(( VRAM * 100 / (TOTAL > 0 ? TOTAL : VRAM) ))% on the GPU"
                else dim "  $M: loaded, on the CPU"; fi
            else
                dim "  $M: not loaded right now"
            fi
        done
        if [[ -f "$URL_FILE" ]]; then
            URL="$(cat "$URL_FILE")"
            dim "  endpoint    $URL  (unauthenticated probe: $(http_code "$URL/api/tags" ))"
        fi
        curl -s --max-time 5 "http://127.0.0.1:$PORT/healthz" && echo
        exit 0 ;;
esac

: > "$INSTALL_LOG"

# --- 1. is there a GPU -------------------------------------------------------------------
say "1/7  GPU"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/     /'
else
    warn "No GPU on this runtime."
    dim "     Runtime > Change runtime type > T4 GPU, then run this cell again."
    dim "     Continuing anyway: it works on the CPU, slowly, which is what the laptop already does."
fi

# --- 2. prerequisites --------------------------------------------------------------------
say "2/7  Prerequisites"
if have zstd; then
    dim "     zstd already present"
elif ensure_pkg zstd zstd; then
    good "installed zstd (Ollama's installer needs it)"
else
    warn "could not install zstd. See $INSTALL_LOG"
    dim "     Ollama's installer needs it; if step 3 fails, that is why."
fi
have curl || die "curl is missing, and nothing here works without it"

# --- 3. ollama ---------------------------------------------------------------------------
# Skipped with --models "": a rerank-only runtime needs no Ollama, and not
# installing, starting and loading one saves minutes and GPU memory.
if [[ -z "${MODELS// /}" ]]; then
    say "3/7  Model server"
    dim "     no Ollama models asked for: skipped"
    SKIP_OLLAMA=1
else
SKIP_OLLAMA=0
say "3/7  Model server"
if have ollama && [[ -n "$OLLAMA_PIN" ]] && ! ollama -v 2>/dev/null | grep -q "$OLLAMA_PIN"; then
    dim "     the installed Ollama is not $OLLAMA_PIN; reinstalling that version"
    pkill -f "ollama serve" 2>/dev/null
    INSTALL_OLLAMA=1
elif ! have ollama; then
    INSTALL_OLLAMA=1
else
    INSTALL_OLLAMA=0
fi
if (( INSTALL_OLLAMA )); then
    dim "     installing Ollama ${OLLAMA_PIN:-(latest)} ..."
    if ! curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_PIN" sh >>"$INSTALL_LOG" 2>&1; then
        tail -n 15 "$INSTALL_LOG" >&2
        die "could not install Ollama. Full log in $INSTALL_LOG"
    fi
    have ollama || die "the Ollama installer finished but there is no ollama on PATH"
fi
if ! pgrep -f "ollama serve" >/dev/null; then
    # Loopback only: the gateway is the only thing that may reach it. Models
    # stay loaded for 30 minutes, so the gaps between an index build's
    # batches, or an evaluation's questions, never pay a reload.
    # One request at a time per model: more slots multiply the KV cache and
    # break the ordered prefix reuse that context lines and rewrites rely on.
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=1 daemon ollama serve >"$OLLAMA_LOG" 2>&1 &
fi
wait_for_http "$OLLAMA_URL/api/version" 60 200 >/dev/null
ollama_up || die "Ollama did not come up. See $OLLAMA_LOG"
good "Ollama $(ollama_ver) on 127.0.0.1:11434"
if [[ -n "$OLLAMA_PIN" && "$(ollama_ver)" != "$OLLAMA_PIN" ]]; then
    warn "asked for Ollama $OLLAMA_PIN but $(ollama_ver) is running; the vector check will say whether it matters"
fi

# --- 4. the models -----------------------------------------------------------------------
say "4/7  Models: $MODELS"
dim "     Each runtime starts empty, so every session downloads them again (usually 1-3 minutes)."
for M in $MODELS; do
    ollama pull "$M" >>"$INSTALL_LOG" 2>&1 || { tail -n 5 "$INSTALL_LOG" >&2; die "could not pull $M. Check the name with: ollama list"; }
    model_present "$M" || die "$M was pulled but is not in /api/tags under that name"
    load_model "$M"
    RESIDENCY="$(gpu_residency "$M")"
    if [[ -z "$RESIDENCY" ]]; then
        warn "$M: could not read /api/ps, so where it runs is unknown"
    else
        read -r VRAM TOTAL <<< "$RESIDENCY"
        if (( VRAM <= 0 )); then
            bad "$M is running on the CPU, not the GPU."
            dim "     Runtime > Change runtime type > T4 GPU, then: bash start.sh stop && bash start.sh"
        elif (( TOTAL > 0 && VRAM < TOTAL )); then
            warn "$M: only $(( VRAM * 100 / TOTAL ))% fits in GPU memory; the rest runs on the CPU."
        else
            good "$M: on the GPU ($(( VRAM / 1024 / 1024 )) MB)"
        fi
    fi
    # Checked alone, then released: every model gets the whole GPU for its check, and nothing is
    # left holding memory the next model or the worker needs.
    release_model "$M"
done

fi   # models asked for

# --- 4b. the worker ---------------------------------------------------------------------
if [[ -n "$WORKER" ]]; then
    say "4b   Worker: $WORKER"
    [[ -f "$HERE/worker.py" ]] || die "worker.py is not next to this script ($HERE)"
    # "context" is a job, not a reranker: it turns on POST /context (through the local Ollama).
    CONTEXT_FLAG=(); RERANKERS="$WORKER"
    if [[ ",$WORKER," == *",context,"* ]]; then
        CONTEXT_FLAG=(--context)
        RERANKERS="$(printf '%s' "$WORKER" | tr ',' '\n' | grep -vx context | paste -sd, -)"
        (( SKIP_OLLAMA )) && die "the context job needs Ollama with qwen3:8b: start with --models qwen3:8b"
    fi
    if [[ -n "$RERANKERS" ]]; then
    dim "     installing pinned libraries (${PINS:-unpinned}) ..."
    NEEDS_BNB=""; [[ "$RERANKERS" == *8b* ]] && NEEDS_BNB="bitsandbytes"
    # shellcheck disable=SC2086
    python3 -m pip install -q $PINS accelerate $NEEDS_BNB >>"$INSTALL_LOG" 2>&1 \
        || { tail -n 15 "$INSTALL_LOG" >&2; die "could not install the worker's libraries. See $INSTALL_LOG"; }
    python3 -c 'import torch, sentence_transformers as s; print("     torch", torch.__version__, "| sentence-transformers", s.__version__, "| cuda", torch.cuda.is_available())' \
        || die "torch or sentence-transformers cannot be imported after the install"
    fi
    pkill -f "$HERE/worker.py" 2>/dev/null
    # Ollama does not know the worker exists: anything it still holds (an earlier run's models, kept
    # 30 minutes) is GPU memory the worker cannot have. Release it all before the worker loads.
    (( SKIP_OLLAMA )) || release_all_models
    sleep 2
    FREE_MB="$(gpu_free_mb || true)"
    dim "     GPU memory free before the worker loads: ${FREE_MB:-?} MiB"
    # Fail before the downloads, not after them: the worker downloads every reranker first and only
    # then loads one, so running out of memory there wastes the whole download (2026-10-23).
    if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB < 3000 )); then
        nvidia-smi >&2 || true
        die "only ${FREE_MB} MiB of GPU memory is free, and the worker needs about 3000 MiB for its first reranker. Something else holds the GPU (see above)."
    fi
    WORKER_PORT="$WORKER_PORT" daemon python3 "$HERE/worker.py" --port "$WORKER_PORT" --models "$RERANKERS" ${CONTEXT_FLAG[@]+"${CONTEXT_FLAG[@]}"} >"$WORKER_LOG" 2>&1 &
    dim "     downloading and loading the weights (minutes the first time; the log is $WORKER_LOG) ..."
    WCODE="$(wait_for_http "http://127.0.0.1:$WORKER_PORT/worker" 1800 200)"
    [[ "$WCODE" == "200" ]] || { tail -n 20 "$WORKER_LOG" >&2; die "the worker did not come up (HTTP $WCODE). See $WORKER_LOG"; }
    # A real request on loopback: two documents, and the answer must rank the
    # relevant one first. A worker that loads but cannot score fails here.
    FIRST="${RERANKERS%%,*}"
    if [[ -z "$FIRST" ]]; then
        PROBE="ok context-only (no reranker to probe)"
    else
    PROBE="$(python3 - "$FIRST" "$WORKER_PORT" <<'PY'
import json, sys, urllib.request
model, port = sys.argv[1], sys.argv[2]
body = {"model": model, "instruction": "Given a question about reports, retrieve the passage that states the answer",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital and largest city of France.", "Bananas are rich in potassium."]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/rerank", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    out = json.load(urllib.request.urlopen(req, timeout=120))
    s = out["scores"]
    print("ok" if s[0] > s[1] else "wrong-order", f"{out['compute_ms']:.0f}ms", f"peak {out['max_memory_mb']}MB")
except Exception as e:
    print("failed", type(e).__name__)
PY
)"
    fi
    case "$PROBE" in
        ok*) good "worker scored a real request on loopback: $PROBE" ;;
        *)   tail -n 20 "$WORKER_LOG" >&2; die "the worker's loopback check said: $PROBE" ;;
    esac
fi

# --- 5. the token ------------------------------------------------------------------------
say "5/7  Token"
TOKEN_SOURCE="generated"
if [[ -n "${RAG_STUDIO_TOKEN:-}" ]]; then
    # Embedded by `./rag llm colab`: the laptop already holds it, so a
    # reconnect needs only the new tunnel URL.
    TOKEN="$RAG_STUDIO_TOKEN"
    TOKEN_SOURCE="laptop"
    dim "     using RAG_STUDIO_TOKEN from the cell (the laptop knows it)"
    ( umask 077 && printf '%s' "$TOKEN" > "$TOKEN_FILE" )
elif [[ -f "$TOKEN_FILE" ]]; then
    TOKEN="$(cat "$TOKEN_FILE")"
    TOKEN_SOURCE="reused"
    dim "     reusing the token in $TOKEN_FILE"
else
    # Made HERE, on the machine that enforces it, so it travels only once:
    # inside the connect code.
    TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    ( umask 077 && printf '%s' "$TOKEN" > "$TOKEN_FILE" )
    dim "     made a new 48-character token"
fi
[[ -n "$TOKEN" ]] || die "the token is empty; refusing to start an unlocked gateway"

# --- 6. the gateway ----------------------------------------------------------------------
say "6/7  Gateway"
[[ -f "$HERE/server.py" ]] || die "server.py is not next to this script ($HERE)"
pkill -f "$HERE/server.py" 2>/dev/null
WAIT_OLLAMA=(--wait-for-ollama 60); (( SKIP_OLLAMA )) && WAIT_OLLAMA=()
RAG_STUDIO_TOKEN="$TOKEN" WORKER_URL="http://127.0.0.1:$WORKER_PORT" daemon python3 "$HERE/server.py" \
    --port "$PORT" "${WAIT_OLLAMA[@]}" >"$GATEWAY_LOG" 2>&1 &
wait_for_http "http://127.0.0.1:$PORT/healthz" 45 200 >/dev/null
[[ "$(http_code "http://127.0.0.1:$PORT/healthz")" == "200" ]] \
    || { tail -n 15 "$GATEWAY_LOG" >&2; die "the gateway did not come up. See $GATEWAY_LOG"; }

# Prove the lock HERE, on loopback, before any tunnel exists: if it fails the
# tunnel is never opened, so a broken lock can never be exposed.
LOCK_PATH="/api/tags"; (( SKIP_OLLAMA )) && LOCK_PATH="/worker"
LOCAL_UNAUTH="$(http_code "http://127.0.0.1:$PORT$LOCK_PATH")"
LOCAL_AUTH="$(http_code "http://127.0.0.1:$PORT$LOCK_PATH" -H "Authorization: Bearer $TOKEN")"
[[ "$LOCAL_UNAUTH" == "401" ]] \
    || die "the gateway answered a request without a token with $LOCAL_UNAUTH, not 401. Refusing to expose it."
[[ "$LOCAL_AUTH" == "200" ]] \
    || { tail -n 15 "$GATEWAY_LOG" >&2
         die "the gateway refused its own token, or cannot reach Ollama (got $LOCAL_AUTH)"; }
good "listening on 0.0.0.0:$PORT: without the token 401, with it 200"

# --- 7. the tunnel -----------------------------------------------------------------------
say "7/7  Tunnel"
if ! have cloudflared; then
    dim "     installing cloudflared ..."
    as_root curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        >>"$INSTALL_LOG" 2>&1 || die "could not download cloudflared. See $INSTALL_LOG"
    as_root chmod +x /usr/local/bin/cloudflared
    have cloudflared || die "cloudflared downloaded but is not on PATH"
fi
pkill -f cloudflared 2>/dev/null
rm -f "$TUNNEL_LOG"
daemon cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" >"$TUNNEL_LOG" 2>&1 &

URL=""
for _ in $(seq 1 40); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)"
    [[ -n "$URL" ]] && break
    sleep 1
done
[[ -n "$URL" ]] || { tail -n 15 "$TUNNEL_LOG" >&2; die "cloudflared never printed a URL. See $TUNNEL_LOG"; }
printf '%s' "$URL" > "$URL_FILE"

# A quick-tunnel URL is unguessable, which is obscurity, not security: prove
# the lock reaches the public address before printing it. cloudflared prints
# the name before DNS carries it, so wait for a real answer, then judge it.
dim "     waiting for Cloudflare to route $URL ..."
CODE="$(wait_for_http "$URL$LOCK_PATH" 120 401 403)"
VERDICT="$(tunnel_verdict "$CODE")"
case "$VERDICT" in
    locked)
        good "verified: the public endpoint refuses a request without the token ($CODE)" ;;
    OPEN)
        pkill -f cloudflared 2>/dev/null
        rm -f "$URL_FILE"
        bad "SECURITY: $URL answered a request without the token (200)."
        bad "The tunnel is shut down and the URL is deliberately not printed."
        dim "     The gateway itself is locked (step 6 proved it on loopback), so something"
        dim "     else answered on that hostname. Run this cell again."
        exit 1 ;;
    unreachable)
        warn "could not reach $URL from this runtime within 120 s (curl: 000)."
        dim "     That is Cloudflare still routing the name, not a broken lock: step 6 proved"
        dim "     the lock on loopback. It is usually live within a minute. Check with:"
        dim "         bash start.sh status      # here"
        dim "         ./rag llm test            # on the laptop" ;;
    *)
        warn "a request without the token to $URL returned $CODE, expected 401."
        dim "     Step 6 proved the lock on loopback, so this is the tunnel answering for"
        dim "     something else. Run the cell again if it persists." ;;
esac

[[ -n "$WORKER" ]] && dim "Worker (rerankers served): $WORKER"
handoff_banner "$URL" "$TOKEN" "$MODELS" "$TOKEN_SOURCE"
