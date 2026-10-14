#!/usr/bin/env bash
# Usage: run_isolated.sh <venv-python> <script.py> [args...]
# Runs the script in a fresh network namespace (loopback only); Ollama is reachable
# only through a unix-socket bridge, so nothing else can leave the machine.
set -u
S=/tmp/claude-1000/-home-bacancy-Dev-RAG-rag-studio/6a7b83bc-7fb2-4ede-9e78-575e4ebae24d/scratchpad
cd "$S"
rm -f o.sock
socat UNIX-LISTEN:o.sock,fork TCP:127.0.0.1:11434 >/dev/null 2>&1 &
BRIDGE=$!
sleep 0.5
PY=$1; shift
unshare -rn bash -c "
  cd $S
  ip link set lo up
  socat TCP-LISTEN:11434,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:o.sock >/dev/null 2>&1 &
  INNER=\$!
  sleep 0.3
  cd $S/scripts && HOME=$S/fakehome $PY $*
  RC=\$?
  kill \$INNER
  exit \$RC
"
RC=$?
kill $BRIDGE 2>/dev/null
exit $RC
