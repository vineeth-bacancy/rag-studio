#!/usr/bin/env bash
# Runs INSIDE `unshare -rn`. $1 = "default" or "lockdown"
S=/tmp/claude-1000/-home-bacancy-Dev-RAG-rag-studio/6a7b83bc-7fb2-4ede-9e78-575e4ebae24d/scratchpad
ip link set lo up
export HOME=$S/fakehome
export PHOENIX_WORKING_DIR=$S/phx-$1
rm -rf "$PHOENIX_WORKING_DIR"; mkdir -p "$PHOENIX_WORKING_DIR"
if [ "$1" = "lockdown" ]; then
  export PHOENIX_HOST=127.0.0.1 PHOENIX_TELEMETRY_ENABLED=false PHOENIX_ALLOW_EXTERNAL_RESOURCES=false
fi
strace -f -qq -e trace=connect -o $S/phx-$1.strace $S/venvs/phoenix-env/bin/phoenix serve > $S/phx-$1.log 2>&1 &
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:6006/healthz && break; sleep 1; done
echo "startup secs ~ $i"
PID=$(pgrep -f "phoenix-env/bin/phoenix serve" | tail -1)
echo "listening:"; ss -ltnp 2>/dev/null | awk 'NR>1{print "  "$4}'
echo "RSS after start (MB): $(( $(ps -o rss= -p $PID) / 1024 ))"
curl -s http://127.0.0.1:6006/ | grep -oE '(https?:)?//[a-zA-Z0-9.-]+\.[a-z]{2,}[^"'"'"' )]*' | sort -u | head -20 | sed 's/^/  ext-ref in index.html: /'
cd $S/scripts && $S/venvs/phoenix-env/bin/python phoenix_client.py 2>&1 | tail -5
echo "RSS after ingest (MB): $(( $(ps -o rss= -p $PID) / 1024 ))"
ls -la $PHOENIX_WORKING_DIR | sed 's/^/  /'
kill $PID; sleep 1
echo "non-loopback connect attempts (strace):"
grep -E 'connect\(' $S/phx-$1.strace | grep -vE 'AF_UNIX|127\.0\.0\.1|inet_addr\("127' | grep -oE 'sin6?_addr=[^,)]*|sin_port=htons\([0-9]+\)' | sort | uniq -c | head
grep -E 'htons\(53\)' $S/phx-$1.strace | wc -l | sed 's/^/  DNS connects: /'
