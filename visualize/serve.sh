#!/usr/bin/env bash
# Serve visualize/ over HTTP, detached, so the page outlives the shell that started it.
#   ./serve.sh start|status|stop|restart      (env: PORT=8099)
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8099}"
PIDFILE="$DIR/.serve.pid"
LOGFILE="$DIR/.serve.log"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python

_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }
_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

case "${1:-start}" in
  start)
    _running && { echo "already running (pid $(cat "$PIDFILE")) on port $PORT"; exit 0; }
    ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "port $PORT in use"; exit 1; }
    setsid bash -c "echo \$\$ > '$PIDFILE'; cd '$DIR' && exec '$PY' -m http.server '$PORT' \
      --bind 0.0.0.0 > '$LOGFILE' 2>&1" < /dev/null &
    sleep 1
    echo "serving $DIR  ->  http://$(_ip):$PORT/index.html  (pid $(cat "$PIDFILE" 2>/dev/null))"
    ;;
  stop)
    _running && kill "$(cat "$PIDFILE")" && echo "stopped" || echo "not running"
    rm -f "$PIDFILE" ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status)
    _running && echo "RUNNING pid $(cat "$PIDFILE")" || echo "STOPPED"
    ss -ltn 2>/dev/null | grep -q ":$PORT " \
      && echo "http://$(_ip):$PORT/index.html" || echo "port $PORT not listening" ;;
  *) echo "usage: $0 {start|stop|restart|status}  (env: PORT)"; exit 1 ;;
esac
