#!/usr/bin/env bash
#
# serve_llamacpp.sh - Chay llama-server cho PhysBrain1.5-8B.
#
#   bash scripts/serve_llamacpp.sh            # chay nen, ghi log
#   bash scripts/serve_llamacpp.sh stop       # tat
#   bash scripts/serve_llamacpp.sh smoke      # thu 1 cau van ban + 1 anh
#
# Cong 8081, KHONG phai 8080: 8080 la web UI cua SDK.
#
# Vi sao can ca -m lan --mmproj: -m la phan ngon ngu, --mmproj la tower thi giac.
# Thieu --mmproj thi server van len, van tra loi chu, nhung khong nhin duoc anh.
set -uo pipefail

BASE="${PHYSBRAIN_BASE:-/home/ktmt-agx-xv/Data/khoanhd/PhysBrain}"
SRC="$BASE/llama.cpp"
MODELS="$BASE/models"
LOGS="$BASE/logs"
QUANT="${PHYSBRAIN_QUANT:-PhysBrain1.5-8B.i1-Q4_K_M.gguf}"
MMPROJ="${PHYSBRAIN_MMPROJ:-PhysBrain1.5-8B.mmproj-f16.gguf}"
PORT="${PHYSBRAIN_PORT:-8081}"
CTX="${PHYSBRAIN_CTX:-8192}"
NGL="${PHYSBRAIN_NGL:-99}"
LOG="$LOGS/serve.log"

# Anh primary 720p -> Qwen3-VL cat patch 16, merge 2 -> ~900 token thi giac.
# CTX 8192 du cho 1 anh + prompt + cau tra loi ngan, va con du cho vai luot.

stop() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  pkill -f "llama-server -m" 2>/dev/null
  sleep 2
  if ss -ltn 2>/dev/null | grep -q ":$PORT"; then
    echo "canh bao: cong $PORT van mo"
  else
    echo "da tat llama-server"
  fi
}

smoke() {
  echo "=== 1. van ban thuan ==="
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16}' \
    | head -c 600
  echo
  echo "=== 2. danh sach model ==="
  curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400
  echo
  echo "=== 3. co nhan anh khong? (kiem tra mmproj da nap) ==="
  grep -iE 'mmproj|clip|vision' "$LOG" 2>/dev/null | tail -5 || echo "  (khong thay dong nao ve vision trong log)"
}

case "${1:-start}" in
  stop) stop; exit 0 ;;
  smoke) smoke; exit 0 ;;
esac

[ -x "$SRC/build/bin/llama-server" ] || { echo "chua build: $SRC/build/bin/llama-server" >&2; exit 1; }
[ -s "$MODELS/$QUANT" ]  || { echo "thieu weights: $MODELS/$QUANT" >&2; exit 1; }
[ -s "$MODELS/$MMPROJ" ] || { echo "thieu mmproj: $MODELS/$MMPROJ  (model se MU ANH)" >&2; exit 1; }

stop >/dev/null 2>&1
mkdir -p "$LOGS"

setsid nohup "$SRC/build/bin/llama-server" \
  -m "$MODELS/$QUANT" \
  --mmproj "$MODELS/$MMPROJ" \
  --host 0.0.0.0 --port "$PORT" \
  -c "$CTX" -ngl "$NGL" \
  > "$LOG" 2>&1 < /dev/null &

echo "dang cho llama-server len (log: $LOG)..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "llama-server SAN SANG sau $((i*2))s  ->  http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 2
done
echo "KHONG len duoc sau 120s. 30 dong cuoi log:" >&2
tail -30 "$LOG" >&2
exit 1
