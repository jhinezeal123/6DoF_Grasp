#!/usr/bin/env bash
#
# serve_llamacpp.sh - Chay llama-server cho PhysBrain1.5-8B.
#
#   bash scripts/serve_llamacpp.sh            # chay nen, ghi log
#   bash scripts/serve_llamacpp.sh stop       # tat
#   bash scripts/serve_llamacpp.sh smoke      # thu van ban + kiem tra vision
#
# Cong 8081, KHONG phai 8080: 8080 la web UI cua SDK.
#
# Vi sao can ca -m lan --mmproj: -m la phan ngon ngu, --mmproj la tower thi giac.
# Thieu --mmproj thi server van len, van tra loi chu, nhung MU ANH - no se tra
# loi nhu mot LLM van ban thuan va moi thu lien quan toi point_2d deu vo nghia.
# Vi vay `smoke` kiem tra ca dong nap mmproj trong log, khong chi hoi van ban.
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

# Ban CUDA nam o build-cuda/, ban CPU o build/. Uu tien CUDA.
BIN=""
for c in "$SRC/build-cuda/bin/llama-server" "$SRC/build/bin/llama-server"; do
  [ -x "$c" ] && { BIN="$c"; break; }
done

stop() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
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
    | head -c 500
  echo; echo
  echo "=== 2. mmproj / vision da nap chua? ==="
  if grep -qiE 'mmproj|clip_|vision|image' "$LOG" 2>/dev/null; then
    grep -iE 'mmproj|clip_|vision|image' "$LOG" | head -8
  else
    echo "  CANH BAO: khong thay dong nao ve vision -> model co the dang MU ANH"
  fi
  echo
  echo "=== 3. backend nao (CUDA hay CPU)? ==="
  grep -iE 'CUDA|device|backend|offload' "$LOG" 2>/dev/null | head -8
}

case "${1:-start}" in
  stop) stop; exit 0 ;;
  smoke) smoke; exit 0 ;;
esac

[ -n "$BIN" ] || { echo "chua build llama-server (khong thay build-cuda/ lan build/)" >&2; exit 1; }
[ -s "$MODELS/$QUANT" ]  || { echo "thieu weights: $MODELS/$QUANT" >&2; exit 1; }
[ -s "$MODELS/$MMPROJ" ] || { echo "thieu mmproj: $MODELS/$MMPROJ  (model se MU ANH)" >&2; exit 1; }

echo "binary: $BIN"
case "$BIN" in *build-cuda*) echo "backend: CUDA";; *) echo "backend: CPU (khong co ban CUDA)";; esac

stop >/dev/null 2>&1
mkdir -p "$LOGS"

# --image-min-tokens 1024: KHONG duoc bo.
# llama.cpp canh bao luc nap model:
#   "Qwen-VL models require at minimum 1024 image tokens to function correctly
#    on grounding tasks"  (ggml-org/llama.cpp issue 16842)
# Test 2 CAN grounding (model tra point_2d), nen day la dieu kien bat buoc,
# khong phai toi uu. Bo co nay thi model van chay, van tra loi, nhung toa do
# diem no tra ve se lech - dung kieu sai im lang.
setsid nohup "$BIN" \
  -m "$MODELS/$QUANT" \
  --mmproj "$MODELS/$MMPROJ" \
  --image-min-tokens "${PHYSBRAIN_MIN_TOKENS:-1024}" \
  --host 0.0.0.0 --port "$PORT" \
  -c "$CTX" -ngl "$NGL" \
  > "$LOG" 2>&1 < /dev/null &

echo "dang cho llama-server len (log: $LOG)..."
for i in $(seq 1 90); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "llama-server SAN SANG sau $((i*2))s  ->  http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 2
done
echo "KHONG len duoc sau 180s. 30 dong cuoi log:" >&2
tail -30 "$LOG" >&2
exit 1
