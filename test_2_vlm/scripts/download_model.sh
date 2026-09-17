#!/usr/bin/env bash
#
# download_model.sh - Tai PhysBrain1.5-8B ve Jetson.
#
#   bash scripts/download_model.sh
#
# HAI file, va chung nam o HAI REPO KHAC NHAU - day la cho de sai nhat:
#
#   weights : mradermacher/PhysBrain1.5-8B-i1-GGUF   (ban imatrix, ban chon)
#   mmproj  : mradermacher/PhysBrain1.5-8B-GGUF      (ban tinh)
#
# Repo "-i1-" KHONG chua file mmproj nao. Tai thieu mmproj thi model chay duoc
# nhung MU ANH - no se tra loi nhu mot LLM van ban thuan, va moi thu lien quan
# toi point_2d deu vo nghia.
#
# Kich thuoc: weights 5.2 GB, mmproj ~1.3 GB. curl -C - de tai tiep neu dut.
set -uo pipefail

BASE="${PHYSBRAIN_BASE:-/home/ktmt-agx-xv/Data/khoanhd/PhysBrain}"
DEST="$BASE/models"
QUANT="${PHYSBRAIN_QUANT:-PhysBrain1.5-8B.i1-Q4_K_M.gguf}"
MMPROJ="${PHYSBRAIN_MMPROJ:-PhysBrain1.5-8B.mmproj-f16.gguf}"

mkdir -p "$DEST"
cd "$DEST" || exit 1

fetch() {
  # $1 = ten file dich, $2 = url
  if [ -s "$1" ]; then
    echo "  da co $1 ($(du -h "$1" | cut -f1)), bo qua"
    return 0
  fi
  echo "  tai $1 ..."
  curl -L -C - --retry 5 --retry-delay 5 -o "$1" "$2" || { echo "  THAT BAI: $1" >&2; return 1; }
  echo "  xong $1 ($(du -h "$1" | cut -f1))"
}

echo "=== 1/2 weights ==="
fetch "$QUANT" \
  "https://huggingface.co/mradermacher/PhysBrain1.5-8B-i1-GGUF/resolve/main/$QUANT" || exit 1

echo "=== 2/2 mmproj (repo KHAC) ==="
fetch "$MMPROJ" \
  "https://huggingface.co/mradermacher/PhysBrain1.5-8B-GGUF/resolve/main/$MMPROJ" || exit 1

echo
ls -la "$DEST"
