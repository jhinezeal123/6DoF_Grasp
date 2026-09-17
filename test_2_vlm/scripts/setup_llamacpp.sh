#!/usr/bin/env bash
#
# setup_llamacpp.sh - Build llama.cpp tren Jetson AGX Xavier (aarch64, CUDA 11.4).
#
#   bash scripts/setup_llamacpp.sh
#
# Thu CUDA truoc (sm_72), khong duoc thi tu dong lui ve CPU-only.
#
# ============================ BAY CMAKE ============================
# llama.cpp/CMakeLists.txt ghi `cmake_minimum_required(3.14)`, nen rat de tuong
# cmake 3.16.3 co san tren JetPack la du. KHONG DU.
# File con ggml/src/ggml-cuda/CMakeLists.txt ghi `cmake_minimum_required(3.18)`,
# va no chi duoc doc khi bat GGML_CUDA=ON. Trieu chung rat de chan doan nham:
#
#   CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:1 (cmake_minimum_required):
#     CMake 3.18 or higher is required.  You are running version 3.16.3
#
# Trong khi do CMakeLists goc bao "3.14" va phan CPU build binh thuong. Nhin qua
# tuong nhu loi CUDA/toolkit, that ra chi la thieu cmake. Script nay tu tai cmake
# moi ve $BASE/tools/ (khong can sudo) khi ban he thong < 3.18.
# ===================================================================
#
# Yeu cau da kiem chung tren may nay:
#   cmake >= 3.18  (he thong co 3.16.3 -> script tu tai 3.31.6)
#   CUDA 11.4.315  (Xavier la sm_72, CUDA > 11.0 la du; da configure thanh cong)
#   GNU 9.4.0 lam host compiler
#   ~2 GB dia cho build
set -uo pipefail

BASE="${PHYSBRAIN_BASE:-/home/ktmt-agx-xv/Data/khoanhd/PhysBrain}"
REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
SRC="$BASE/llama.cpp"
LOGS="$BASE/logs"
TOOLS="$BASE/tools"
ARCH="${CUDA_ARCH:-72}"
JOBS="${JOBS:-$(nproc)}"
CMAKE_MIN="3.18"
# Thu lan luot, ban nao tai duoc thi dung.
CMAKE_VERSIONS="${CMAKE_VERSIONS:-3.31.6 3.30.5 3.29.6}"

mkdir -p "$LOGS" "$TOOLS"

# ---------------------------------------------------------------- cmake >= 3.18
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

CM="$(command -v cmake || true)"
if [ -n "$CM" ] && ver_ge "$("$CM" --version | head -1 | awk '{print $3}')" "$CMAKE_MIN"; then
  echo "cmake he thong du moi: $("$CM" --version | head -1)"
else
  echo "cmake he thong $( [ -n "$CM" ] && "$CM" --version | head -1 | awk '{print $3}' || echo 'khong co') < $CMAKE_MIN"
  echo "-> tai cmake rieng vao $TOOLS/cmake (khong can sudo)"
  if [ ! -x "$TOOLS/cmake/bin/cmake" ]; then
    cd "$TOOLS" || exit 1
    for V in $CMAKE_VERSIONS; do
      echo "   thu cmake $V ..."
      if curl -sL --retry 3 -o cmake.tgz \
           "https://github.com/Kitware/CMake/releases/download/v$V/cmake-$V-linux-aarch64.tar.gz" \
         && tar xzf cmake.tgz 2>/dev/null; then
        mv "cmake-$V-linux-aarch64" cmake 2>/dev/null && rm -f cmake.tgz && break
      fi
      rm -f cmake.tgz
    done
  fi
  CM="$TOOLS/cmake/bin/cmake"
  [ -x "$CM" ] || { echo "khong cai duoc cmake >= $CMAKE_MIN" >&2; exit 1; }
  echo "-> dung $("$CM" --version | head -1)"
fi

# ---------------------------------------------------------------------- source
if [ ! -d "$SRC" ]; then
  echo "clone $REPO"
  git clone --depth 1 "$REPO" "$SRC" || { echo "clone that bai" >&2; exit 1; }
else
  echo "da co $SRC, dung lai"
fi

export PATH="/usr/local/cuda/bin:$PATH"
cd "$SRC" || exit 1
echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'khong co')"
echo "jobs: $JOBS"

# ----------------------------------------------------------------------- build
build_with() {
  # $1 = "cuda" | "cpu";  thu muc build tach rieng de ban CUDA khong de len ban CPU
  local tag="$1" dir="build-$1"
  [ "$1" = "cpu" ] && dir="build"
  rm -rf "$dir"
  if [ "$1" = "cuda" ]; then
    "$CM" -B "$dir" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_cuda.log" 2>&1
  else
    "$CM" -B "$dir" -DGGML_CUDA=OFF \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_cpu.log" 2>&1
  fi
  [ -f "$dir/CMakeCache.txt" ] || { echo "configure ($tag) that bai"; return 1; }
  "$CM" --build "$dir" --config Release -j"$JOBS" > "$LOGS/make_$tag.log" 2>&1
  [ -x "$dir/bin/llama-server" ]
}

echo "=== THU 1: CUDA sm_$ARCH (build-cuda/) ==="
if build_with cuda; then
  echo "KET QUA: CUDA BUILD OK"
  BIN="$SRC/build-cuda/bin/llama-server"
else
  echo "--- CUDA that bai, 15 dong cuoi make_cuda.log ---"
  tail -15 "$LOGS/make_cuda.log" 2>/dev/null
  echo "=== THU 2: CPU only (build/) ==="
  if build_with cpu; then
    echo "KET QUA: CPU BUILD OK"
    BIN="$SRC/build/bin/llama-server"
  else
    echo "KET QUA: THAT BAI CA HAI"
    exit 1
  fi
fi

echo
echo "Binary: $BIN"
"$BIN" --version 2>&1 | head -3
