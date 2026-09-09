#!/usr/bin/env bash
# Sipher FHE GPU Build Pipeline — H100 (Hopper) uyarlı
# 0. Bağımlılıklar (GCC 13 + CUDA kontrolü)
# 1. FIDESlib clone (v2.1.3, submodule'ler dahil)
# 2. Patched OpenFHE (static) → ~/.local/fides-openfhe
# 3. FIDESlib → ~/.local/fideslib   (FIDESLIB_ARCH="90-real" — H100)
# 4. Sipher GPU engine → cpp/gpu/build/
set -e

FIDESLIB_REPO="${FIDESLIB_REPO:-https://github.com/CAPS-UMU/FIDESlib.git}"
FIDESLIB_SRC="$HOME/FIDESlib"
OPENFHE_PREFIX="$HOME/.local/fides-openfhe"
FIDESLIB_PREFIX="$HOME/.local/fideslib"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
# Arch: 90-real (H100/Hopper), 89-real (RTX 4090/Ada), 86-real (RTX 3090/Ampere)
FIDESLIB_ARCH="${FIDESLIB_ARCH:-90-real}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Script cpp/ içinde veya kökte duruyor olabilir — GPU_DIR'i esnek bul
if [ -d "$SCRIPT_DIR/cpp/gpu" ]; then
    GPU_DIR="$SCRIPT_DIR/cpp/gpu"
elif [ -d "$SCRIPT_DIR/gpu" ]; then
    GPU_DIR="$SCRIPT_DIR/gpu"
else
    echo "ERROR: cpp/gpu bulunamadı (SCRIPT_DIR=$SCRIPT_DIR)"
    exit 1
fi
NPROC=$(nproc)

log() { echo -e "\033[1;36m══ $* ══\033[0m"; }

log "Sipher FHE GPU Build (H100)"
echo "  FIDESlib:  $FIDESLIB_SRC"
echo "  OpenFHE:   $OPENFHE_PREFIX"
echo "  FIDESlib:  $FIDESLIB_PREFIX"
echo "  CUDA:      $CUDA_PATH"
echo "  Jobs:      $NPROC"
echo ""

# ── Step 0: GCC 13 (CUDA 12.x GCC 13'e kadar destekler) ──
if ! command -v gcc-13 &>/dev/null; then
    log "gcc-13 kuruluyor"
    (apt-get update -qq && apt-get install -y -qq gcc-13 g++-13) 2>/dev/null \
        || (sudo apt-get update -qq && sudo apt-get install -y -qq gcc-13 g++-13)
fi
gcc-13 --version | head -1

# ── Step 0b: Build bağımlılıkları (cmake/make/git/bc eksikse kur) ──
MISSING=""
for cmd in cmake make git g++; do
    command -v $cmd >/dev/null 2>&1 || MISSING="$MISSING $cmd"
done
command -v bc >/dev/null 2>&1 || MISSING="$MISSING bc"
if [ -n "$MISSING" ]; then
    log "Eksik build araçları kuruluyor:$MISSING"
    (apt-get update -qq && apt-get install -y -qq cmake build-essential git bc) 2>/dev/null \
        || (sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential git bc)
fi
for cmd in cmake make git; do
    command -v $cmd >/dev/null 2>&1 || { echo "ERROR: $cmd hâlâ yok"; exit 1; }
done
echo "✓ cmake: $(cmake --version | head -1)"

# ── CUDA kontrolü ──
if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    # /usr/local/cuda symlink yoksa gerçek toolkit'i bul
    CUDA_PATH=$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 || true)
    [ -z "$CUDA_PATH" ] && { echo "ERROR: nvcc bulunamadı"; exit 1; }
fi
echo "✓ CUDA: $($CUDA_PATH/bin/nvcc --version | grep release)"

# ── glibc >= 2.40 ise CUDA 12.4 math_functions.h noexcept yaması ──
GLIBC_VER=$(ldd --version | head -1 | grep -oP '\d+\.\d+$')
if [ "$(echo "$GLIBC_VER >= 2.40" | bc -l 2>/dev/null)" = "1" ]; then
    CUDA_INC="$CUDA_PATH/targets/x86_64-linux/include/crt"
    if [ -f "$CUDA_INC/math_functions.h" ]; then
        log "glibc $GLIBC_VER — math_functions.h noexcept yaması"
        sed -i '/__device_builtin__.*\(rsqrt\|cospi\|sinpi\|acospi\|asinpi\)/ { /__THROW/!s/);$/) __THROW;/ }' "$CUDA_INC/math_functions.h" || true
    fi
fi

# ── Step 1: FIDESlib kaynak (submodule'ler dahil) ──
if [ ! -d "$FIDESLIB_SRC" ]; then
    log "FIDESlib clone (v2.1.3)"
    git clone --recurse-submodules --depth 1 "$FIDESLIB_REPO" "$FIDESLIB_SRC"
else
    echo "✓ FIDESlib kaynak mevcut: $FIDESLIB_SRC"
    git -C "$FIDESLIB_SRC" submodule update --init --recursive 2>/dev/null || true
fi

# ── Step 2: Patched OpenFHE (static libs) ──
if [ ! -f "$OPENFHE_PREFIX/lib/libOPENFHEpke_static.a" ]; then
    log "Patched OpenFHE (static) derleniyor"
    cd "$FIDESLIB_SRC/deps"
    cd openfhe-src
    git checkout fideslib-ref-v1.5.1.1 2>/dev/null || true
    git apply --check ../fideslib-ref-1.5.1.1.patch 2>/dev/null && \
        git apply ../fideslib-ref-1.5.1.1.patch || \
        echo "  (patch already applied)"
    rm -rf build && mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$OPENFHE_PREFIX" \
          -DBUILD_STATIC=ON \
          ..
    make -j$NPROC
    make install
    echo "✓ Patched OpenFHE → $OPENFHE_PREFIX"
else
    echo "✓ Patched OpenFHE already installed"
fi

# ── Step 3: FIDESlib (H100 = compute 9.0) ──
if [ ! -d "$FIDESLIB_PREFIX/share/fideslib" ]; then
    log "FIDESlib derleniyor (90-real)"
    cd "$FIDESLIB_SRC"
    rm -rf build && mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DFIDESLIB_INSTALL_PREFIX="$FIDESLIB_PREFIX" \
          -DOPENFHE_INSTALL_PREFIX="$OPENFHE_PREFIX" \
          -DOpenFHE_DIR="$OPENFHE_PREFIX/lib/OpenFHE" \
          -DCUDA_PATH="$CUDA_PATH" \
          -DFIDESLIB_INSTALL_OPENFHE=OFF \
          -DFIDESLIB_COMPILE_TESTS=OFF \
          -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
          -DFIDESLIB_ARCH="$FIDESLIB_ARCH" \
          ..
    make -j$NPROC
    make install
    echo "✓ FIDESlib → $FIDESLIB_PREFIX"
else
    echo "✓ FIDESlib already installed"
fi

# ── Step 4: Sipher GPU Engine ──
log "Sipher GPU engine derleniyor"
cd "$GPU_DIR"
rm -rf build && mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -Dfideslib_DIR="$FIDESLIB_PREFIX/share/fideslib/cmake" \
      ..
make -j$NPROC

echo ""
log "Build tamam!"
echo "  Binary: $GPU_DIR/build/sipher_fhe_gpu"
echo ""
echo "Usage:"
echo "  ./build/sipher_fhe_gpu --benchmark"
echo "  ./build/sipher_fhe_gpu --bootstrap-test"
echo "  ./build/sipher_fhe_gpu --infer ../weights/"
echo "  ./build/sipher_fhe_gpu --layers 3 ../weights/"
echo ""
echo "Smoke test:"
echo "  cd $GPU_DIR/build && ./sipher_fhe_gpu --benchmark 2>&1 | tail -20"
