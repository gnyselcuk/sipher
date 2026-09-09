#!/usr/bin/env bash
# Sipher FHE GPU Build Pipeline
# 1. Patched OpenFHE (static) → ~/.local/fides-openfhe
# 2. FIDESlib → ~/.local/fideslib
# 3. Sipher GPU engine → cpp/gpu/build/
#
# Gereksinimler:
#   - CUDA 12.4 toolkit (/usr/local/cuda)
#   - GCC 13 (gcc-13, g++-13) — CUDA 12.4 GCC 15 desteklemiyor
#   - CUDA math_functions.h yaması (glibc 2.41+ noexcept uyumluluğu)
set -e

FIDESLIB_SRC="$HOME/FIDESlib"
OPENFHE_PREFIX="$HOME/.local/fides-openfhe"
FIDESLIB_PREFIX="$HOME/.local/fideslib"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$SCRIPT_DIR/gpu"
NPROC=$(nproc)

echo "═══ Sipher FHE GPU Build ═══"
echo "  CUDA:      $CUDA_PATH"
echo "  OpenFHE:   $OPENFHE_PREFIX"
echo "  FIDESlib:  $FIDESLIB_PREFIX"
echo "  GCC:       $(gcc-13 --version | head -1)"
echo "  Jobs:      $NPROC"
echo ""

# ── Step 0: Check prerequisites ──
if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_PATH/bin/nvcc"
    exit 1
fi
if ! command -v gcc-13 &>/dev/null; then
    echo "ERROR: gcc-13 not found. Install: sudo apt install gcc-13 g++-13"
    exit 1
fi
echo "✓ CUDA: $($CUDA_PATH/bin/nvcc --version | grep release)"

# ── Step 1: Patched OpenFHE (static libs) ──
if [ ! -f "$OPENFHE_PREFIX/lib/libOPENFHEpke_static.a" ]; then
    echo ""
    echo "── Step 1/3: Building patched OpenFHE (static) ──"
    cd "$FIDESLIB_SRC/deps"
    rm -rf openfhe-src/build

    git submodule update --init --recursive --remote
    cd openfhe-src
    git checkout fideslib-ref-v1.5.1.1 2>/dev/null || true
    git apply --check ../fideslib-ref-1.5.1.1.patch 2>/dev/null && \
        git apply ../fideslib-ref-1.5.1.1.patch || \
        echo "  (patch already applied)"

    mkdir -p build && cd build
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

# ── Step 2: FIDESlib ──
if [ ! -d "$FIDESLIB_PREFIX/share/fideslib" ]; then
    echo ""
    echo "── Step 2/3: Building FIDESlib ──"
    cd "$FIDESLIB_SRC"
    rm -rf build

    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DFIDESLIB_INSTALL_PREFIX="$FIDESLIB_PREFIX" \
          -DOPENFHE_INSTALL_PREFIX="$OPENFHE_PREFIX" \
          -DOpenFHE_DIR="$OPENFHE_PREFIX/lib/OpenFHE" \
          -DCUDA_PATH="$CUDA_PATH" \
          -DFIDESLIB_INSTALL_OPENFHE=OFF \
          -DFIDESLIB_COMPILE_TESTS=OFF \
          -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
          -DFIDESLIB_ARCH="89-real" \
          ..
    make -j$NPROC
    make install
    echo "✓ FIDESlib → $FIDESLIB_PREFIX"
else
    echo "✓ FIDESlib already installed"
fi

# ── Step 3: Sipher GPU Engine ──
echo ""
echo "── Step 3/3: Building Sipher GPU Engine ──"
cd "$GPU_DIR"
rm -rf build
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -Dfideslib_DIR="$FIDESLIB_PREFIX/share/fideslib/cmake" \
      ..
make -j$NPROC

echo ""
echo "═══════════════════════════════════════════"
echo "✓ Build complete!"
echo "  Binary: $GPU_DIR/build/sipher_fhe_gpu"
echo ""
echo "Usage:"
echo "  ./build/sipher_fhe_gpu --benchmark"
echo "  ./build/sipher_fhe_gpu --bootstrap-test"
echo "  ./build/sipher_fhe_gpu --infer ../weights/"
echo "  ./build/sipher_fhe_gpu --layers 3 ../weights/"
echo "═══════════════════════════════════════════"
