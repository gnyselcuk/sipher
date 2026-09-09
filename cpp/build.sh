#!/bin/bash
# Sipher FHE — Build Script
set -e

echo "═══ Sipher FHE Build ═══"

# OpenFHE kontrolü
if ! pkg-config --exists OpenFHE 2>/dev/null && [ ! -d "/usr/local/include/openfhe" ]; then
    echo "OpenFHE bulunamadı. Kurulum:"
    echo ""
    echo "  # Ubuntu/Debian:"
    echo "  sudo apt install cmake g++ libgmp-dev libntl-dev"
    echo "  git clone https://github.com/openfheorg/openfhe-release.git"
    echo "  cd openfhe-release && mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc) && sudo make install"
    echo ""
    echo "  # macOS:"
    echo "  brew install cmake gmp ntl"
    echo "  # (aynı git clone + cmake + make adımları)"
    echo ""
    exit 1
fi

echo "OpenFHE bulundu ✓"

# Weight export
if [ ! -d "weights" ] || [ ! -f "weights/config.txt" ]; then
    echo ""
    echo "Weight export ediliyor..."
    python3 export_weights.py
fi

# Build
echo ""
echo "Build ediliyor..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "═══ Build tamamlandı ═══"
echo "  Binary: build/sipher_fhe"
echo ""
echo "  Kullanım:"
echo "    ./build/sipher_fhe --weights weights --benchmark"
echo "    ./build/sipher_fhe --weights weights --prompt \"Merkez Bankası\""
echo ""
