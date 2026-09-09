#!/bin/bash
# Verify the FHE global-attention path parity (q²-mutation + rotation-key fixes, 2026-09-09).
#
# The global path was historically skipped (every checkpoint froze gate_global=0), so two
# bugs went uncaught: (1) the degree-2 q² term squared a ciphertext that matvec_rep had
# already mutated, and (2) compute_rotation_indices omitted the baby-step rotations the
# block-diagonal global matvec needs at d=1024. This script reproduces the verification:
# it temporarily sets gate_global=0.5 on weights_en16 (feat_degree=2 → q² path active),
# runs single-layer parity, then restores gate_global=0 (trap → safe even on crash).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$ROOT/cpp/weights_en16"
BIN="$ROOT/cpp/gpu/build/sipher_fhe_gpu"

restore() {
  python3 -c "import struct,glob; [open(f,'wb').write(struct.pack('<d',0.0)) for f in glob.glob('$W/layers.*.gate_global.bin')]" 2>/dev/null
  echo "[restore] gate_global -> 0.0 (weights faithful)"
}
trap restore EXIT

[ -x "$BIN" ] || { echo "binary yok: $BIN — önce 'cd cpp/gpu/build && make'"; exit 1; }
[ -d "$W" ]   || { echo "weights yok: $W — export_weights.py ile üret"; exit 1; }

FD=$(grep feat_degree "$W/config.txt" | awk '{print $2}')
echo "[hack] gate_global -> 0.5 (feat_degree=$FD; 2 ise q² yolu aktif)"
python3 -c "import struct,glob; [open(f,'wb').write(struct.pack('<d',0.5)) for f in glob.glob('$W/layers.*.gate_global.bin')]"

echo "[run] parity layer 0 (GPU ~3.2GB, ring=16384)..."
echo "---------------------------------------------------------------"
"$BIN" --parity 1 "$W" --ring 16384 --dump-layer 0 2>&1 \
  | grep -iE "DBG global|DBG gated|DBG q\]|Layer 0:|cos_sim\(logits|PARITY|not found|terminate"
echo "---------------------------------------------------------------"
echo "Beklenen: 'global' max_err ~1e-5..1e-6, Layer 0 cos_sim=1, PARITY OK."
echo "  'Rotation index N not found' → rotation-key fix eksik (compute_rotation_indices)."
echo "  global ~2× REF                → q² mutation fix eksik (taze encrypt)."
echo "Not: d=1024 flagship için weights_localonly (feat_degree=1) ile de koşulabilir."
