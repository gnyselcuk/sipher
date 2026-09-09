"""
BSGS Matrix-Vector Multiplication for CKKS
============================================
Baby-Step Giant-Step algorithm for encrypted matrix-vector multiply.

Standard diagonal encoding: O(n) rotations + O(n) ct-pt mults
BSGS:                      O(√n) rotations + O(n) ct-pt mults

The rotation reduction is significant because each rotation requires
a key-switching operation (≈3 NTTs), which is the most expensive
operation after ct-ct multiplication.

Reference: Halevi & Shoup (2014), THOR (CCS 2025)
"""

import torch
import numpy as np
import math
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class BSGSMatVec:
    """
    Baby-Step Giant-Step matrix-vector multiplication.
    
    For y = W · x where W is (out_dim × in_dim) and x is encrypted:
    
    Standard: for each diagonal k in 0..in_dim-1:
                rotate(x, k) * diag_k → accumulate
              Cost: in_dim rotations + in_dim ct-pt mults
    
    BSGS:     baby_size = √in_dim
              Step 1 (baby): precompute rot(x, 0), rot(x, 1), ..., rot(x, baby-1)
              Step 2 (giant): for each giant j in 0..ceil(in_dim/baby)-1:
                                for each baby i in 0..baby-1:
                                  k = j*baby + i
                                  if k < in_dim:
                                    baby_result[i] = rot(x, i) * diag_k
                                giant_sum = sum(baby_result)
                                rotate(giant_sum, j*baby) → accumulate
              Cost: baby rotations (precompute) + ceil(in_dim/baby) giant rotations
                    + in_dim ct-pt mults (unchanged)
    """
    
    def __init__(self, in_dim, out_dim=None):
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.baby = math.ceil(math.sqrt(in_dim))
        self.giant = math.ceil(in_dim / self.baby)
    
    def get_diagonals(self, W):
        """
        Extract diagonals from weight matrix W.
        W: (out_dim × in_dim) numpy array
        Returns: list of in_dim diagonal vectors, each of length out_dim
        
        For CKKS rotation convention: rot(x, k)[i] = x[(i+k) mod n]
        diag_k[i] = W[i, (i+k) mod in_dim]
        """
        diags = []
        for k in range(self.in_dim):
            d = np.zeros(max(self.out_dim, self.in_dim), dtype=np.float64)
            for i in range(self.out_dim):
                d[i] = W[i, (i + k) % self.in_dim]
            diags.append(d)
        return diags
    
    def count_ops(self):
        """Count FHE operations for BSGS matvec."""
        return {
            'rotations': self.baby + self.giant,  # baby precompute + giant steps
            'ct_pt_mults': self.in_dim,            # one per diagonal (unchanged)
            'adds': self.in_dim + self.giant,       # accumulation
            'baby_size': self.baby,
            'giant_size': self.giant,
        }
    
    def count_ops_standard(self):
        """Count FHE operations for standard diagonal encoding."""
        return {
            'rotations': self.in_dim,
            'ct_pt_mults': self.in_dim,
            'adds': self.in_dim,
        }
    
    def simulate_plaintext(self, W, x):
        """Standard matvec for reference."""
        return W @ x
    
    def simulate_bsgs(self, W, x, n_slots=None):
        """
        Simulate BSGS matvec (plaintext, Halevi-Shoup form).
        CKKS: rot(x, k)[i] = x[(i+k) mod N]

        Fix (2026-08-01): slot shift + col wrapping + input replikasyonu.
        Eski kod g>0'da yanlış köşegen kullanıyordu.
        """
        n = self.in_dim
        N = n_slots or max(self.out_dim, n)
        result = np.zeros(self.out_dim, dtype=np.float64)

        # Input replikasyonu: x_rep[i] = x[i % in_dim] (periyodik N slot)
        x_rep = np.zeros(N, dtype=np.float64)
        for i in range(N):
            x_rep[i] = x[i % n]

        # Baby step: precompute rot(x_rep, b) for b = 0..baby-1
        baby_rots = []
        for b in range(self.baby):
            rotated = np.zeros(N, dtype=np.float64)
            for j in range(N):
                rotated[j] = x_rep[(j + b) % N]
            baby_rots.append(rotated)

        # Giant step (Halevi-Shoup: slot g*baby+row, col wrapping)
        for g in range(self.giant):
            inner_sum = np.zeros(N, dtype=np.float64)
            for b in range(self.baby):
                k = g * self.baby + b
                if k >= n:
                    break
                # BSGS diagonal: diag[g*baby+row] = W[row, (g*baby+row+b) % in_dim]
                diag = np.zeros(N, dtype=np.float64)
                for slot in range(g * self.baby, min(g * self.baby + self.out_dim, N)):
                    row = slot - g * self.baby
                    col = (slot + b) % n
                    diag[slot] = W[row, col]

                inner_sum += diag * baby_rots[b]

            # Giant rotation: rot(inner_sum, g*baby)
            giant_shift = g * self.baby
            rotated_sum = np.zeros(N, dtype=np.float64)
            for idx in range(N):
                rotated_sum[idx] = inner_sum[(idx + giant_shift) % N]

            result += rotated_sum[:self.out_dim]

        return result


def verify_bsgs():
    """Verify diagonal encoding correctness + BSGS operation counts."""
    print("=" * 60)
    print("BSGS MatVec — Verification")
    print("=" * 60)
    
    # Verify standard diagonal encoding produces correct matvec
    print("\n  Standard diagonal encoding correctness:")
    test_cases = [(16, 16), (32, 32), (64, 64), (128, 128)]
    
    for in_dim, out_dim in test_cases:
        W = np.random.randn(out_dim, in_dim)
        x = np.random.randn(in_dim)
        
        # Standard matvec
        y_std = W @ x
        
        # Diagonal encoding
        bsgs = BSGSMatVec(in_dim, out_dim)
        diags = bsgs.get_diagonals(W)
        y_diag = np.zeros(out_dim)
        for k in range(in_dim):
            # rot(x, k): CKKS convention rot(x,k)[i] = x[(i+k) mod n]
            rot_x = np.zeros(in_dim)
            for i in range(in_dim):
                rot_x[i] = x[(i + k) % in_dim]
            y_diag += diags[k][:out_dim] * rot_x[:out_dim]
        
        err = np.max(np.abs(y_std - y_diag))
        status = "✅" if err < 1e-10 else "❌"
        print(f"    {in_dim}×{out_dim}: err={err:.2e} {status}")
    
    # BSGS simulate_bsgs ground-truth karşılaştırması
    print(f"\n  BSGS simulate_bsgs vs W@x (ground truth):")
    for in_dim, out_dim in [(16, 16), (32, 32), (64, 64), (32, 64), (64, 32)]:
        W = np.random.randn(out_dim, in_dim)
        x = np.random.randn(in_dim)
        y_ref = W @ x
        bsgs = BSGSMatVec(in_dim, out_dim)
        y_bsgs = bsgs.simulate_bsgs(W, x, n_slots=128)
        err = np.max(np.abs(y_ref - y_bsgs))
        status = "✅" if err < 1e-10 else "❌"
        print(f"    {in_dim}×{out_dim} (N=128): err={err:.2e} {status}")

    # BSGS operation counts (algorithm is proven in literature)
    print(f"\n  BSGS rotation reduction (Halevi-Shoup 2014, THOR CCS 2025):")
    print(f"  {'seq':>6} {'Std rot':>8} {'BSGS rot':>9} {'Reduction':>10} "
          f"{'ct-pt':>7} {'baby':>5} {'giant':>6}")
    print(f"  {'-'*55}")
    
    for seq in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        bsgs = BSGSMatVec(seq)
        ops = bsgs.count_ops()
        ops_std = bsgs.count_ops_standard()
        reduction = ops_std['rotations'] / ops['rotations']
        print(f"  {seq:>6} {ops_std['rotations']:>8} {ops['rotations']:>9} "
              f"{reduction:>9.1f}× {ops['ct_pt_mults']:>7} "
              f"{ops['baby_size']:>5} {ops['giant_size']:>6}")
    
    print(f"\n  Note: BSGS reduces rotations O(n)→O(√n), ct-pt mults unchanged.")
    print(f"  Correctness proven in Halevi-Shoup (2014), used in THOR (CCS 2025).")
    print("=" * 60)


def benchmark_bsgs_scaling():
    """Benchmark BSGS operation counts at different scales."""
    print("\n── BSGS Scaling: Rotation Reduction ──")
    print(f"  {'seq':>6} {'Std rot':>8} {'BSGS rot':>9} {'Reduction':>10} "
          f"{'ct-pt':>7} {'baby':>5} {'giant':>6}")
    print(f"  {'-'*55}")
    
    for seq in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        bsgs = BSGSMatVec(seq)
        ops = bsgs.count_ops()
        ops_std = bsgs.count_ops_standard()
        reduction = ops_std['rotations'] / ops['rotations']
        print(f"  {seq:>6} {ops_std['rotations']:>8} {ops['rotations']:>9} "
              f"{reduction:>9.1f}× {ops['ct_pt_mults']:>7} "
              f"{ops['baby_size']:>5} {ops['giant_size']:>6}")
    
    # Time estimates for Pythia-70M
    print(f"\n── Pythia-70M Estimate (d=896, seq=1024, 24L) ──")
    
    # GPU costs (N=65536)
    t_rot = 0.032    # ms per rotation
    t_ct_pt = 0.027  # ms per ct-pt mult
    t_ct_ct = 0.036  # ms per ct-ct mult
    
    N_slots = 32768
    d_model = 896
    seq = 1024
    n_layers = 24
    
    ch_per_ct = N_slots // seq  # 32
    n_groups = math.ceil(d_model / ch_per_ct)  # 28
    
    bsgs = BSGSMatVec(seq)
    ops = bsgs.count_ops()
    
    # Per group per layer
    rot_per_group = 2 * ops['rotations']  # W_up + W_down
    ct_pt_per_group = 2 * ops['ct_pt_mults']  # W_up + W_down
    ct_ct_per_group = 2  # polynomial activation
    
    # Total
    total_rot = rot_per_group * n_groups * n_layers
    total_ct_pt = ct_pt_per_group * n_groups * n_layers
    total_ct_ct = ct_ct_per_group * n_groups * n_layers
    
    t_rot_total = total_rot * t_rot / 1000  # seconds
    t_ct_pt_total = total_ct_pt * t_ct_pt / 1000
    t_ct_ct_total = total_ct_ct * t_ct_ct / 1000
    t_total = t_rot_total + t_ct_pt_total + t_ct_ct_total
    
    # Standard (no BSGS)
    std_rot_per_group = 2 * seq
    total_std_rot = std_rot_per_group * n_groups * n_layers
    t_std_rot_total = total_std_rot * t_rot / 1000
    t_std_total = t_std_rot_total + t_ct_pt_total + t_ct_ct_total
    
    print(f"  Channel packing: {ch_per_ct} ch/ct, {n_groups} groups")
    print(f"  BSGS: baby={bsgs.baby}, giant={bsgs.giant}")
    print(f"")
    print(f"  {'Component':>15} {'Standard':>12} {'BSGS':>12} {'Savings':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Rotations':>15} {total_std_rot:>11,} {total_rot:>11,} "
          f"{total_std_rot/total_rot:>9.1f}×")
    print(f"  {'ct-pt mults':>15} {total_ct_pt:>11,} {total_ct_pt:>11,} "
          f"{'1.0×':>10}")
    print(f"  {'ct-ct mults':>15} {total_ct_ct:>11,} {total_ct_ct:>11,} "
          f"{'1.0×':>10}")
    print(f"  {'':>15} {'':>12} {'':>12} {'':>10}")
    print(f"  {'Rot time':>15} {t_std_rot_total:>11.1f}s {t_rot_total:>11.1f}s")
    print(f"  {'ct-pt time':>15} {t_ct_pt_total:>11.1f}s {t_ct_pt_total:>11.1f}s")
    print(f"  {'ct-ct time':>15} {t_ct_ct_total:>11.1f}s {t_ct_ct_total:>11.1f}s")
    print(f"  {'TOTAL':>15} {t_std_total:>11.1f}s {t_total:>11.1f}s")
    
    print(f"\n  Power-Softmax target: 8.6s (A100)")
    print(f"  CipherFormer BSGS:    {t_total:.1f}s (RTX 4060)")
    
    # A100 estimate (3-5× faster NTT)
    for factor in [3, 4, 5]:
        t_a100 = t_total / factor
        status = "✅" if t_a100 < 8.6 else "⚠"
        print(f"  A100 ({factor}× factor):    {t_a100:.1f}s {status}")
    
    print(f"\n  Bottleneck: ct-pt mults = {t_ct_pt_total/t_total*100:.0f}% of total")
    print(f"  → ct-pt mults are the binding constraint, not rotations")
    print(f"  → Further optimization needs: fewer ct-pt mults or faster GPU")


if __name__ == "__main__":
    verify_bsgs()
    benchmark_bsgs_scaling()
