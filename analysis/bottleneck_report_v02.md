# Phase 1: Transformer FHE Bottleneck Analysis (v2, Calibrated)

**Date:** 2026-07-27
**Status:** Calibrated with empirical OpenFHE measurements

## Setup

- **Model:** BERT-base (d_model=768, n_heads=12, d_ff=3072, seq_len=512, 12 layers)
- **FHE Scheme:** CKKS, OpenFHE 1.5.1
- **Parameters:** N=32768 (medium), N=65536 (large), L=40, Δ=50-bit
- **Method:** Analytical model calibrated with empirical wall-clock measurements

## Empirical Reference Timings (OpenFHE 1.5.1)

| Operation | N=32768 | N=65536 | v1 assumption | Error factor |
|---|---|---|---|---|
| ct-ct Add | 3.4 ms | 15.8 ms | 0.01 ms | 337× |
| ct-ct Mult | 51.0 ms | 161.3 ms | 0.1 ms | 510× |
| ct-pt Mult | 8.7 ms | 30.0 ms | — | — |
| Rotation | 50.3 ms | 120.2 ms | 0.5 ms | 101× |

## Critical Finding: The True Cost

| Metric | N=32768 | N=65536 |
|---|---|---|
| Block wall time | 1,391,416 s (16 days) | 3,643,164 s (42 days) |
| Full model (12 layers) | **4,638 hours (193 days)** | **12,144 hours (506 days)** |
| FHE/GPU ratio | **139,000,000×** | **364,000,000×** |
| With 21× packing | 6,600,000× | — |

The standard Transformer is fundamentally incompatible with FHE.
This is not an optimization problem: it is an architecture problem.

## Bottleneck Breakdown

### Block-level
| Component | Share | Wall time |
|---|---|---|
| Attention | **~100%** | 1,391,358 s |
| FFN | ~0% | 55 s |
| LayerNorm (×2) | ~0% | 3 s |

### Attention sub-breakdown
| Component | Share | Wall time | Why |
|---|---|---|---|
| **Q·K^T** | **84.3%** | 1,173,105 s (13.6 days) | seq² ct-ct mults × 12 heads |
| Attn·V | 15.1% | 209,938 s (2.4 days) | seq×d_k ct-ct mults × 12 heads |
| Softmax | 0.6% | 8,225 s | Polynomial approx is cheap |
| QKV_proj | 0.0% | 36 s | ct-pt mults (weights are plaintext) |

### Key insight: ct-ct vs ct-pt

Operations where BOTH operands are encrypted (Q·K^T, Attn·V, Softmax)
use ct-ct multiplication at **51 ms/op** (N=32768).

Operations where one operand is plaintext (QKV projections, FFN)
use ct-pt multiplication at **8.7 ms/op**, 6× cheaper.

The attention mechanism forces ct-ct operations because Q, K, V are
all derived from encrypted user input.

## Why This Happens

1. **Q·K^T** is O(seq²) ct-ct multiplications.
   seq=512 → 262,144 ct-ct mults per head × 12 heads = 3,145,728 total.
   At 51ms each: 160,432 seconds = 44.6 hours. Just for the mults.
   Add rotations for reductions: another 1,012,673 seconds.

2. **Reductions** are rotation-heavy.
   Each dot product of length d_k requires log₂(d_k) = 6 rotations.
   seq² × 6 rotations × 12 heads = 18,874,368 rotations.
   At 50ms each: 943,718 seconds.

3. **Packing** doesn't help enough.
   d_k=64 in 16,384 slots = 0.4% utilization for Q·K^T.
   Even with multi-token packing (21×), the ratio stays at 6.6M×.

## What This Means for CipherFormer

### The three existential design requirements:

1. **ELIMINATE Q·K^T.** Linear attention (Q·(K^T·V)) or kernel-based
   attention removes the O(seq²) ct-ct bottleneck. This alone removes
   84% of total cost.

2. **MINIMIZE ct-ct operations.** Every operation where both operands
   are encrypted costs 6× more than ct-pt. Architecture should maximize
   the ratio of ct-pt to ct-ct operations.

3. **DESIGN FOR THE CIPHERTEXT.** d_k=64 in 16,384 slots is 0.4%
   utilization. Dimensions must be native to the SIMD slot count.

### Revised optimization potential:

| Optimization | Effect | Remaining ratio |
|---|---|---|
| Baseline (unmodified Transformer) | — | 139,000,000× |
| Linear attention (kill Q·K^T) | ~85% attention reduction | ~20,000,000× |
| Multi-token packing (21×) | 21× | ~950,000× |
| Fewer heads (12→2) | ~6× | ~160,000× |
| Depth-opt + shorter seq | ~3× | ~53,000× |
| FHE hardware accelerator | ~100× | ~530× |

Even with ALL optimizations: ~530×. This is the floor for CKKS-based
Transformer inference on general-purpose hardware.

This is why FHE-native design is not optional: it's existential.

## Model Limitations

- Rotation counts for encrypted matrix multiplication may be overestimated
  (more sophisticated packing could reduce them)
- Bootstrap timing is estimated, not measured
- Doesn't account for parallelism (multiple ciphertexts processed concurrently)
- Real implementations may use different matrix encoding strategies
- N=32768 gives ~128-bit security; N=65536 gives ~256-bit

## Artifacts

- `benchmarks/analytical/cost_model_v2.py`: calibrated analytical model
- `benchmarks/empirical/ckks_bench.py`: empirical benchmark suite
- `benchmarks/empirical/results/ckks_bench_results.json`: raw timings
- `docs/literature_survey_fhe_nn.md`: 14-paper literature survey
- `analysis/research_positioning.md`: gap analysis
