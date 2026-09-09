# Phase 1: Transformer FHE Bottleneck Analysis (v0.1)

## Setup

- **Model:** BERT-base config (d_model=768, n_heads=12, d_ff=3072, seq_len=512, 12 layers)
- **FHE Scheme:** CKKS, N=2^16, L=40 levels, Δ=50-bit scaling, 32,768 SIMD slots
- **Method:** Analytical cost model (operation counts × reference timings)

## Key Findings

### 1. Packing Utilization is Catastrophic: 2%

d_model=768 in a 32,768-slot ciphertext = **2.3% utilization**.
98% of SIMD slots sit idle for every linear operation.

This is arguably the #1 structural problem. Transformer dimensions
were designed for GPU tensor cores (multiples of 64/128), not for
CKKS SIMD slots (N/2 = 32,768).

**Implication for CipherFormer:** Model dimensions should be designed
around the ciphertext slot count. An FHE-native model might use
d_model=32,768 (or a significant fraction of N/2) with far fewer
heads, or pack multiple tokens/heads per ciphertext.

### 2. Attention Dominates: 65% of Block Cost

| Component | Share | Wall time (per block) |
|-----------|-------|-----------------------|
| Attention | 64.8% | ~1,192 s |
| FFN | 35.2% | ~648 s |
| LayerNorm (×2) | ~0% | ~22 ms |
| Residual (×2) | ~0% | ~0 ms |

The attention cost is NOT primarily from softmax (only 3.3% of
attention cost). It's from the **matrix multiplications**:
- QKV projections: 3 × (768→768) linear
- Q·K^T: (512×64) × (64×512) per head × 12 heads
- Attention·V: (512×512) × (512×64) per head × 12 heads

**Implication:** Linear attention variants (no Q·K^T outer product)
or kernel-based attention could eliminate the dominant cost.

### 3. Depth Budget is Tight: 1 Block ≈ Full Modulus Chain

- Depth per block: 32 levels
- Available levels (L): 40
- One block consumes 80% of the modulus chain
- Every subsequent layer requires bootstrapping

Bootstrapping overhead (~0.6s total) is negligible vs compute (~22,000s),
but the depth constraint forces architectural choices:
- Each layer must minimize multiplicative depth
- Polynomial activations should use low-degree approximations
- Layer composition must be "depth-aware"

### 4. The FHE/Plaintext Ratio: ~2,000,000×

- Estimated FHE forward pass: ~22,000 seconds (~6 hours)
- Plaintext GPU forward pass: ~10 ms
- Ratio: ~2,000,000×

This is the number that makes people say "FHE is impossible for LLMs."
But it's based on running an **unchanged** Transformer architecture.

### 5. Softmax is NOT the Primary Bottleneck

Contrary to initial hypothesis (H1), softmax accounts for only ~3.3%
of attention cost and ~2.1% of total block cost. The polynomial
approximation (degree 8) is actually quite efficient per-instance.

The real bottlenecks are:
1. **Matrix multiplications** (operation count)
2. **Packing inefficiency** (wasted SIMD slots)
3. **Depth accumulation** (forces bootstrapping)

## Bottleneck Ranking (by impact on total cost)

| Rank | Bottleneck | Impact | Addressable by |
|------|-----------|--------|----------------|
| 1 | Packing utilization (2%) | ~50× waste | Dimension redesign |
| 2 | Attention matmuls | 65% of cost | Linear attention / kernel attention |
| 3 | FFN matmuls | 35% of cost | Structured matrices, sparsity |
| 4 | Depth per block (32) | Forces bootstrap | Low-depth activations, depth-aware design |
| 5 | Softmax | 3.3% of attention | Already manageable with poly approx |

## What This Means for CipherFormer

The three design principles that emerge:

1. **Pack to the ciphertext:** Model dimensions should be native to
   the SIMD slot count. Don't use d_model=768; use d_model that
   fills (or efficiently partitions) 32,768 slots.

2. **Kill the outer product:** Standard attention's Q·K^T creates a
   seq×seq matrix, the most expensive operation. Linear attention
   (Q·(K^T·V) instead of (Q·K^T)·V) or kernel-based attention
   eliminates this.

3. **Budget depth like memory:** Each layer should consume ≤4-5
   multiplicative levels, allowing 8-10 layers per modulus chain
   before bootstrapping.

## Model Limitations

- Analytical only; needs empirical validation with OpenFHE/Concrete
- Reference timings (0.1ms/mult, 0.5ms/rot) are rough estimates
- BSGS rotation count may not match actual library implementations
- Doesn't model memory bandwidth, key storage, or ciphertext expansion
- Polynomial approximation degrees are assumed, not optimized

## Sensitivity Analysis: Combined Optimizations

Starting from BERT-base baseline (unpacked, standard attention):

| Optimization | Per-token cost | Cumulative speedup | FHE/GPU ratio |
|---|---|---|---|
| Baseline (unpacked) | 1,840,151 ms | 1× | ~184,000× |
| + Multi-token packing (42×) | 43,813 ms | 42× | ~4,381× |
| + Linear attention | 18,977 ms | 97× | ~1,913× |
| + Depth optimization | 16,131 ms | 114× | ~1,613× |

**Key insight:** Multi-token SIMD packing alone gives 42× speedup.
This is the single biggest lever, and it requires NO algorithmic change,
just proper data layout. An FHE-native architecture would design its
data layout around ciphertext packing from day one.

With dedicated FHE hardware (NTT accelerators), another 10-100× is
plausible, bringing the ratio to ~16-160×, viable for premium
privacy applications (medical, financial, government).

## Parameter Sensitivity Summary

| Parameter | Impact | Mechanism |
|---|---|---|
| d_model | Quadratic on matmul cost | More parameters, but better packing |
| seq_len | Quadratic on attention | Q·K^T outer product is O(seq²) |
| n_heads | Linear on attention cost | Each head adds independent computation |
| poly_degree | Minimal on speed, major on depth | Higher degree = more depth = more bootstraps |
| ring_dim (N) | Tradeoff: packing vs per-op speed | Larger N = more slots but slower NTT |

**Design implications for CipherFormer:**
- Few heads, large d_k (1-4 heads instead of 12)
- Linear attention (eliminate Q·K^T)
- Low-degree polynomial activations (degree 4-6)
- Data layout designed for maximum SIMD packing

## Next Steps

1. [ ] Empirical validation with OpenFHE (requires Python 3.12 or source build)
2. [ ] Literature survey: how do THE-X, BOLT, NEXUS handle these bottlenecks?
3. [ ] Sensitivity analysis: vary d_model, seq_len, poly_degree
4. [ ] Design first FHE-native layer based on bottleneck ranking
