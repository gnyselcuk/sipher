# CipherFormer: Research Positioning & Gap Analysis

**Date:** 2026-07-27
**Based on:** Bottleneck analysis v0.1 + Sensitivity analysis + Literature survey (14 papers)

---

## The Gap (Confirmed)

Nobody has designed a Transformer architecture from scratch for FHE.

Every existing paper takes a plaintext-designed Transformer and adapts it:
- THE-X: polynomial approximations for GELU/Softmax/LayerNorm
- BOLT: MPC+HE protocol optimization
- Iron: custom MPC protocols for Transformer ops
- Liu & Liu: operator substitution (closest to us, but still adaptive)
- PRISM: HE-aware pruning of existing ResNets
- Concrete ML: framework for small models, can't scale to Transformers

FHE training is limited to **4-layer MLPs** (Chiang 2025). Nobody has trained
anything Transformer-like under FHE.

## Our Unique Position

| Dimension | Existing Work | CipherFormer |
|---|---|---|
| Starting point | Plaintext Transformer | FHE cost model |
| Design target | Minimize adaptation loss | Minimize ciphertext cost |
| Attention | Standard + poly softmax | Linear/polynomial-native |
| Packing | Afterthought (2% util.) | First-class (100% util.) |
| Depth management | Bootstrap when needed | Budget-aware architecture |
| Training | Plaintext → adapt | FHE-native loss from day 1 |
| Scheme co-design | Pick one scheme | CKKS+TFHE hybrid by design |

## Quantitative Motivation

From our analytical cost model (BERT-base, CKKS N=2^16):

| Metric | Standard Transformer | With FHE-native optimizations |
|---|---|---|
| Packing utilization | 2.3% | ~100% (multi-token packing) |
| FHE/GPU ratio | ~184,000× | ~1,600× |
| Attention cost share | 65% (Q·K^T dominated) | ~30% (linear attention) |
| Depth per block | 32 levels | Target: ≤8 levels |
| Bootstraps per forward | 12+ | Target: 1-2 |

## Closest Competitors & Differentiation

### vs. Liu & Liu (2023): "Privacy-Computing Friendly Transformers"
- They substitute operators in existing Transformers → 5× speedup
- We design from scratch → potential 100×+ improvement
- They don't address packing, depth budgeting, or training

### vs. PRISM (2026): "HE-Aware Pruning"
- They prune existing ResNets for CKKS → 45% rotation reduction
- We design FHE-native layers → rotations minimized by construction
- Their insight (only 1.1% of layers are fault-critical) validates our approach

### vs. SFPDML (2022): "FHE-Native Activation"
- They design ONE FHE-native activation function → 10× vs Taylor
- We extend this principle to the ENTIRE architecture
- They are the closest philosophical ancestor to CipherFormer

### vs. Chiang (2025): "FHE Training"
- They train a 4-layer NN under FHE
- Key insight: Sigmoid+BCE replaces Softmax for FHE training
- We scale this to Transformer-like architectures with FHE-native design

## Proposed Paper Sequence (Revised)

### Paper 1: "Why Transformers Are Not Homomorphic-Friendly"
**Type:** Measurement + Analysis
**Content:**
- Quantitative FHE cost breakdown of every Transformer operation
- Packing utilization analysis (2% → the hidden 50× tax)
- Depth budget analysis (1 block = 80% of modulus chain)
- Sensitivity analysis across d_model, seq_len, n_heads, poly_degree
- Comparison with existing adaptation approaches (THE-X, BOLT, Iron)
**Novelty:** First systematic, quantitative FHE bottleneck analysis of Transformers
**Target venue:** USENIX Security / CCS workshop / IACR ePrint

### Paper 2: "Design Principles for Encryption-Native Neural Networks"
**Type:** Principles + Framework
**Content:**
- FHE Complexity Theory (rotation, noise, depth, packing complexity)
- Design principles: pack-to-ciphertext, kill-the-outer-product, budget-depth
- FHE-native layer catalog: CipherLinear, PolyGate, LinearAttention
- FHE-aware training loss formulation
**Novelty:** First principled framework for FHE-native NN design
**Target venue:** NeurIPS / ICML (ML theory track)

### Paper 3: "CipherFormer: The First FHE-Native Transformer"
**Type:** Architecture + Implementation
**Content:**
- CipherFormer architecture (linear attention, packed dimensions, low-depth)
- FHE-native training on small language modeling task
- Benchmark vs adapted Transformers (THE-X style) on FHE
- Ablation: contribution of each design principle
**Novelty:** First Transformer designed from scratch for FHE, trained in FHE
**Target venue:** NeurIPS / ICML / USENIX Security

### Paper 4: "Cipher Compiler" (future)
### Paper 5: "Cipher Accelerator" (future)

## Immediate Next Steps

1. [ ] Fix Python version for OpenFHE empirical validation (pyenv install 3.12)
2. [ ] Validate analytical model against real OpenFHE measurements
3. [ ] Read the SFPDML and Chiang papers closely (closest ancestors)
4. [ ] Formalize FHE cost model as a complexity framework
5. [ ] Design first FHE-native attention mechanism (linear + packed)
6. [ ] Prototype: tiny CipherFormer (2 layers, d_model=slots) on toy task
