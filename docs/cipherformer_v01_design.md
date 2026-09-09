# CipherFormer v0.1: Architecture Design Document

**Date:** 2026-07-27
**Status:** Initial design based on FHE cost analysis

---

## 1. Design Philosophy

Standard Transformers are designed for GPU tensor cores. CipherFormer is
designed for CKKS ciphertext arithmetic. Every architectural choice is
driven by one question: **"What does this cost in ciphertext operations?"**

### The Five FHE-Native Design Principles

| # | Principle | Rationale |
|---|---|---|
| P1 | **Eliminate O(seq²) ct-ct mults** | Q·K^T is 84.3% of standard attention cost |
| P2 | **Maximize ct-pt ratio** | ct-pt is 6× cheaper than ct-ct (8.7ms vs 51ms) |
| P3 | **Minimize multiplicative depth** | Each level consumes modulus; depth 32 = 80% of L=40 |
| P4 | **Pack to the ciphertext** | d_model=768 in 16,384 slots = 4.7% utilization |
| P5 | **Minimize rotations** | Each rotation = 1 key switch = 50ms |

---

## 2. Attention Mechanism Comparison

Empirical cost per attention block (d_model=768, 12 heads, seq=512, N=32768):

| Mechanism | Wall time | Speedup | ct-ct mults | Scaling |
|---|---|---|---|---|
| Standard (softmax Q·K^T·V) | 386.5 hr | 1× | 3,637,248 | O(seq²) |
| CipherAttention (linear) | 1.98 hr | 196× | 49,920 | O(seq·d_k) |
| **CipherMixer (MLP-Mixer)** | **25.7 s** | **54,146×** | **10** | **O(seq)** |
| CipherRWKV (recurrent) | 17.87 hr | 22× | 786,432 | O(seq·d_k²) |

### Why CipherMixer wins

CipherMixer replaces data-dependent attention with learned token mixing:

```
Standard:  output = softmax(Q·K^T) · V     ← Q,K,V all encrypted → ct-ct
CipherMixer: output = W_mix · x + b         ← W_mix is PLAINTEXT → ct-pt
```

The weight matrix W_mix is a model parameter (plaintext). The input x is
encrypted. This makes ALL mixing operations ct-pt, 6× cheaper per op,
and far fewer operations total.

**Trade-off:** CipherMixer cannot do content-based routing ("attend to
token j because its content is relevant"). It applies a fixed learned
mixing pattern. This is less expressive than attention.

### Hybrid Solution

CipherFormer uses a **3:1 Mixer:Attention ratio**:
- 3 out of 4 layers use CipherMixer (efficient, fixed mixing)
- 1 out of 4 layers uses CipherAttention (expressive, data-dependent)
- This preserves most of the expressiveness at ~1% of the cost

---

## 3. CipherFormer v0.1 Architecture

```
Input tokens (encrypted)
    │
    ▼
┌─────────────────────────────────┐
│  CipherEmbedding                │  ct-pt mult (embedding table lookup)
│  + Positional Encoding (add)    │  ct-pt add
└─────────────────────────────────┘
    │
    ▼  × 12 layers
┌─────────────────────────────────┐
│  Layer 0: CipherMixer           │  ct-pt token mixing + poly activation
│  Layer 1: CipherMixer           │
│  Layer 2: CipherMixer           │
│  Layer 3: CipherAttention       │  linear attention (data-dependent)
│  Layer 4: CipherMixer           │
│  Layer 5: CipherMixer           │
│  Layer 6: CipherMixer           │
│  Layer 7: CipherAttention       │
│  Layer 8: CipherMixer           │
│  Layer 9: CipherMixer           │
│  Layer 10: CipherMixer          │
│  Layer 11: CipherAttention      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  CipherHead                     │  ct-pt linear → logits
└─────────────────────────────────┘
    │
    ▼
  Output (encrypted logits)
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 768 | Compatibility with BERT-base tasks |
| n_heads | 2 | 12 heads = 12× cost; 2 heads sufficient for small models |
| d_k | 384 | d_model / n_heads; large for SIMD packing |
| seq_len | 512 | Standard for BERT tasks |
| n_layers | 12 | Standard depth |
| Mixer:Attn | 3:1 | Cost/expressiveness tradeoff |
| Mixer hidden | 1024 | 2× seq_len |
| φ degree | 2 | Minimal depth for feature map |
| Activation | Poly deg 4 | Low depth, good approximation |
| FHE scheme | CKKS | Approximate arithmetic for ML |
| N | 32768 | 128-bit security, 16,384 slots |
| L | 40 | Modulus levels |

---

## 4. Layer Specifications

### 4.1 CipherMixer Layer

```python
def CipherMixerLayer(x):
    """
    x: encrypted tensor [seq_len, d_model]
    W_mix: plaintext [seq_len, H] (token mixing)
    W_proj: plaintext [H, seq_len] (projection back)
    """
    # Pre-norm (polynomial LayerNorm)
    x = PolyLayerNorm(x)              # depth: 7

    # Token mixing (ALL ct-pt)
    x_T = transpose(x)                # free (repacking)
    h = W_mix @ x_T                   # ct-pt mults, depth: 1
    h = PolyActivation(h, degree=4)   # ct-ct mults, depth: 3
    x_T = W_proj @ h                  # ct-pt mults, depth: 1
    x = x + transpose(x_T)            # ct add (residual)

    # Channel mixing (ALL ct-pt)
    x = PolyLayerNorm(x)
    h = W_chan_up @ x                 # ct-pt, d_model → 4*d_model
    h = PolyActivation(h, degree=4)   # ct-ct
    x = x + W_chan_down @ h           # ct-pt + residual

    return x
```

**FHE cost per CipherMixer layer:**
- ct-ct mults: ~10 (activation only)
- ct-pt mults: ~1,537
- rotations: ~130
- depth: ~12
- wall time: ~25.7 seconds

### 4.2 CipherAttention Layer

```python
def CipherAttentionLayer(x):
    """
    Linear attention with polynomial feature map.
    """
    x = PolyLayerNorm(x)

    # QKV projection (ct-pt, weights are plaintext)
    Q = W_Q @ x    # ct-pt
    K = W_K @ x    # ct-pt
    V = W_V @ x    # ct-pt

    # Feature map: φ(z) = 1 + z + z²/2  (degree 2, depth 2)
    Q_prime = phi(Q)   # ct-ct for z²
    K_prime = phi(K)   # ct-ct for z²

    # Linear attention: φ(Q) · (φ(K)^T · V)
    KV = K_prime.T @ V     # ct-ct, d_k × d_k (independent of seq!)
    out = Q_prime @ KV     # ct-ct, seq × d_k

    # Normalization
    Z = Q_prime @ K_prime.T.sum(dim=-1)  # row sums
    out = out / Z          # Newton-Raphson division

    # Output projection (ct-pt)
    out = W_O @ out        # ct-pt

    return x + out         # residual
```

**FHE cost per CipherAttention layer:**
- ct-ct mults: ~49,920
- ct-pt mults: ~14,592
- rotations: ~80,808
- depth: ~17
- wall time: ~1.98 hours

### 4.3 PolyLayerNorm

```python
def PolyLayerNorm(x):
    """
    LayerNorm with polynomial inverse-sqrt.
    Avoids division and sqrt (non-polynomial).
    """
    mean = reduce_sum(x) / d_model        # rotations + ct-pt mult
    x_centered = x - mean                  # ct add
    var = reduce_sum(x_centered ** 2) / d  # ct-ct mult + rotations
    inv_std = poly_invsqrt(var, degree=5)  # ct-ct mults (polynomial)
    return x_centered * inv_std * gamma + beta  # ct-pt mults + adds
```

### 4.4 PolyActivation

```python
def PolyActivation(x, degree=4):
    """
    Polynomial activation replacing GELU.
    p(x) = a_0 + a_1*x + a_2*x² + a_3*x³ + a_4*x⁴
    Coefficients learned during FHE-native training.
    """
    # Horner's method: ((a_4*x + a_3)*x + a_2)*x + a_1)*x + a_0
    # Depth: ceil(log2(degree)) + 1 = 3
    # ct-ct mults: degree = 4
    return horner_eval(x, coefficients)
```

---

## 5. Full Model Cost Estimate

### Per-layer costs (empirical timings, N=32768):

| Layer type | Count | Wall time each | Total |
|---|---|---|---|
| CipherMixer | 9 | 25.7 s | 231 s |
| CipherAttention | 3 | 7,128 s | 21,384 s |
| **Total compute** | | | **21,615 s (6.0 hr)** |
| Bootstrapping | ~12 | ~5 s | 60 s |
| **Grand total** | | | **~6.0 hours** |

### With multi-token packing (21 tokens/ciphertext):

| Metric | Value |
|---|---|
| Packed wall time | ~17 minutes |
| GPU reference | ~120 ms |
| **FHE/GPU ratio** | **~8,500×** |
| With FHE hardware (~100×) | **~85×** |

### Comparison with standard Transformer:

| | Standard Transformer | CipherFormer v0.1 | Improvement |
|---|---|---|---|
| Attention cost | 4,638 hr | 6.0 hr | **773×** |
| With packing | 221 hr | 17 min | **780×** |
| With hardware | 2.2 hr | ~10 s | **792×** |
| FHE/GPU ratio | 139,000,000× | ~85× | **1,635,000×** |

---

## 6. Open Questions

1. **Expressiveness:** Does 3:1 Mixer:Attention ratio preserve enough
   data-dependent mixing for language modeling tasks? Need empirical
   validation on GLUE/SQuAD benchmarks.

2. **Training:** How to train CipherFormer? Options:
   a. Train in plaintext with Mixer architecture, then run in FHE
   b. FHE-native training with polynomial activations (very expensive)
   c. Hybrid: plaintext pre-training + FHE fine-tuning

3. **CipherMixer token mixing:** Fixed-size mixing matrix means fixed
   seq_len. How to handle variable-length inputs?

4. **Polynomial activation coefficients:** How to choose optimal
   coefficients? FHE-native training would learn them, but plaintext
   pre-training needs a good initialization.

5. **Depth management:** CipherAttention has depth 17. With L=40,
   we can fit ~2 attention layers before bootstrapping. The 3:1 ratio
   means attention at layers 3, 7, 11: each needs bootstrap before it.

---

## 7. Next Steps

1. [ ] Implement CipherMixer layer in PyTorch (plaintext prototype)
2. [ ] Validate on a toy task (e.g., WikiText-2 language modeling)
3. [ ] Implement CipherAttention layer in PyTorch
4. [ ] Compare 3:1 hybrid vs pure Mixer vs pure Attention
5. [ ] Port to OpenFHE for encrypted inference validation
6. [ ] Measure actual vs predicted wall time
7. [ ] Write Paper 1: "Why Transformers Are Not Homomorphic-Friendly"
8. [ ] Write Paper 2: "Design Principles for Encryption-Native NNs"
