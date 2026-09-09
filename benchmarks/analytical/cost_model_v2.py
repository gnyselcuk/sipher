"""
Analytical FHE Cost Model for Transformer Operations — v2 (Calibrated)
======================================================================

Corrected version with:
1. Empirical reference timings from OpenFHE benchmarks
2. Correct SIMD operation counting (diagonal encoding for matmul)
3. Distinction between ct-ct and ct-pt multiplications

Reference timings (empirical, OpenFHE 1.5.1):
  N=32768 (medium):  Add=3.4ms  Mult_ct=51ms  Mult_pt=8.7ms  Rot=50ms
  N=65536 (large):   Add=15.8ms Mult_ct=161ms Mult_pt=30ms   Rot=120ms
"""

from dataclasses import dataclass
import math


@dataclass
class EmpiricalTimings:
    """Per-operation wall-clock timings from empirical benchmarks."""
    name: str
    N: int
    add_ms: float
    mult_ct_ms: float      # ciphertext × ciphertext
    mult_pt_ms: float      # ciphertext × plaintext
    rotate_ms: float
    bootstrap_ms: float = 0  # TODO: measure

    @classmethod
    def medium(cls):
        """N=32768, mult_depth=10, batch=4096."""
        return cls("medium_N32768", 32768,
                   add_ms=3.37, mult_ct_ms=50.96, mult_pt_ms=8.71,
                   rotate_ms=50.29, bootstrap_ms=5000)  # estimate

    @classmethod
    def large(cls):
        """N=65536, mult_depth=20, batch=8192."""
        return cls("large_N65536", 65536,
                   add_ms=15.81, mult_ct_ms=161.29, mult_pt_ms=29.97,
                   rotate_ms=120.15, bootstrap_ms=15000)  # estimate


@dataclass
class FHECost:
    """FHE cost profile using correct SIMD operation counts."""
    name: str
    mult_ct: int = 0       # ciphertext × ciphertext mults
    mult_pt: int = 0       # ciphertext × plaintext mults
    adds: int = 0
    rotations: int = 0
    depth: int = 0
    bootstraps: int = 0
    packing_util: float = 1.0
    notes: str = ""

    def wall_ms(self, t: EmpiricalTimings) -> float:
        return (
            self.mult_ct * t.mult_ct_ms
            + self.mult_pt * t.mult_pt_ms
            + self.adds * t.add_ms
            + self.rotations * t.rotate_ms
            + self.bootstraps * t.bootstrap_ms
        )

    def __str__(self):
        return (
            f"{self.name:35s} | "
            f"ct×ct={self.mult_ct:>8,}  ct×pt={self.mult_pt:>8,}  "
            f"adds={self.adds:>8,}  rots={self.rotations:>7,}  "
            f"depth={self.depth:>3}  boot={self.bootstraps}"
        )


@dataclass
class TransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    seq_len: int = 512
    n_layers: int = 12
    poly_degree_softmax: int = 8
    poly_degree_gelu: int = 6


@dataclass
class CKKSParams:
    N: int = 32768
    L: int = 40
    n_slots: int = 0

    def __post_init__(self):
        if self.n_slots == 0:
            self.n_slots = self.N // 2


def cost_linear(cfg, params, d_in, d_out, label="Linear"):
    """
    Matrix-vector multiply using diagonal encoding (Halevi-Shoup).

    Key insight: ONE plaintext-cipher mult processes ALL SIMD slots.
    With BSGS, cost is O(d_in) mults + O(√d_in) rotations,
    NOT O(d_out × d_in).

    All d_out outputs are computed simultaneously if d_out ≤ n_slots.
    """
    n = params.n_slots

    # Number of ciphertexts for input
    n_ct_in = math.ceil(d_in / n)

    # Diagonal encoding: d_in diagonals, each needs 1 ct-pt mult
    # With BSGS: ~2√d_in rotations instead of d_in
    bsgs_rotations = 2 * math.ceil(math.sqrt(d_in))

    # If d_out > n_slots, we need multiple passes
    n_out_groups = math.ceil(d_out / n)

    mult_pt = d_in * n_out_groups
    rotations = bsgs_rotations * n_out_groups
    adds = (d_in - 1) * n_out_groups  # accumulation

    packing_util = d_in / (n_ct_in * n)

    return FHECost(
        name=label,
        mult_pt=mult_pt,
        rotations=rotations,
        adds=adds,
        depth=1,
        packing_util=packing_util,
        notes=f"d_in={d_in}, d_out={d_out}, BSGS_rot={bsgs_rotations}, "
              f"out_groups={n_out_groups}",
    )


def cost_softmax(cfg, params, seq_len=0):
    """
    Softmax via polynomial exp + Newton-Raphson division.
    Applied per-row of attention matrix.
    """
    if seq_len == 0:
        seq_len = cfg.seq_len
    d = cfg.poly_degree_softmax
    n = params.n_slots
    n_ct = math.ceil(seq_len / n)

    # Polynomial exp: degree d, Horner's method
    # d ct-ct mults (squaring chain) + d adds
    exp_mult_ct = d * n_ct
    exp_depth = math.ceil(math.log2(d)) + 1

    # Sum reduction: log2(seq_len) rotations + adds
    log_seq = math.ceil(math.log2(seq_len))
    sum_rot = log_seq * n_ct
    sum_adds = log_seq * n_ct

    # Newton-Raphson division: ~4 iterations, 2 ct-ct mults each
    newton = 4
    div_mult_ct = 2 * newton * n_ct
    div_depth = newton

    total_depth = exp_depth + div_depth

    return FHECost(
        name="Softmax (per row)",
        mult_ct=exp_mult_ct + div_mult_ct,
        adds=sum_adds + d * n_ct + newton * n_ct,
        rotations=sum_rot,
        depth=total_depth,
        packing_util=seq_len / (n_ct * n),
        notes=f"poly_deg={d}, seq={seq_len}, depth={total_depth}",
    )


def cost_gelu(cfg, params):
    """GELU via polynomial approximation. Element-wise, no rotations."""
    d = cfg.poly_degree_gelu
    depth = math.ceil(math.log2(d)) + 1 + 1  # poly + final mult by x
    return FHECost(
        name="GELU (element-wise)",
        mult_ct=d + 1,
        adds=d,
        depth=depth,
        notes=f"poly_deg={d}",
    )


def cost_layernorm(cfg, params):
    """LayerNorm: mean reduction + variance + invsqrt polynomial + scale."""
    d = cfg.d_model
    n = params.n_slots
    n_ct = math.ceil(d / n)
    d_inv = 5  # invsqrt polynomial degree

    log_d = math.ceil(math.log2(d))

    # Mean: log(d) rotations + adds
    mean_rot = log_d * n_ct
    mean_adds = log_d * n_ct

    # Variance: square (ct-ct mult) + reduction
    var_mult_ct = n_ct
    var_rot = log_d * n_ct
    var_adds = (log_d + 1) * n_ct

    # Invsqrt polynomial: d_inv ct-ct mults
    inv_mult_ct = d_inv * n_ct
    inv_depth = math.ceil(math.log2(d_inv)) + 1

    # Scale + shift: 1 ct-pt mult + 1 add
    scale_mult_pt = n_ct
    scale_adds = n_ct

    total_depth = 1 + 1 + inv_depth + 1  # square + reduce + invsqrt + scale

    return FHECost(
        name="LayerNorm",
        mult_ct=var_mult_ct + inv_mult_ct,
        mult_pt=scale_mult_pt,
        adds=mean_adds + var_adds + scale_adds,
        rotations=mean_rot + var_rot,
        depth=total_depth,
        packing_util=d / (n_ct * n),
        notes=f"d_model={d}, invsqrt_deg={d_inv}",
    )


def cost_attention(cfg, params):
    """
    Full multi-head attention.

    Components:
    1. QKV projections: 3 linear layers (ct-pt mults)
    2. Q·K^T per head: matrix-matrix product
    3. Scale: ct-pt mult (cheap)
    4. Softmax per row: ct-ct mults (expensive)
    5. Attn·V per head: matrix-matrix product
    """
    d_k = cfg.d_model // cfg.n_heads
    seq = cfg.seq_len
    n = params.n_slots
    nh = cfg.n_heads

    # 1. QKV: 3 × Linear(d_model → d_model)
    qkv = cost_linear(cfg, params, cfg.d_model, cfg.d_model, "QKV")
    qkv.mult_pt *= 3
    qkv.adds *= 3
    qkv.rotations *= 3

    # 2. Q·K^T per head: (seq × d_k) × (d_k × seq) → (seq × seq)
    # BOTH Q and K are encrypted (derived from user input) → ct-ct mults!
    # Per output element: 1 ct-ct mult (element-wise) + log(d_k) reduction
    # With SIMD packing of d_k elements:
    #   ct-ct mults per output row: ceil(d_k / n_slots) ≈ 1 (d_k << n_slots)
    #   Reduction: log(d_k) rotations per output element
    # Total per head: seq² output elements
    n_ct_seq = math.ceil(seq / n)
    log_dk = math.ceil(math.log2(d_k))

    # ct-ct mults: seq² per head (one per output element, SIMD handles d_k)
    qk_mult_ct = seq * seq * nh
    # Rotations for reduction: log(d_k) per output element
    qk_rot = seq * seq * log_dk * nh
    qk_adds = seq * seq * log_dk * nh

    # 3. Scale: 1 ct-pt mult per row (scale factor is plaintext)
    scale_mult_pt = seq * nh

    # 4. Softmax: per head, per row — all ct-ct (encrypted data)
    sm = cost_softmax(cfg, params, seq)
    sm_mult_ct = sm.mult_ct * nh * seq
    sm_adds = sm.adds * nh * seq
    sm_rot = sm.rotations * nh * seq
    sm_depth = sm.depth

    # 5. Attn·V: (seq × seq) × (seq × d_k) → (seq × d_k)
    # BOTH A (attention weights) and V are encrypted → ct-ct mults!
    # Per output element: 1 ct-ct mult + log(seq) reduction
    log_seq = math.ceil(math.log2(seq))
    av_mult_ct = seq * d_k * nh  # seq×d_k output elements
    av_rot = seq * d_k * log_seq * nh
    av_adds = seq * d_k * log_seq * nh

    total_depth = 1 + 1 + sm_depth + 1  # QKV + QK + softmax + AV

    return FHECost(
        name="Attention (full, all heads)",
        mult_ct=qk_mult_ct + sm_mult_ct + av_mult_ct,
        mult_pt=qkv.mult_pt + scale_mult_pt,
        adds=qkv.adds + qk_adds + sm_adds + av_adds,
        rotations=qkv.rotations + qk_rot + sm_rot + av_rot,
        depth=total_depth,
        packing_util=seq / (n_ct_seq * n),
        notes=f"nh={nh}, d_k={d_k}, seq={seq}, "
              f"QK_ct={qk_mult_ct:,}, AV_ct={av_mult_ct:,}, sm_ct={sm_mult_ct:,}",
    )


def cost_ffn(cfg, params):
    """FFN: Linear(d→4d) + GELU + Linear(4d→d)."""
    lin1 = cost_linear(cfg, params, cfg.d_model, cfg.d_ff, "FFN_up")
    gelu = cost_gelu(cfg, params)
    lin2 = cost_linear(cfg, params, cfg.d_ff, cfg.d_model, "FFN_down")

    # GELU is element-wise on d_ff values
    n_ct_ff = math.ceil(cfg.d_ff / params.n_slots)
    gelu_mult_ct = gelu.mult_ct * n_ct_ff
    gelu_adds = gelu.adds * n_ct_ff

    return FHECost(
        name="FFN (up + GELU + down)",
        mult_ct=gelu_mult_ct,
        mult_pt=lin1.mult_pt + lin2.mult_pt,
        adds=lin1.adds + gelu_adds + lin2.adds,
        rotations=lin1.rotations + lin2.rotations,
        depth=lin1.depth + gelu.depth + lin2.depth,
        notes=f"d={cfg.d_model}→{cfg.d_ff}→{cfg.d_model}",
    )


def cost_transformer_block(cfg, params):
    """Full block: LN → Attn → Residual → LN → FFN → Residual."""
    ln1 = cost_layernorm(cfg, params)
    attn = cost_attention(cfg, params)
    ln2 = cost_layernorm(cfg, params)
    ffn = cost_ffn(cfg, params)

    components = [ln1, attn, ln2, ffn]

    return FHECost(
        name="Transformer Block",
        mult_ct=sum(c.mult_ct for c in components),
        mult_pt=sum(c.mult_pt for c in components),
        adds=sum(c.adds for c in components),
        rotations=sum(c.rotations for c in components),
        depth=sum(c.depth for c in components),
        bootstraps=sum(c.bootstraps for c in components),
        notes=f"d={cfg.d_model}, nh={cfg.n_heads}, d_ff={cfg.d_ff}",
    )


def run_analysis():
    params = CKKSParams()
    cfg = TransformerConfig()
    t_med = EmpiricalTimings.medium()
    t_large = EmpiricalTimings.large()

    print("=" * 100)
    print("FHE COST MODEL v2 — CALIBRATED WITH EMPIRICAL TIMINGS")
    print(f"CKKS: N={params.N}, L={params.L}, slots={params.n_slots}")
    print(f"Transformer: d={cfg.d_model}, nh={cfg.n_heads}, d_ff={cfg.d_ff}, "
          f"seq={cfg.seq_len}, layers={cfg.n_layers}")
    print("=" * 100)

    # Individual operations
    ops = [
        cost_linear(cfg, params, cfg.d_model, cfg.d_model, "Linear (d→d)"),
        cost_linear(cfg, params, cfg.d_model, cfg.d_ff, "Linear (d→4d)"),
        cost_linear(cfg, params, cfg.d_ff, cfg.d_model, "Linear (4d→d)"),
        cost_softmax(cfg, params),
        cost_gelu(cfg, params),
        cost_layernorm(cfg, params),
        cost_attention(cfg, params),
        cost_ffn(cfg, params),
    ]

    print("\n--- Individual Operations ---\n")
    for op in ops:
        print(op)

    # Full block
    block = cost_transformer_block(cfg, params)
    print(f"\n{'=' * 100}")
    print("\n--- Transformer Block ---\n")
    print(block)

    # Wall time with both timing sets
    for timings in [t_med, t_large]:
        wall = block.wall_ms(timings)
        print(f"\n  Wall time ({timings.name}): {wall:,.0f} ms = {wall/1000:,.1f} s")

    # Full model
    print(f"\n{'=' * 100}")
    print(f"\n--- Full Model ({cfg.n_layers} layers) ---\n")

    for timings in [t_med, t_large]:
        block_ms = block.wall_ms(timings)
        model_ms = block_ms * cfg.n_layers

        # Bootstrap overhead
        depth_per_block = block.depth
        boots_per_layer = max(1, math.ceil(depth_per_block / params.L))
        total_boots = boots_per_layer * cfg.n_layers
        boot_ms = total_boots * timings.bootstrap_ms
        total_ms = model_ms + boot_ms

        print(f"  [{timings.name}]")
        print(f"    Block wall time:      {block_ms:>15,.0f} ms  ({block_ms/1000:,.1f} s)")
        print(f"    Model compute:        {model_ms:>15,.0f} ms  ({model_ms/1000:,.1f} s)")
        print(f"    Bootstraps:           {total_boots:>15}")
        print(f"    Bootstrap overhead:   {boot_ms:>15,.0f} ms  ({boot_ms/1000:,.1f} s)")
        print(f"    TOTAL:                {total_ms:>15,.0f} ms  ({total_ms/1000:,.1f} s = {total_ms/3600000:.2f} hr)")
        print(f"    GPU reference:        {'~120 ms':>15}")
        print(f"    FHE/GPU ratio:        {total_ms/120:>15,.0f}×")
        print()

    # Bottleneck breakdown
    print(f"{'=' * 100}")
    print("\n--- Bottleneck Breakdown (medium timings) ---\n")

    components = {
        "LayerNorm (×2)": cost_layernorm(cfg, params),
        "Attention": cost_attention(cfg, params),
        "FFN": cost_ffn(cfg, params),
    }
    # Double LN
    components["LayerNorm (×2)"].mult_ct *= 2
    components["LayerNorm (×2)"].mult_pt *= 2
    components["LayerNorm (×2)"].adds *= 2
    components["LayerNorm (×2)"].rotations *= 2

    total_ms = sum(c.wall_ms(t_med) for c in components.values())
    for name, c in sorted(components.items(), key=lambda x: -x[1].wall_ms(t_med)):
        ms = c.wall_ms(t_med)
        pct = ms / total_ms * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:25s} {pct:6.1f}%  {bar}  ({ms/1000:,.1f} s)")

    # Attention sub-breakdown
    print(f"\n--- Attention Sub-breakdown ---\n")
    d_k = cfg.d_model // cfg.n_heads
    nh = cfg.n_heads
    seq = cfg.seq_len
    n = params.n_slots
    log_dk = math.ceil(math.log2(d_k))
    log_seq = math.ceil(math.log2(seq))

    qkv = cost_linear(cfg, params, cfg.d_model, cfg.d_model, "QKV_proj")
    qkv.mult_pt *= 3; qkv.adds *= 3; qkv.rotations *= 3

    qk = FHECost("Q·K^T (ct-ct!)",
                  mult_ct=seq*seq*nh,
                  rotations=seq*seq*log_dk*nh,
                  adds=seq*seq*log_dk*nh)
    sm = cost_softmax(cfg, params, seq)
    sm_total = FHECost("Softmax (ct-ct)",
                       mult_ct=sm.mult_ct*nh*seq,
                       adds=sm.adds*nh*seq,
                       rotations=sm.rotations*nh*seq)
    av = FHECost("Attn·V (ct-ct!)",
                  mult_ct=seq*d_k*nh,
                  rotations=seq*d_k*log_seq*nh,
                  adds=seq*d_k*log_seq*nh)

    attn_parts = {"QKV_proj (ct-pt)": qkv, "Q·K^T (ct-ct)": qk,
                  "Softmax (ct-ct)": sm_total, "Attn·V (ct-ct)": av}
    attn_total_ms = sum(c.wall_ms(t_med) for c in attn_parts.values())
    for name, c in sorted(attn_parts.items(), key=lambda x: -x[1].wall_ms(t_med)):
        ms = c.wall_ms(t_med)
        pct = ms / attn_total_ms * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:25s} {pct:6.1f}%  {bar}  ({ms/1000:,.1f} s)")

    # Multi-token packing analysis
    print(f"\n{'=' * 100}")
    print("\n--- Multi-token Packing (the FHE-native advantage) ---\n")
    tokens_per_ct = params.n_slots // cfg.d_model
    print(f"  Tokens per ciphertext: {tokens_per_ct}")
    print(f"  Effective per-token cost: {block.wall_ms(t_med)/tokens_per_ct:,.0f} ms")
    model_packed_ms = block.wall_ms(t_med) * cfg.n_layers / tokens_per_ct
    print(f"  Full model (packed): {model_packed_ms/1000:,.1f} s")
    print(f"  FHE/GPU ratio (packed): {model_packed_ms/120:,.0f}×")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    run_analysis()
