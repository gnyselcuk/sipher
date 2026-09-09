"""
Analytical FHE Cost Model for Transformer Operations
=====================================================

Computes the theoretical FHE cost of each Transformer operation under CKKS,
without running any actual FHE computation.

Cost dimensions:
  - mults:       number of ciphertext multiplications
  - adds:        number of ciphertext additions
  - rotations:   number of ciphertext rotations (each = 1 key switch)
  - key_switches: total key-switching operations
  - depth:       multiplicative depth consumed
  - rescales:    number of rescale operations (≈ mults)
  - bootstraps:  number of bootstrapping operations needed
  - wall_estimate: rough wall-clock estimate (seconds) on modern CPU

Reference costs (CKKS, N=2^16, L=40, Δ=50-bit):
  - Add:       ~0.01 ms
  - Mult:      ~0.1 ms  (+ 1 rescale)
  - Rotation:  ~0.5 ms  (1 key switch)
  - Bootstrap: ~50 ms   (consumes ~15 levels, refreshes ~25)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class FHECost:
    """FHE cost profile of a single operation."""
    name: str
    mults: int = 0
    adds: int = 0
    rotations: int = 0
    key_switches: int = 0
    depth: int = 0
    rescales: int = 0
    bootstraps: int = 0
    packing_util: float = 1.0  # fraction of SIMD slots used
    notes: str = ""

    # Reference timings (ms) for CKKS N=2^16
    T_ADD: float = 0.01
    T_MULT: float = 0.1
    T_ROT: float = 0.5
    T_BOOT: float = 50.0

    @property
    def wall_ms(self) -> float:
        return (
            self.adds * self.T_ADD
            + self.mults * self.T_MULT
            + self.rotations * self.T_ROT
            + self.bootstraps * self.T_BOOT
        )

    @property
    def total_ops(self) -> int:
        return self.mults + self.adds + self.rotations + self.bootstraps

    def __str__(self) -> str:
        return (
            f"{self.name:30s} | "
            f"mults={self.mults:>8,}  adds={self.adds:>8,}  "
            f"rots={self.rotations:>8,}  depth={self.depth:>3}  "
            f"boot={self.bootstraps:>2}  pack={self.packing_util:.0%}  "
            f"~{self.wall_ms:>10,.1f} ms"
        )


@dataclass
class CKKSParams:
    """CKKS scheme parameters."""
    N: int = 2**16          # ring dimension
    L: int = 40             # total modulus levels
    delta: int = 50         # scaling factor bits
    n_slots: int = 0        # SIMD slots (N/2 for CKKS)

    def __post_init__(self):
        if self.n_slots == 0:
            self.n_slots = self.N // 2


@dataclass
class TransformerConfig:
    """Standard Transformer block configuration."""
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072        # typically 4 * d_model
    seq_len: int = 512
    vocab_size: int = 30522
    n_layers: int = 12
    poly_degree_softmax: int = 8   # degree of polynomial approx for softmax
    poly_degree_gelu: int = 6      # degree of polynomial approx for GELU
    poly_degree_sigmoid: int = 6
    poly_degree_invsqrt: int = 5   # for LayerNorm 1/sqrt(x)
    newton_iters_div: int = 4      # Newton-Raphson iterations for division


def cost_linear(cfg: TransformerConfig, params: CKKSParams,
                d_in: int, d_out: int, label: str = "Linear") -> FHECost:
    """
    Matrix-vector multiply: y = W·x + b
    x packed in ciphertext(s), W encoded as plaintext.

    With BSGS (baby-step giant-step) optimization:
      rotations ≈ 2 * sqrt(d_in) per output element group
      mults ≈ d_in (one per input element, accumulated)

    Without BSGS:
      rotations ≈ d_in per output row
    """
    n = params.n_slots

    # How many ciphertexts to hold the input
    n_ct_in = math.ceil(d_in / n)

    # Multiplications: d_out rows × d_in elements (plaintext-cipher mult)
    mults = d_out * d_in

    # Additions: d_out rows × (d_in - 1) accumulations
    adds = d_out * (d_in - 1)

    # Rotations with BSGS: for each output row, ~2*sqrt(d_in) rotations
    # to align input elements with weight diagonals
    bsgs_rot_per_row = 2 * math.ceil(math.sqrt(d_in))
    rotations = d_out * bsgs_rot_per_row

    # Packing utilization: how well d_in fills the SIMD slots
    packing_util = d_in / (n_ct_in * n) if n_ct_in > 0 else 0

    return FHECost(
        name=label,
        mults=mults,
        adds=adds,
        rotations=rotations,
        key_switches=rotations,  # each rotation = 1 key switch
        depth=1,
        rescales=1,
        packing_util=packing_util,
        notes=f"d_in={d_in}, d_out={d_out}, BSGS rot/row={bsgs_rot_per_row}",
    )


def cost_softmax(cfg: TransformerConfig, params: CKKSParams,
                 seq_len: int = 0) -> FHECost:
    """
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Steps:
    1. exp(x) via polynomial approximation (degree d)
    2. Sum reduction across seq_len elements
    3. Division (Newton-Raphson)

    This is the MOST EXPENSIVE operation in FHE Transformers.
    """
    if seq_len == 0:
        seq_len = cfg.seq_len
    d = cfg.poly_degree_softmax
    n = params.n_slots
    newton = cfg.newton_iters_div

    # Step 1: Polynomial evaluation for exp(x)
    # Degree-d polynomial needs d multiplications (Horner's method)
    # Applied to seq_len elements (packed in ciphertexts)
    n_ct = math.ceil(seq_len / n)
    exp_mults = d * n_ct
    exp_depth = math.ceil(math.log2(d)) + 1  # baby-step giant-step eval
    exp_adds = d * n_ct

    # Step 2: Sum reduction
    # log2(seq_len) rotations + additions to reduce
    reduction_steps = math.ceil(math.log2(seq_len))
    sum_rotations = reduction_steps * n_ct
    sum_adds = reduction_steps * n_ct

    # Step 3: Division via Newton-Raphson
    # Each iteration: ~2 mults + 1 add
    # Depth: newton iterations
    div_mults = 2 * newton * n_ct
    div_adds = newton * n_ct
    div_depth = newton

    total_depth = exp_depth + div_depth
    # Check if bootstrapping is needed
    bootstraps = max(0, math.ceil(total_depth / params.L) - 1)

    return FHECost(
        name="Softmax",
        mults=exp_mults + div_mults,
        adds=exp_adds + sum_adds + div_adds,
        rotations=sum_rotations,
        key_switches=sum_rotations,
        depth=total_depth,
        rescales=total_depth,
        bootstraps=bootstraps,
        packing_util=seq_len / (n_ct * n),
        notes=f"poly_deg={d}, seq_len={seq_len}, newton_iters={newton}, "
              f"exp_depth={exp_depth}, div_depth={div_depth}",
    )


def cost_gelu(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    GELU(x) ≈ x * sigmoid(1.702 * x)

    Polynomial approximation of sigmoid, then multiply by x.
    """
    d = cfg.poly_degree_gelu
    n = params.n_slots

    # Polynomial evaluation: d mults (Horner)
    poly_mults = d
    poly_depth = math.ceil(math.log2(d)) + 1
    poly_adds = d

    # Final multiply by x: 1 mult
    total_mults = poly_mults + 1
    total_depth = poly_depth + 1

    return FHECost(
        name="GELU",
        mults=total_mults,
        adds=poly_adds,
        depth=total_depth,
        rescales=total_depth,
        packing_util=1.0,
        notes=f"poly_deg={d}, element-wise",
    )


def cost_layernorm(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    LayerNorm(x) = (x - mean) / sqrt(variance + eps) * gamma + beta

    Steps:
    1. Compute mean: sum reduction + division by d_model
    2. Compute variance: (x - mean)^2, sum reduction, division
    3. Inverse sqrt: polynomial approximation
    4. Scale and shift
    """
    d = cfg.d_model
    n = params.n_slots
    d_inv = cfg.poly_degree_invsqrt
    n_ct = math.ceil(d / n)

    # Step 1: Mean — sum reduction
    mean_steps = math.ceil(math.log2(d))
    mean_rots = mean_steps * n_ct
    mean_adds = mean_steps * n_ct

    # Step 2: Variance — subtract mean, square, reduce
    var_mults = n_ct  # squaring
    var_rots = mean_steps * n_ct  # another reduction
    var_adds = mean_steps * n_ct + n_ct  # reduction + subtract

    # Step 3: Inverse sqrt via polynomial
    invsqrt_mults = d_inv * n_ct
    invsqrt_depth = math.ceil(math.log2(d_inv)) + 1

    # Step 4: Scale (multiply by invsqrt * gamma) + shift (add beta)
    scale_mults = n_ct
    scale_adds = n_ct

    total_depth = 1 + 1 + invsqrt_depth + 1  # square + reduce + invsqrt + scale
    bootstraps = max(0, math.ceil(total_depth / params.L) - 1)

    return FHECost(
        name="LayerNorm",
        mults=var_mults + invsqrt_mults + scale_mults,
        adds=mean_adds + var_adds + scale_adds,
        rotations=mean_rots + var_rots,
        key_switches=mean_rots + var_rots,
        depth=total_depth,
        rescales=total_depth,
        bootstraps=bootstraps,
        packing_util=d / (n_ct * n),
        notes=f"d_model={d}, invsqrt_deg={d_inv}",
    )


def cost_attention(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    Scaled Dot-Product Attention:
      Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    Components:
    1. QKV projections: 3 linear layers
    2. Q·K^T: matrix multiply (seq_len × d_k) × (d_k × seq_len)
    3. Scale: multiply by 1/sqrt(d_k) — cheap
    4. Softmax: expensive
    5. ·V: matrix multiply (seq_len × seq_len) × (seq_len × d_k)
    """
    d_k = cfg.d_model // cfg.n_heads
    seq = cfg.seq_len
    n = params.n_slots

    # QKV projections: 3 × Linear(d_model, d_model)
    qkv = cost_linear(cfg, params, cfg.d_model, cfg.d_model, "QKV_proj")
    qkv.mults *= 3
    qkv.adds *= 3
    qkv.rotations *= 3
    qkv.key_switches *= 3

    # Q·K^T: for each head, (seq × d_k) · (d_k × seq) → (seq × seq)
    # Per head: seq × seq output elements, each needs d_k mults + reductions
    # With packing: seq elements per ciphertext row
    n_ct_seq = math.ceil(seq / n)
    qk_mults_per_head = seq * d_k * n_ct_seq
    qk_adds_per_head = seq * (d_k - 1) * n_ct_seq
    qk_rots_per_head = seq * 2 * math.ceil(math.sqrt(d_k)) * n_ct_seq
    qk_mults = qk_mults_per_head * cfg.n_heads
    qk_adds = qk_adds_per_head * cfg.n_heads
    qk_rots = qk_rots_per_head * cfg.n_heads

    # Scale: 1 mult per element (cheap)
    scale_mults = seq * seq * cfg.n_heads

    # Softmax: applied per head per row
    sm = cost_softmax(cfg, params, seq)
    sm_total_mults = sm.mults * cfg.n_heads * seq
    sm_total_adds = sm.adds * cfg.n_heads * seq
    sm_total_rots = sm.rotations * cfg.n_heads * seq
    sm_depth = sm.depth
    sm_bootstraps = sm.bootstraps * cfg.n_heads

    # Attention · V: (seq × seq) · (seq × d_k) → (seq × d_k)
    av_mults_per_head = seq * seq * n_ct_seq
    av_adds_per_head = seq * (seq - 1) * n_ct_seq
    av_rots_per_head = seq * 2 * math.ceil(math.sqrt(seq)) * n_ct_seq
    av_mults = av_mults_per_head * cfg.n_heads
    av_adds = av_adds_per_head * cfg.n_heads
    av_rots = av_rots_per_head * cfg.n_heads

    total_depth = 1 + 1 + sm_depth + 1  # QKV + QK + softmax + AV
    total_bootstraps = sm_bootstraps + max(0, math.ceil(total_depth / params.L) - 1)

    return FHECost(
        name="Attention (full)",
        mults=qkv.mults + qk_mults + scale_mults + sm_total_mults + av_mults,
        adds=qkv.adds + qk_adds + sm_total_adds + av_adds,
        rotations=qkv.rotations + qk_rots + sm_total_rots + av_rots,
        key_switches=qkv.key_switches + qk_rots + sm_total_rots + av_rots,
        depth=total_depth,
        rescales=total_depth,
        bootstraps=total_bootstraps,
        packing_util=seq / (n_ct_seq * n),
        notes=f"n_heads={cfg.n_heads}, d_k={d_k}, seq={seq}",
    )


def cost_ffn(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    Feed-Forward Network:
      FFN(x) = GELU(x·W1 + b1)·W2 + b2

    Two linear layers with GELU in between.
    """
    lin1 = cost_linear(cfg, params, cfg.d_model, cfg.d_ff, "FFN_up")
    gelu = cost_gelu(cfg, params)
    lin2 = cost_linear(cfg, params, cfg.d_ff, cfg.d_model, "FFN_down")

    return FHECost(
        name="FFN (2-layer + GELU)",
        mults=lin1.mults + gelu.mults + lin2.mults,
        adds=lin1.adds + gelu.adds + lin2.adds,
        rotations=lin1.rotations + lin2.rotations,
        key_switches=lin1.key_switches + lin2.key_switches,
        depth=lin1.depth + gelu.depth + lin2.depth,
        rescales=lin1.depth + gelu.depth + lin2.depth,
        notes=f"d_model={cfg.d_model} → d_ff={cfg.d_ff} → d_model={cfg.d_model}",
    )


def cost_embedding(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    Embedding lookup + positional encoding.
    In FHE: lookup is expensive (PIR-like or one-hot encoding).
    Positional encoding: plaintext addition (cheap if positions are public).
    """
    n = params.n_slots
    # One-hot approach: vocab_size multiplications per token
    # PIR approach: log(vocab_size) rotations
    one_hot_mults = cfg.vocab_size
    pir_rots = math.ceil(math.log2(cfg.vocab_size))

    return FHECost(
        name="Embedding (one-hot)",
        mults=one_hot_mults,
        rotations=pir_rots,
        key_switches=pir_rots,
        depth=1,
        rescales=1,
        packing_util=cfg.d_model / n,
        notes=f"vocab={cfg.vocab_size}, one-hot mults={one_hot_mults}, "
              f"PIR rots={pir_rots}",
    )


def cost_residual(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """Residual connection: just an addition. Cheapest operation."""
    return FHECost(
        name="Residual Add",
        adds=1,
        depth=0,
        notes="Essentially free in FHE",
    )


def cost_transformer_block(cfg: TransformerConfig, params: CKKSParams) -> FHECost:
    """
    Full Transformer block:
      x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """
    ln1 = cost_layernorm(cfg, params)
    attn = cost_attention(cfg, params)
    ln2 = cost_layernorm(cfg, params)
    ffn = cost_ffn(cfg, params)
    res = cost_residual(cfg, params)

    components = [ln1, attn, ln2, ffn, res, res]

    return FHECost(
        name="Transformer Block (total)",
        mults=sum(c.mults for c in components),
        adds=sum(c.adds for c in components),
        rotations=sum(c.rotations for c in components),
        key_switches=sum(c.key_switches for c in components),
        depth=sum(c.depth for c in components),
        rescales=sum(c.rescales for c in components),
        bootstraps=sum(c.bootstraps for c in components),
        notes=f"d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
              f"d_ff={cfg.d_ff}, seq={cfg.seq_len}",
    )


def print_separator():
    print("=" * 120)


def run_analysis():
    """Run the full analytical cost model."""
    params = CKKSParams()
    cfg = TransformerConfig()

    print_separator()
    print("ANALYTICAL FHE COST MODEL — TRANSFORMER OPERATIONS")
    print(f"CKKS Params: N={params.N}, L={params.L}, Δ={params.delta}-bit, "
          f"slots={params.n_slots}")
    print(f"Transformer: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"d_ff={cfg.d_ff}, seq_len={cfg.seq_len}")
    print_separator()

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
        cost_embedding(cfg, params),
        cost_residual(cfg, params),
    ]

    print("\n--- Individual Operation Costs ---\n")
    for op in ops:
        print(op)

    # Full block
    block = cost_transformer_block(cfg, params)
    print_separator()
    print("\n--- Full Transformer Block ---\n")
    print(block)

    # Full model — with cross-layer depth accumulation
    total_depth = block.depth * cfg.n_layers
    # Bootstrapping needed whenever accumulated depth exceeds L
    # Each bootstrap refreshes ~L levels but costs ~T_BOOT ms
    bootstraps_per_layer = max(1, math.ceil(block.depth / params.L))
    total_bootstraps = bootstraps_per_layer * cfg.n_layers
    bootstrap_overhead_s = total_bootstraps * 50 / 1000

    print(f"\n--- Full Model ({cfg.n_layers} layers) ---\n")
    print(f"  Total mults:          {block.mults * cfg.n_layers:>15,}")
    print(f"  Total adds:           {block.adds * cfg.n_layers:>15,}")
    print(f"  Total rotations:      {block.rotations * cfg.n_layers:>15,}")
    print(f"  Total depth:          {total_depth:>15,}")
    print(f"  Depth per block:      {block.depth:>15}")
    print(f"  Available levels (L): {params.L:>15}")
    print(f"  ⚠ Depth > L after:    {max(1, params.L // max(1, block.depth)):>15} block(s)")
    print(f"  Bootstraps/layer:     {bootstraps_per_layer:>15}")
    print(f"  Total bootstraps:     {total_bootstraps:>15}")
    print(f"  Bootstrap overhead:   {bootstrap_overhead_s:>15.1f} s")
    print(f"  Compute wall time:    {block.wall_ms * cfg.n_layers / 1000:>15,.1f} s")
    print(f"  Total wall time:      {(block.wall_ms * cfg.n_layers / 1000 + bootstrap_overhead_s):>15,.1f} s")

    # Bottleneck breakdown
    print_separator()
    print("\n--- Bottleneck Breakdown (% of block wall time) ---\n")

    components = {
        "LayerNorm (×2)": cost_layernorm(cfg, params),
        "Attention": cost_attention(cfg, params),
        "FFN": cost_ffn(cfg, params),
        "Residual (×2)": cost_residual(cfg, params),
    }
    # Double LayerNorm and Residual
    components["LayerNorm (×2)"].mults *= 2
    components["LayerNorm (×2)"].adds *= 2
    components["LayerNorm (×2)"].rotations *= 2
    components["Residual (×2)"].adds *= 2

    total_ms = sum(c.wall_ms for c in components.values())
    for name, c in sorted(components.items(), key=lambda x: -x[1].wall_ms):
        pct = c.wall_ms / total_ms * 100 if total_ms > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:25s} {pct:6.1f}%  {bar}  (~{c.wall_ms:,.0f} ms)")

    print_separator()

    # Depth budget analysis
    print("\n--- Depth Budget Analysis ---\n")
    print(f"  Available levels (L):     {params.L}")
    print(f"  Depth per block:          {block.depth}")
    print(f"  Blocks before bootstrap:  {max(1, params.L // max(1, block.depth))}")
    print(f"  Bootstraps per block:     {block.bootstraps}")
    print(f"  Total bootstraps (model): {block.bootstraps * cfg.n_layers}")
    print(f"  Bootstrap overhead:       ~{block.bootstraps * cfg.n_layers * 50 / 1000:.1f} s")

    print_separator()

    # Comparison: what if we remove softmax?
    print("\n--- Counterfactual: No Softmax (polynomial attention) ---\n")
    attn_no_sm = cost_attention(cfg, params)
    sm_cost = cost_softmax(cfg, params)
    sm_share = sm_cost.wall_ms * cfg.n_heads * cfg.seq_len
    attn_total = attn_no_sm.wall_ms
    if attn_total > 0:
        print(f"  Softmax share of attention: {sm_share / attn_total * 100:.1f}%")
    print(f"  Softmax wall time (per head per row): ~{sm_cost.wall_ms:,.0f} ms")
    print(f"  Total softmax in block: ~{sm_share / 1000:,.1f} s")

    print_separator()


if __name__ == "__main__":
    run_analysis()
