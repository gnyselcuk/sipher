"""
FHE-Native Attention Design — Cost Comparison
===============================================

Compares standard Transformer attention against three FHE-native alternatives.
All costs use empirical timings from OpenFHE benchmarks.

Candidates:
  1. Standard Attention:  softmax(Q·K^T/√d)·V     — baseline
  2. CipherAttention:     φ(Q)·(φ(K)^T·V)         — linear attention
  3. CipherMixer:         MLP-Mixer token mixing   — all ct-pt
  4. CipherRWKV:          Recurrent linear attention — sequential

Design principles:
  P1: Eliminate O(seq²) ct-ct multiplications
  P2: Maximize ct-pt / ct-ct ratio (ct-pt is 6× cheaper)
  P3: Minimize multiplicative depth
  P4: Maximize SIMD packing utilization
  P5: Minimize rotation count
"""

from dataclasses import dataclass
import math


@dataclass
class Timings:
    """Empirical timings from OpenFHE (N=32768)."""
    add: float = 3.37        # ms
    mult_ct: float = 50.96   # ms (ct × ct)
    mult_pt: float = 8.71    # ms (ct × pt)
    rot: float = 50.29       # ms
    boot: float = 5000.0     # ms (estimate)


@dataclass
class Cost:
    name: str
    mult_ct: int = 0
    mult_pt: int = 0
    adds: int = 0
    rots: int = 0
    depth: int = 0
    boots: int = 0
    notes: str = ""

    def wall_ms(self, t: Timings) -> float:
        return (self.mult_ct * t.mult_ct + self.mult_pt * t.mult_pt +
                self.adds * t.add + self.rots * t.rot + self.boots * t.boot)

    def wall_str(self, t: Timings) -> str:
        ms = self.wall_ms(t)
        if ms < 1000:
            return f"{ms:.0f} ms"
        elif ms < 60_000:
            return f"{ms/1000:.1f} s"
        elif ms < 3_600_000:
            return f"{ms/60_000:.1f} min"
        else:
            return f"{ms/3_600_000:.2f} hr"


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12
    seq_len: int = 512
    n_slots: int = 16384     # N/2 for N=32768
    poly_deg_phi: int = 2    # feature map degree for linear attention
    poly_deg_act: int = 4    # activation polynomial degree
    mixer_hidden: int = 1024 # hidden dim for MLP-Mixer token mixing


# ─────────────────────────────────────────────────────────────────────
# 1. STANDARD ATTENTION
# ─────────────────────────────────────────────────────────────────────

def standard_attention(cfg: Config, t: Timings) -> Cost:
    """
    softmax(Q·K^T / √d_k) · V

    Q, K, V all encrypted (derived from user input).
    Q·K^T: seq² ct-ct mults per head (the bottleneck).
    Softmax: polynomial exp + Newton division.
    A·V: seq×d_k ct-ct mults per head.
    """
    nh = cfg.n_heads
    seq = cfg.seq_len
    d_k = cfg.d_model // nh
    log_dk = math.ceil(math.log2(d_k))
    log_seq = math.ceil(math.log2(seq))

    # QKV projections: 3 × Linear(d→d), ct-pt (weights plaintext)
    bsgs = 2 * math.ceil(math.sqrt(cfg.d_model))
    qkv_pt = 3 * cfg.d_model
    qkv_rot = 3 * bsgs
    qkv_adds = 3 * (cfg.d_model - 1)

    # Q·K^T: seq² ct-ct mults per head + log(d_k) reductions
    qk_ct = seq * seq * nh
    qk_rot = seq * seq * log_dk * nh
    qk_adds = seq * seq * log_dk * nh

    # Softmax per row: poly exp (deg 8) + Newton division
    sm_deg = 8
    sm_ct_per_row = sm_deg + 8  # exp poly + Newton
    sm_rot_per_row = log_seq
    sm_ct = sm_ct_per_row * seq * nh
    sm_rot = sm_rot_per_row * seq * nh
    sm_adds = (sm_deg + 12) * seq * nh
    sm_depth = math.ceil(math.log2(sm_deg)) + 1 + 4  # exp + Newton

    # A·V: seq×d_k ct-ct mults per head + log(seq) reductions
    av_ct = seq * d_k * nh
    av_rot = seq * d_k * log_seq * nh
    av_adds = seq * d_k * log_seq * nh

    return Cost(
        name="Standard Attention",
        mult_ct=qk_ct + sm_ct + av_ct,
        mult_pt=qkv_pt,
        adds=qkv_adds + qk_adds + sm_adds + av_adds,
        rots=qkv_rot + qk_rot + sm_rot + av_rot,
        depth=1 + 1 + sm_depth + 1,
        notes=f"Q·K^T={qk_ct:,} ct-ct, A·V={av_ct:,} ct-ct",
    )


# ─────────────────────────────────────────────────────────────────────
# 2. CIPHER ATTENTION (Linear Attention)
# ─────────────────────────────────────────────────────────────────────

def cipher_attention(cfg: Config, t: Timings) -> Cost:
    """
    φ(Q) · (φ(K)^T · V)

    Linear attention: changes associativity to avoid seq×seq matrix.
    φ(x) = 1 + x + x²/2  (degree-2 polynomial feature map)

    Key insight: φ(K)^T · V is d_k × d_k (independent of seq!).
    Then φ(Q) · result is seq × d_k.

    Cost reduction: O(seq²) → O(seq·d_k + d_k²)
    """
    nh = cfg.n_heads
    seq = cfg.seq_len
    d_k = cfg.d_model // nh
    deg = cfg.poly_deg_phi
    log_dk = math.ceil(math.log2(d_k))
    log_seq = math.ceil(math.log2(seq))

    # QKV projections: same as standard (ct-pt)
    bsgs = 2 * math.ceil(math.sqrt(cfg.d_model))
    qkv_pt = 3 * cfg.d_model
    qkv_rot = 3 * bsgs
    qkv_adds = 3 * (cfg.d_model - 1)

    # Feature map φ(x) = 1 + x + x²/2
    # Per element: 1 ct-ct mult (x²) + 2 adds + 1 ct-pt mult (×0.5)
    # Applied to Q and K: 2 × seq × d_k elements per head
    phi_ct = 2 * seq * nh  # x² per row (SIMD handles d_k)
    phi_pt = 2 * seq * nh  # ×0.5 scaling
    phi_adds = 4 * seq * nh
    phi_depth = 2  # x² is depth 1, +1 for accumulation

    # φ(K)^T · V: (d_k × seq) · (seq × d_k) → d_k × d_k
    # d_k² output elements, each a dot product of length seq
    # Per element: 1 ct-ct mult + log(seq) rotations for reduction
    # BUT: with SIMD, d_k elements packed → d_k²/d_k = d_k ct-ct mults per "row"
    # Total: d_k ct-ct mults + d_k × log(seq) rotations per head
    kv_ct = d_k * nh
    kv_rot = d_k * log_seq * nh
    kv_adds = d_k * log_seq * nh
    kv_depth = 1 + log_seq  # mult + reduction tree

    # φ(Q) · (φ(K)^T·V): (seq × d_k) · (d_k × d_k) → seq × d_k
    # seq × d_k output elements, each dot product of length d_k
    # With SIMD: seq ct-ct mults + seq × log(d_k) rotations per head
    qkv_ct = seq * nh
    qkv_rot2 = seq * log_dk * nh
    qkv_adds2 = seq * log_dk * nh

    # Normalization: φ(Q) · φ(K)^T · 1 (row sums for denominator)
    # d_k ct-ct mults + log(d_k) rotations per row
    norm_ct = seq * nh  # per row
    norm_rot = seq * log_dk * nh
    norm_adds = seq * (log_dk + 1) * nh
    # Division: Newton-Raphson, ~4 iterations
    div_ct = 4 * seq * nh
    div_depth = 4

    total_depth = phi_depth + max(kv_depth, 1) + 1 + div_depth

    return Cost(
        name="CipherAttention (linear)",
        mult_ct=phi_ct + kv_ct + qkv_ct + norm_ct + div_ct,
        mult_pt=qkv_pt + phi_pt,
        adds=qkv_adds + phi_adds + kv_adds + qkv_adds2 + norm_adds,
        rots=qkv_rot + kv_rot + qkv_rot2 + norm_rot,
        depth=total_depth,
        notes=f"φ=deg{deg}, K^T·V={kv_ct} ct-ct (d_k²={d_k*d_k}), "
              f"Q·KV={qkv_ct} ct-ct",
    )


# ─────────────────────────────────────────────────────────────────────
# 3. CIPHER MIXER (MLP-Mixer style)
# ─────────────────────────────────────────────────────────────────────

def cipher_mixer(cfg: Config, t: Timings) -> Cost:
    """
    MLP-Mixer token mixing:
      x → LayerNorm → Transpose → Linear(seq→H) → Act → Linear(H→seq) → Transpose → + residual

    Key FHE advantage: ALL weight matrices are PLAINTEXT.
    Token mixing uses ct-pt multiplications exclusively.
    Only the activation function requires ct-ct mults.

    No data-dependent mixing (unlike attention) — fixed learned weights.
    """
    seq = cfg.seq_len
    d = cfg.d_model
    H = cfg.mixer_hidden
    n = cfg.n_slots
    act_deg = cfg.poly_deg_act

    # Number of ciphertexts to hold the sequence dimension
    n_ct_seq = math.ceil(seq / n)
    n_ct_d = math.ceil(d / n)

    # Linear(seq → H): diagonal encoding
    # d_in=seq, d_out=H, all ct-pt
    bsgs_seq = 2 * math.ceil(math.sqrt(seq))
    up_pt = seq * n_ct_d  # seq diagonals × d_model groups
    up_rot = bsgs_seq * n_ct_d
    up_adds = (seq - 1) * n_ct_d

    # Activation: polynomial (ct-ct, element-wise)
    # Applied to H values across d_model channels
    n_ct_H = math.ceil(H / n)
    act_ct = act_deg * n_ct_H * n_ct_d
    act_adds = act_deg * n_ct_H * n_ct_d
    act_depth = math.ceil(math.log2(act_deg)) + 1

    # Linear(H → seq): diagonal encoding, ct-pt
    bsgs_H = 2 * math.ceil(math.sqrt(H))
    down_pt = H * n_ct_d
    down_rot = bsgs_H * n_ct_d
    down_adds = (H - 1) * n_ct_d

    # LayerNorm (before mixing)
    log_d = math.ceil(math.log2(d))
    ln_ct = 6 * n_ct_d  # variance + invsqrt poly
    ln_pt = n_ct_d
    ln_rot = 2 * log_d * n_ct_d
    ln_adds = (2 * log_d + 3) * n_ct_d
    ln_depth = 7

    total_depth = ln_depth + 1 + act_depth + 1

    return Cost(
        name="CipherMixer (MLP-Mixer)",
        mult_ct=act_ct + ln_ct,
        mult_pt=up_pt + down_pt + ln_pt,
        adds=up_adds + act_adds + down_adds + ln_adds,
        rots=up_rot + down_rot + ln_rot,
        depth=total_depth,
        notes=f"seq→H={H}→seq, ALL mixing is ct-pt! "
              f"ct-ct only for activation ({act_ct})",
    )


# ─────────────────────────────────────────────────────────────────────
# 4. CIPHER RWKV (Recurrent Linear Attention)
# ─────────────────────────────────────────────────────────────────────

def cipher_rwkv(cfg: Config, t: Timings) -> Cost:
    """
    RWKV-style recurrent attention:
      For each token t:
        h_t = λ · h_{t-1} + k_t^T · v_t     (state update)
        o_t = q_t · h_t                       (output)

    h is a d_k × d_k state matrix (encrypted).
    λ (decay) is a plaintext scalar.

    Per-token cost:
      - λ · h_{t-1}: 1 ct-pt mult (scalar × ciphertext)
      - k_t^T · v_t: outer product, d_k² ct-ct mults... but with SIMD:
        k_t and v_t are vectors of length d_k, packed in 1 ciphertext
        outer product via rotations: d_k ct-ct mults + d_k rotations
      - h_t = sum: 1 add
      - o_t = q_t · h_t: matrix-vector product, d_k ct-ct mults + log(d_k) rots

    Total per token per head: ~2×d_k ct-ct + 1 ct-pt + rotations
    Total for sequence: seq × per-token cost

    Advantage: O(seq × d_k) total, no seq² term.
    Disadvantage: sequential (can't parallelize across tokens).
    """
    nh = cfg.n_heads
    seq = cfg.seq_len
    d_k = cfg.d_model // nh
    log_dk = math.ceil(math.log2(d_k))

    # QKV projections: ct-pt (same as others)
    bsgs = 2 * math.ceil(math.sqrt(cfg.d_model))
    qkv_pt = 3 * cfg.d_model
    qkv_rot = 3 * bsgs
    qkv_adds = 3 * (cfg.d_model - 1)

    # Per token per head:
    # State update: h_t = λ·h_{t-1} + k_t ⊗ v_t
    #   λ·h: 1 ct-pt mult (state is packed in ceil(d_k²/n_slots) cts)
    n_ct_state = math.ceil(d_k * d_k / cfg.n_slots)
    decay_pt = n_ct_state  # per token per head

    #   k_t ⊗ v_t: outer product of two d_k vectors
    #   With SIMD: d_k ct-ct mults (each slot computes one element)
    #   + rotations to align
    outer_ct = d_k  # per token per head (SIMD packs d_k elements)
    outer_rot = d_k  # rotations to broadcast k across slots

    # Output: o_t = q_t · h_t
    #   Matrix-vector product: d_k ct-ct mults + log(d_k) reduction
    out_ct = d_k  # per token per head
    out_rot = log_dk
    out_adds = log_dk

    # Total across sequence and heads
    total_decay_pt = decay_pt * seq * nh
    total_outer_ct = outer_ct * seq * nh
    total_outer_rot = outer_rot * seq * nh
    total_out_ct = out_ct * seq * nh
    total_out_rot = out_rot * seq * nh
    total_out_adds = out_adds * seq * nh
    total_adds = seq * nh  # h_t accumulation

    # Depth: sequential! Each token adds depth for:
    #   decay (1) + outer product (1) + output (1 + log_dk)
    # But state is carried forward → depth accumulates across tokens
    # Need bootstrapping every L/depth_per_token tokens
    depth_per_token = 3  # decay mult + outer mult + output mult
    tokens_per_bootstrap = max(1, 40 // depth_per_token)  # L=40
    total_boots = math.ceil(seq / tokens_per_bootstrap) * nh

    return Cost(
        name="CipherRWKV (recurrent)",
        mult_ct=total_outer_ct + total_out_ct,
        mult_pt=qkv_pt + total_decay_pt,
        adds=qkv_adds + total_out_adds + total_adds,
        rots=qkv_rot + total_outer_rot + total_out_rot,
        depth=depth_per_token,  # per token (resets after bootstrap)
        boots=total_boots,
        notes=f"per-token: {outer_ct+out_ct} ct-ct + {decay_pt} ct-pt, "
              f"boots={total_boots}, sequential!",
    )


# ─────────────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────────────

def compare_all():
    t = Timings()
    cfg = Config()

    print("=" * 110)
    print("FHE-NATIVE ATTENTION DESIGN — COST COMPARISON")
    print(f"Config: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"seq_len={cfg.seq_len}, slots={cfg.n_slots}")
    print(f"Timings: ct-ct={t.mult_ct}ms, ct-pt={t.mult_pt}ms, "
          f"rot={t.rot}ms, add={t.add}ms")
    print("=" * 110)

    candidates = [
        standard_attention(cfg, t),
        cipher_attention(cfg, t),
        cipher_mixer(cfg, t),
        cipher_rwkv(cfg, t),
    ]

    # Detailed comparison
    print(f"\n{'Metric':<25}", end="")
    for c in candidates:
        short = c.name.split("(")[0].strip()[:18]
        print(f" {short:>18}", end="")
    print()
    print("─" * 110)

    rows = [
        ("ct-ct mults", lambda c: f"{c.mult_ct:,}"),
        ("ct-pt mults", lambda c: f"{c.mult_pt:,}"),
        ("additions", lambda c: f"{c.adds:,}"),
        ("rotations", lambda c: f"{c.rots:,}"),
        ("depth", lambda c: f"{c.depth}"),
        ("bootstraps", lambda c: f"{c.boots}"),
        ("wall time", lambda c: c.wall_str(t)),
    ]

    for label, fn in rows:
        print(f"  {label:<23}", end="")
        for c in candidates:
            print(f" {fn(c):>18}", end="")
        print()

    # Speedup vs standard
    base_ms = candidates[0].wall_ms(t)
    print(f"\n  {'Speedup vs Standard':<23}", end="")
    for c in candidates:
        ratio = base_ms / c.wall_ms(t) if c.wall_ms(t) > 0 else float('inf')
        print(f" {ratio:>17.1f}×", end="")
    print()

    # Cost composition
    print(f"\n{'=' * 110}")
    print("COST COMPOSITION (what dominates?)")
    print("=" * 110)

    for c in candidates:
        ms = c.wall_ms(t)
        ct_ms = c.mult_ct * t.mult_ct
        pt_ms = c.mult_pt * t.mult_pt
        rot_ms = c.rots * t.rot
        add_ms = c.adds * t.add
        boot_ms = c.boots * t.boot

        print(f"\n  {c.name}")
        print(f"    Total: {c.wall_str(t)}")
        parts = [
            ("ct-ct mults", ct_ms),
            ("ct-pt mults", pt_ms),
            ("rotations", rot_ms),
            ("additions", add_ms),
            ("bootstraps", boot_ms),
        ]
        for name, part_ms in sorted(parts, key=lambda x: -x[1]):
            if part_ms > 0:
                pct = part_ms / ms * 100
                bar = "█" * max(1, int(pct / 3))
                print(f"      {name:<18} {pct:6.1f}%  {bar}")

    # Scaling with sequence length
    print(f"\n{'=' * 110}")
    print("SCALING WITH SEQUENCE LENGTH")
    print("=" * 110)
    print(f"\n  {'seq_len':>8}", end="")
    for c_fn in [standard_attention, cipher_attention, cipher_mixer, cipher_rwkv]:
        name = c_fn(Config(), t).name.split("(")[0].strip()[:15]
        print(f" {name:>16}", end="")
    print()
    print("  " + "─" * 80)

    for seq in [128, 256, 512, 1024, 2048, 4096]:
        cfg2 = Config(seq_len=seq)
        print(f"  {seq:>8}", end="")
        for c_fn in [standard_attention, cipher_attention, cipher_mixer, cipher_rwkv]:
            c = c_fn(cfg2, t)
            print(f" {c.wall_str(t):>16}", end="")
        print()

    # Scaling with n_heads
    print(f"\n{'=' * 110}")
    print("SCALING WITH NUMBER OF HEADS (d_model=768 fixed)")
    print("=" * 110)
    print(f"\n  {'n_heads':>8}", end="")
    for c_fn in [standard_attention, cipher_attention, cipher_mixer, cipher_rwkv]:
        name = c_fn(Config(), t).name.split("(")[0].strip()[:15]
        print(f" {name:>16}", end="")
    print()
    print("  " + "─" * 80)

    for nh in [1, 2, 4, 8, 12]:
        if 768 % nh != 0:
            continue
        cfg2 = Config(n_heads=nh)
        print(f"  {nh:>8}", end="")
        for c_fn in [standard_attention, cipher_attention, cipher_mixer, cipher_rwkv]:
            c = c_fn(cfg2, t)
            print(f" {c.wall_str(t):>16}", end="")
        print()

    # Design recommendation
    print(f"\n{'=' * 110}")
    print("DESIGN RECOMMENDATION")
    print("=" * 110)
    print("""
  For CipherFormer v0.1, we recommend a HYBRID approach:

  ┌─────────────────────────────────────────────────────────────────┐
  │  CipherMixer (primary) + CipherAttention (selective)           │
  │                                                                 │
  │  • Most layers: CipherMixer (all ct-pt, maximum efficiency)    │
  │  • Every Nth layer: CipherAttention (data-dependent mixing)    │
  │  • Ratio: 3 Mixer : 1 Attention (tunable)                     │
  │                                                                 │
  │  Rationale:                                                     │
  │  • Mixer gives 100%+ speedup (all ct-pt mixing)                │
  │  • Occasional attention preserves data-dependent expressiveness │
  │  • Linear attention (not standard) when attention is needed     │
  │  • RWKV for streaming/long-sequence use cases                   │
  └─────────────────────────────────────────────────────────────────┘

  Key design parameters for CipherFormer:
    n_heads:     1-2 (not 12 — each head adds independent cost)
    d_k:         384-768 (large, fills SIMD slots better)
    seq_len:     ≤512 (quadratic cost still present in linear attn)
    φ degree:    2 (minimal depth for feature map)
    Mixer H:     2×seq_len (token mixing hidden dim)
    Mixer:Attn:  3:1 ratio (tunable per task)
""")


if __name__ == "__main__":
    compare_all()
