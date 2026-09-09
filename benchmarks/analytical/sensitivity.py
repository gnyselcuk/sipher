"""
Sensitivity Analysis — FHE Cost Model
======================================

Varies key parameters to identify which design choices
most impact FHE cost. Informs CipherFormer architecture decisions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cost_model import (
    CKKSParams, TransformerConfig, FHECost,
    cost_transformer_block, cost_attention, cost_ffn,
    cost_softmax, cost_linear, cost_layernorm,
)
import math


def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    elif ms < 60_000:
        return f"{ms/1000:.1f} s"
    elif ms < 3_600_000:
        return f"{ms/60_000:.1f} min"
    else:
        return f"{ms/3_600_000:.1f} hr"


def sweep_d_model():
    """How does d_model affect cost and packing utilization?"""
    print("\n" + "=" * 100)
    print("SWEEP 1: d_model (Transformer hidden dimension)")
    print("=" * 100)
    print(f"{'d_model':>10} {'pack%':>7} {'block_ms':>14} {'attn_ms':>14} "
          f"{'ffn_ms':>14} {'depth':>6} {'rots':>12}")
    print("-" * 100)

    params = CKKSParams()
    for d in [128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768]:
        cfg = TransformerConfig(d_model=d, d_ff=4*d, n_heads=max(1, d//64))
        block = cost_transformer_block(cfg, params)
        attn = cost_attention(cfg, params)
        ffn = cost_ffn(cfg, params)
        pack = d / params.n_slots * 100
        print(f"{d:>10,} {pack:>6.1f}% {block.wall_ms:>14,.0f} {attn.wall_ms:>14,.0f} "
              f"{ffn.wall_ms:>14,.0f} {block.depth:>6} {block.rotations:>12,}")


def sweep_seq_len():
    """How does sequence length affect cost?"""
    print("\n" + "=" * 100)
    print("SWEEP 2: seq_len (sequence length)")
    print("=" * 100)
    print(f"{'seq_len':>10} {'block_ms':>14} {'attn_ms':>14} "
          f"{'softmax_ms':>14} {'depth':>6} {'boot':>5}")
    print("-" * 100)

    params = CKKSParams()
    cfg_base = TransformerConfig()
    for seq in [64, 128, 256, 512, 1024, 2048, 4096]:
        cfg = TransformerConfig(seq_len=seq)
        block = cost_transformer_block(cfg, params)
        attn = cost_attention(cfg, params)
        sm = cost_softmax(cfg, params, seq)
        sm_total = sm.wall_ms * cfg.n_heads * seq
        print(f"{seq:>10,} {block.wall_ms:>14,.0f} {attn.wall_ms:>14,.0f} "
              f"{sm_total:>14,.0f} {block.depth:>6} {block.bootstraps:>5}")


def sweep_poly_degree():
    """How does polynomial approximation degree affect cost and depth?"""
    print("\n" + "=" * 100)
    print("SWEEP 3: Polynomial degree (softmax approximation)")
    print("=" * 100)
    print(f"{'degree':>8} {'sm_ms':>12} {'sm_depth':>9} {'block_ms':>14} "
          f"{'block_depth':>12} {'accuracy_note':>20}")
    print("-" * 100)

    params = CKKSParams()
    for deg in [2, 4, 6, 8, 10, 12, 16, 20, 32]:
        cfg = TransformerConfig(poly_degree_softmax=deg)
        sm = cost_softmax(cfg, params)
        block = cost_transformer_block(cfg, params)
        # Rough accuracy note
        if deg <= 4:
            note = "poor (>5% err)"
        elif deg <= 8:
            note = "fair (~1-2% err)"
        elif deg <= 12:
            note = "good (~0.5% err)"
        else:
            note = "excellent (<0.1%)"
        print(f"{deg:>8} {sm.wall_ms:>12,.1f} {sm.depth:>9} {block.wall_ms:>14,.0f} "
              f"{block.depth:>12} {note:>20}")


def sweep_ring_dimension():
    """How does CKKS ring dimension N affect cost?"""
    print("\n" + "=" * 100)
    print("SWEEP 4: Ring dimension N (CKKS security parameter)")
    print("=" * 100)
    print(f"{'N':>10} {'slots':>8} {'pack%':>7} {'block_ms':>14} "
          f"{'mult_ms':>10} {'rot_ms':>10} {'security':>10}")
    print("-" * 100)

    cfg = TransformerConfig()
    # Reference timings scale roughly as N*log(N)
    for log_n in [14, 15, 16, 17]:
        n = 2 ** log_n
        slots = n // 2
        # Rough timing model: mult ~ N*log(N)/2^20 ms, rot ~ 5× mult
        t_mult = n * log_n / (2**20) * 0.1 * 10  # scaled reference
        t_rot = t_mult * 5
        params = CKKSParams(N=n, n_slots=slots)
        params.T_MULT = t_mult
        params.T_ROT = t_rot
        block = cost_transformer_block(cfg, params)
        pack = cfg.d_model / slots * 100
        # Security: N=2^14 ~ 128-bit, N=2^15 ~ 192-bit, N=2^16 ~ 256-bit (rough)
        sec = {14: "~128-bit", 15: "~192-bit", 16: "~256-bit", 17: ">256-bit"}
        print(f"{n:>10,} {slots:>8,} {pack:>6.1f}% {block.wall_ms:>14,.0f} "
              f"{t_mult:>10.3f} {t_rot:>10.3f} {sec.get(log_n, '?'):>10}")


def sweep_n_heads():
    """How does number of attention heads affect cost?"""
    print("\n" + "=" * 100)
    print("SWEEP 5: Number of attention heads (d_model=768 fixed)")
    print("=" * 100)
    print(f"{'n_heads':>8} {'d_k':>6} {'block_ms':>14} {'attn_ms':>14} "
          f"{'depth':>6} {'rots':>12}")
    print("-" * 100)

    params = CKKSParams()
    for nh in [1, 2, 4, 6, 8, 12, 16, 24]:
        if 768 % nh != 0:
            continue
        cfg = TransformerConfig(n_heads=nh)
        block = cost_transformer_block(cfg, params)
        attn = cost_attention(cfg, params)
        d_k = 768 // nh
        print(f"{nh:>8} {d_k:>6} {block.wall_ms:>14,.0f} {attn.wall_ms:>14,.0f} "
              f"{block.depth:>6} {block.rotations:>12,}")


def packing_analysis():
    """Detailed packing utilization analysis."""
    print("\n" + "=" * 100)
    print("PACKING UTILIZATION ANALYSIS")
    print("=" * 100)

    params = CKKSParams()
    cfg = TransformerConfig()

    print(f"\nCKKS slots: {params.n_slots:,}")
    print(f"d_model: {cfg.d_model} → {cfg.d_model/params.n_slots*100:.1f}% utilization")
    print(f"seq_len: {cfg.seq_len} → {cfg.seq_len/params.n_slots*100:.1f}% utilization")
    print(f"d_ff: {cfg.d_ff} → {cfg.d_ff/params.n_slots*100:.1f}% utilization")

    print("\n--- What if we pack multiple items per ciphertext? ---\n")

    # Pack multiple tokens
    tokens_per_ct = params.n_slots // cfg.d_model
    print(f"Tokens per ciphertext (d_model packing): {tokens_per_ct}")
    print(f"  → {cfg.seq_len} tokens need {math.ceil(cfg.seq_len / tokens_per_ct)} ciphertexts")

    # Pack multiple heads
    heads_per_ct = params.n_slots // (cfg.seq_len * (cfg.d_model // cfg.n_heads))
    print(f"\nHeads per ciphertext (seq×d_k packing): {max(1, heads_per_ct)}")

    # Ideal d_model for 100% utilization
    print(f"\nIdeal d_model for 100% packing: {params.n_slots:,}")
    print(f"Ideal d_model for 50% packing: {params.n_slots // 2:,}")
    print(f"Ideal d_model for 25% packing: {params.n_slots // 4:,}")

    # Cost at ideal packing
    print("\n--- Cost at different packing levels ---\n")
    print(f"{'d_model':>10} {'pack%':>7} {'block_ms':>14} {'vs_768':>10}")
    print("-" * 50)

    base_cfg = TransformerConfig(d_model=768, d_ff=3072)
    base_block = cost_transformer_block(base_cfg, params)
    base_ms = base_block.wall_ms

    for d in [768, 4096, 8192, 16384, 32768]:
        cfg2 = TransformerConfig(d_model=d, d_ff=4*d, n_heads=max(1, d//64))
        block2 = cost_transformer_block(cfg2, params)
        ratio = block2.wall_ms / base_ms if base_ms > 0 else 0
        pack = d / params.n_slots * 100
        print(f"{d:>10,} {pack:>6.1f}% {block2.wall_ms:>14,.0f} {ratio:>9.1f}×")


def multi_token_packing_analysis():
    """
    The key insight: CKKS SIMD slots can hold MULTIPLE tokens.
    With d_model=768 and n_slots=32768, we can pack 42 tokens per ciphertext.
    This changes the effective per-token cost dramatically.
    """
    print("\n" + "=" * 100)
    print("MULTI-TOKEN PACKING ANALYSIS (The FHE-Native Advantage)")
    print("=" * 100)

    params = CKKSParams()
    cfg = TransformerConfig()

    tokens_per_ct = params.n_slots // cfg.d_model  # 42
    effective_seq = math.ceil(cfg.seq_len / tokens_per_ct)  # ciphertexts needed

    print(f"\n  d_model:          {cfg.d_model}")
    print(f"  n_slots:          {params.n_slots:,}")
    print(f"  tokens/ct:        {tokens_per_ct}")
    print(f"  seq_len:          {cfg.seq_len}")
    print(f"  ciphertexts/seq:  {effective_seq}")

    # Cost with packing: operations on packed ciphertexts
    # A packed linear layer processes tokens_per_ct tokens simultaneously
    # So the cost is divided by tokens_per_ct (per-token effective cost)
    block = cost_transformer_block(cfg, params)

    print(f"\n  --- Per-token effective cost (with {tokens_per_ct}× packing) ---\n")
    print(f"  {'Metric':<25} {'Unpacked':>15} {'Packed':>15} {'Speedup':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Wall time (ms)':<25} {block.wall_ms:>15,.0f} {block.wall_ms/tokens_per_ct:>15,.0f} {tokens_per_ct:>9}×")
    print(f"  {'Mults':<25} {block.mults:>15,} {block.mults//tokens_per_ct:>15,} {tokens_per_ct:>9}×")
    print(f"  {'Rotations':<25} {block.rotations:>15,} {block.rotations//tokens_per_ct:>15,} {tokens_per_ct:>9}×")

    # Full model with packing
    full_ms = block.wall_ms * cfg.n_layers
    packed_ms = full_ms / tokens_per_ct
    print(f"\n  Full model (12 layers):")
    print(f"    Unpacked: {fmt_ms(full_ms)}")
    print(f"    Packed:   {fmt_ms(packed_ms)}")
    print(f"    Plaintext GPU reference: ~120 ms")
    print(f"    FHE/Plaintext ratio (packed): ~{packed_ms/120:,.0f}×")

    # What if we also use linear attention?
    print(f"\n  --- Combined optimizations ---\n")
    # Linear attention: O(seq·d) instead of O(seq²·d)
    # Rough estimate: attention cost reduced by seq/d_k factor
    attn = cost_attention(cfg, params)
    ffn = cost_ffn(cfg, params)
    linear_attn_reduction = cfg.seq_len / (cfg.d_model // cfg.n_heads)  # ~8×
    linear_attn_ms = attn.wall_ms / linear_attn_reduction
    combined_block_ms = linear_attn_ms + ffn.wall_ms
    combined_packed_ms = combined_block_ms * cfg.n_layers / tokens_per_ct

    print(f"  {'Optimization':<35} {'Per-token ms':>15} {'vs baseline':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Baseline (unpacked)':<35} {block.wall_ms:>15,.0f} {'1×':>12}")
    print(f"  {'+ Multi-token packing (42×)':<35} {block.wall_ms/tokens_per_ct:>15,.0f} {f'{tokens_per_ct}×':>12}")
    print(f"  {'+ Linear attention':<35} {combined_block_ms/tokens_per_ct:>15,.0f} {f'{block.wall_ms/combined_block_ms*tokens_per_ct:.0f}×':>12}")
    print(f"  {'+ Depth-opt (low-deg poly)':<35} {combined_block_ms/tokens_per_ct*0.85:>15,.0f} {f'{block.wall_ms/(combined_block_ms*0.85)*tokens_per_ct:.0f}×':>12}")

    gpu_ref = 120  # ms for BERT-base on GPU
    final_ms = combined_block_ms * cfg.n_layers / tokens_per_ct * 0.85
    print(f"\n  Final FHE/Plaintext ratio: ~{final_ms/gpu_ref:,.0f}×")
    print(f"  (Down from ~{full_ms/gpu_ref:,.0f}× without any optimization)")


def main():
    print("FHE COST MODEL — SENSITIVITY ANALYSIS")
    print("Baseline: BERT-base, CKKS N=2^16, L=40")

    sweep_d_model()
    sweep_seq_len()
    sweep_poly_degree()
    sweep_ring_dimension()
    sweep_n_heads()
    packing_analysis()
    multi_token_packing_analysis()

    print("\n" + "=" * 100)
    print("KEY TAKEAWAYS")
    print("=" * 100)
    print("""
1. d_model is the #1 lever for packing utilization.
   Going from 768 → 32,768 eliminates the 50× packing waste.
   But this fundamentally changes the model architecture.

2. seq_len has quadratic impact on attention cost (Q·K^T is O(seq²)).
   Shorter sequences or linear attention are critical.

3. Polynomial degree has modest impact on wall time but major impact
   on depth. Lower degree = less depth = fewer bootstraps.

4. Ring dimension N is a security/cost tradeoff.
   Larger N = more slots = better packing, but slower per-operation.

5. Fewer heads with larger d_k reduces rotation overhead
   (fewer independent attention computations to manage).
""")


if __name__ == "__main__":
    main()
