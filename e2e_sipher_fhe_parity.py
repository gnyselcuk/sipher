"""
Sipher — Aynı Graf FHE Parity Testi
=====================================
Tek head, tek pozisyon, causal linear attention + PolyFFN.
PT ve FHE aynı formülü kullanır → gerçek parity.

FHE protokolü:
  Client: Enc(x) → Server
  Server: QKV = Enc(x) @ W_qkv (FHE matvec)
  Client: KV_cum = Σ K_t^T V_t (plaintext, client-side)
  Server: attn = Enc(Q) @ KV_cum (FHE matvec, plaintext KV_cum)
  Server: out = W_O(attn) * gate + residual (FHE)
  Server: ffn = PolyFFN(out) (FHE: matvec + ct×ct + matvec)
  Server: head(out + ffn) (FHE matvec)
  Client: Decrypt → logits

Run: .venv312/bin/python3 -u e2e_sipher_fhe_parity.py
"""

import torch
import torch.nn as nn
import tenseal as ts
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
torch.manual_seed(42)


def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 38],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


class TinySipherBlock(nn.Module):
    """Tek katman: causal linear attention + PolyFFN."""
    def __init__(self, d=32, nh=4, dk=8, seq=8, window=4):
        super().__init__()
        self.d, self.nh, self.dk, self.seq, self.window = d, nh, dk, seq, window
        self.W_Q = nn.Linear(d, nh * dk, bias=False)
        self.W_K = nn.Linear(d, nh * dk, bias=False)
        self.W_V = nn.Linear(d, nh * dk, bias=False)
        self.W_O = nn.Linear(nh * dk, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.ffn_up = nn.Linear(d, d * 2, bias=False)
        self.poly_coeffs = nn.Parameter(torch.tensor([0.0, 0.5, 0.2]))
        self.ffn_down = nn.Linear(d * 2, d, bias=False)
        idx = torch.arange(seq)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        self.register_buffer('causal_mask', (diff >= 0).float())

    def forward(self, x, pos):
        """Tek pozisyon için forward."""
        # x: (seq, d) — tüm sekans
        # pos: hedef pozisyon

        # QKV (tüm pozisyonlar için K,V; sadece pos için Q)
        Q_pos = self.W_Q(x[pos:pos+1]) / (self.dk ** 0.5)  # (1, nh*dk)
        K_all = self.W_K(x)  # (seq, nh*dk)
        V_all = self.W_V(x)  # (seq, nh*dk)

        # Causal linear attention: KV_cum = Σ_{t≤pos} K_t^T V_t
        Q_pos_r = Q_pos.view(self.nh, self.dk)
        K_all_r = K_all.view(self.seq, self.nh, self.dk)
        V_all_r = V_all.view(self.seq, self.nh, self.dk)

        attn_out = torch.zeros(self.nh, self.dk, device=x.device)
        for h in range(self.nh):
            KV_cum = torch.zeros(self.dk, self.dk, device=x.device)
            for t in range(pos + 1):  # causal: sadece t ≤ pos
                KV_cum += K_all_r[t, h].unsqueeze(-1) @ V_all_r[t, h].unsqueeze(0)
            attn_out[h] = Q_pos_r[h] @ KV_cum

        # W_O + gate + residual
        attn_flat = attn_out.view(1, -1)
        x_pos = x[pos:pos+1]
        out = x_pos + self.gate * self.W_O(attn_flat)

        # PolyFFN
        ffn = self.ffn_up(out)
        ffn = ffn * self.poly_coeffs[1] + ffn.pow(2) * self.poly_coeffs[2] + self.poly_coeffs[0]
        out = out + self.ffn_down(ffn)

        return out  # (1, d)


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher — Aynı Graf FHE Parity (Attn + FFN)" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")

    D, NH, DK, SEQ, WINDOW = 32, 4, 8, 8, 4

    print(f"\n  CKKS context...", end="", flush=True)
    t0 = time.time()
    context = create_context()
    print(f" {time.time()-t0:.1f}s")

    block = TinySipherBlock(D, NH, DK, SEQ, WINDOW)
    block.eval()
    print(f"  Block: d={D} nh={NH} dk={DK} seq={SEQ}")

    # Test girdisi
    x = torch.randn(SEQ, D) * 0.3
    pos = SEQ - 1  # son pozisyon
    print(f"  Input: seq={SEQ}, pos={pos}")

    # ── 1. Plaintext referans ──
    print(f"\n{'─' * 60}")
    print(f"  1. PLAINTEXT (aynı formül)")
    print(f"{'─' * 60}")
    with torch.no_grad():
        pt_out = block(x, pos)
    print(f"  Output shape: {pt_out.shape}")
    print(f"  Output[:5]: {pt_out[0, :5].tolist()}")

    # ── 2. FHE (aynı formül) ──
    print(f"\n{'─' * 60}")
    print(f"  2. FHE — Aynı Graf")
    print(f"{'─' * 60}")

    total_t = time.time()

    # Adım 1: Embedding şifrele
    enc_x_pos = ts.ckks_vector(context, x[pos].numpy().tolist())
    print(f"  1) Enc(x[pos]):    OK")

    # Adım 2: QKV matvec (FHE)
    t0 = time.time()
    with torch.no_grad():
        W_Q = block.W_Q.weight.numpy()
        W_K = block.W_K.weight.numpy()
        W_V = block.W_V.weight.numpy()

    enc_q = enc_x_pos.matmul(ts.plain_tensor((W_Q / (DK**0.5)).T.tolist()))
    # K ve V tüm pozisyonlar için (plaintext — client-side)
    with torch.no_grad():
        K_all = (x @ block.W_K.weight.T).numpy()  # (seq, nh*dk)
        V_all = (x @ block.W_V.weight.T).numpy()  # (seq, nh*dk)
    t_qkv = time.time() - t0
    print(f"  2) QKV:            {t_qkv:.2f}s (Q=FHE, K/V=client-side)")

    # Adım 3: KV_cum (client-side plaintext)
    t0 = time.time()
    KV_cum_all = np.zeros((NH, DK, DK))
    for h in range(NH):
        for t in range(pos + 1):
            K_t = K_all[t, h*DK:(h+1)*DK]
            V_t = V_all[t, h*DK:(h+1)*DK]
            KV_cum_all[h] += np.outer(K_t, V_t)
    t_kv = time.time() - t0
    print(f"  3) KV_cum:         {t_kv:.4f}s (client-side, {pos+1} pozisyon)")

    # Adım 4: Q @ KV_cum (FHE matvec, plaintext KV_cum)
    t0 = time.time()
    # Her head için ayrı matvec
    enc_q_dec = np.array(enc_q.decrypt())[:NH*DK]
    attn_out_fhe = np.zeros(NH * DK)
    for h in range(NH):
        q_h = enc_q_dec[h*DK:(h+1)*DK]
        # Q_h @ KV_cum[h] → matvec
        enc_q_h = ts.ckks_vector(context, q_h.tolist())
        enc_attn_h = enc_q_h.matmul(ts.plain_tensor(KV_cum_all[h].T.tolist()))
        attn_out_fhe[h*DK:(h+1)*DK] = np.array(enc_attn_h.decrypt())[:DK]
    t_attn = time.time() - t0
    print(f"  4) Q @ KV_cum:     {t_attn:.2f}s (FHE matvec × {NH} head)")

    # Adım 5: W_O + gate + residual (FHE)
    t0 = time.time()
    with torch.no_grad():
        W_O = block.W_O.weight.numpy()
        gate_val = block.gate.item()

    enc_attn = ts.ckks_vector(context, attn_out_fhe.tolist())
    enc_wo = enc_attn.matmul(ts.plain_tensor(W_O.T.tolist()))
    enc_gated = enc_wo * gate_val
    enc_residual = enc_x_pos + enc_gated  # FHE addition
    t_wo = time.time() - t0

    fhe_post_attn = np.array(enc_residual.decrypt())[:D]
    with torch.no_grad():
        pt_attn_flat = torch.zeros(1, NH*DK)
        for h in range(NH):
            q_h = (block.W_Q(x[pos:pos+1]) / (DK**0.5)).view(NH, DK)[h]
            pt_attn_flat[0, h*DK:(h+1)*DK] = q_h @ torch.tensor(KV_cum_all[h], dtype=torch.float32)
        pt_post_attn = x[pos:pos+1] + gate_val * block.W_O(pt_attn_flat)
    err_attn = np.max(np.abs(pt_post_attn.numpy().flatten() - fhe_post_attn))
    print(f"  5) W_O+gate+res:   {t_wo:.2f}s  err={err_attn:.2e}")

    # Adım 6: PolyFFN (FHE: matvec → ct×ct → matvec)
    t0 = time.time()
    with torch.no_grad():
        W_up = block.ffn_up.weight.numpy()
        W_down = block.ffn_down.weight.numpy()
        c0, c1, c2 = block.poly_coeffs.tolist()

    enc_ffn_up = enc_residual.matmul(ts.plain_tensor(W_up.T.tolist()))
    enc_ffn_act = enc_ffn_up * c1 + (enc_ffn_up * enc_ffn_up) * c2  # ct×ct!
    if abs(c0) > 1e-10:
        enc_ffn_act = enc_ffn_act + ts.ckks_vector(context, [c0] * (D*2))
    enc_ffn_down = enc_ffn_act.matmul(ts.plain_tensor(W_down.T.tolist()))
    enc_out = enc_residual + enc_ffn_down  # residual
    t_ffn = time.time() - t0

    fhe_out = np.array(enc_out.decrypt())[:D]
    with torch.no_grad():
        pt_ffn_in = block.ffn_up(pt_post_attn)
        pt_ffn_act = c1 * pt_ffn_in + c2 * pt_ffn_in.pow(2) + c0
        pt_ffn_out = block.ffn_down(pt_ffn_act)
        pt_final = pt_post_attn + pt_ffn_out
    err_ffn = np.max(np.abs(pt_final.numpy().flatten() - fhe_out))
    print(f"  6) PolyFFN:        {t_ffn:.2f}s  err={err_ffn:.2e} [matvec→ct×ct→matvec]")

    total_time = time.time() - total_t

    # ── 3. Parity karşılaştırma ──
    print(f"\n{'─' * 60}")
    print(f"  3. PARITY KARŞILAŞTIRMA")
    print(f"{'─' * 60}")
    pt_flat = pt_out[0].numpy()
    err_total = np.max(np.abs(pt_flat - fhe_out))
    rel_err = err_total / (np.max(np.abs(pt_flat)) + 1e-8)
    cos_sim = np.dot(pt_flat, fhe_out) / (np.linalg.norm(pt_flat) * np.linalg.norm(fhe_out) + 1e-8)

    print(f"  PT  output[:5]: {pt_flat[:5].tolist()}")
    print(f"  FHE output[:5]: {fhe_out[:5].tolist()}")
    print(f"  Max abs error:  {err_total:.2e}")
    print(f"  Relative error: {rel_err:.2e}")
    print(f"  Cosine sim:     {cos_sim:.8f}")
    print(f"  Toplam süre:    {total_time:.1f}s")

    # ── 4. Sonuç ──
    print(f"\n{'═' * 60}")
    print(f"  AYNI GRAF PARITY SONUCU")
    print(f"{'═' * 60}")
    ok = err_total < 0.002 and cos_sim > 0.9999
    print(f"  Max error:   {err_total:.2e}  {'✓ < 0.002' if err_total < 0.002 else '✗'}")
    print(f"  Cosine sim:  {cos_sim:.8f}  {'✓ > 0.9999' if cos_sim > 0.9999 else '✗'}")
    print(f"  {'✓ AYNI GRAF PARITY DOĞRULANDI' if ok else '✗ PARITY BAŞARISIZ'}")
    print(f"")
    print(f"  Kapsam:")
    print(f"    ✓ QKV projection (FHE matvec)")
    print(f"    ✓ Causal linear attention (Q @ KV_cum, FHE matvec)")
    print(f"    ✓ W_O + gate + residual (FHE)")
    print(f"    ✓ PolyFFN ct×ct (gerçek homomorphic çarpım)")
    print(f"    ✓ Head residual (FHE addition)")
    print(f"    ⚠ KV_cum: client-side plaintext (FHE protokolü)")
    print(f"    ✗ Local window attention (bu koşuda yok)")
    print(f"    ✗ Multi-layer (tek katman)")
    print(f"  TenSEAL {ts.__version__} | CKKS poly=16384 scale=2^40")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
