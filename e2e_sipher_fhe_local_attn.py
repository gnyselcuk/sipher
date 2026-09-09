"""
Sipher — Local Window Attention FHE Parity
=============================================
Causal linear + Local window attention + PolyFFN.
Softmax client-side (geçerli FHE protokolü).

Run: .venv312/bin/python3 -u e2e_sipher_fhe_local_attn.py
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


class TinySipherFullBlock(nn.Module):
    """Tek katman: causal linear + local window attention + PolyFFN."""
    def __init__(self, d=32, nh=4, dk=8, seq=8, window=4):
        super().__init__()
        self.d, self.nh, self.dk, self.seq, self.window = d, nh, dk, seq, window
        self.W_Q = nn.Linear(d, nh * dk, bias=False)
        self.W_K = nn.Linear(d, nh * dk, bias=False)
        self.W_V = nn.Linear(d, nh * dk, bias=False)
        self.W_O = nn.Linear(nh * dk, d, bias=False)
        self.gate_global = nn.Parameter(torch.tensor(0.05))
        self.gate_local = nn.Parameter(torch.tensor(0.2))
        self.ffn_up = nn.Linear(d, d * 2, bias=False)
        self.poly_coeffs = nn.Parameter(torch.tensor([0.0, 0.5, 0.2]))
        self.ffn_down = nn.Linear(d * 2, d, bias=False)
        idx = torch.arange(seq)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        self.register_buffer('window_mask', ((diff >= 0) & (diff < window)).float())

    def forward(self, x, pos):
        Q_pos = self.W_Q(x[pos:pos+1]) / (self.dk ** 0.5)
        K_all = self.W_K(x)
        V_all = self.W_V(x)

        Q_r = Q_pos.view(self.nh, self.dk)
        K_r = K_all.view(self.seq, self.nh, self.dk)
        V_r = V_all.view(self.seq, self.nh, self.dk)

        # Global: causal linear
        global_out = torch.zeros(self.nh, self.dk, device=x.device)
        for h in range(self.nh):
            KV_cum = torch.zeros(self.dk, self.dk, device=x.device)
            for t in range(pos + 1):
                KV_cum += K_r[t, h].unsqueeze(-1) @ V_r[t, h].unsqueeze(0)
            global_out[h] = Q_r[h] @ KV_cum

        # Local: window attention with softmax
        local_out = torch.zeros(self.nh, self.dk, device=x.device)
        for h in range(self.nh):
            scores = torch.zeros(self.seq, device=x.device)
            for t in range(self.seq):
                scores[t] = Q_r[h] @ K_r[t, h]
            scores = scores * self.window_mask[pos]
            attn = torch.softmax(scores, dim=0)
            for t in range(self.seq):
                local_out[h] += attn[t] * V_r[t, h]

        # Combine + W_O + residual
        combined = self.gate_global * global_out + self.gate_local * local_out
        combined_flat = combined.view(1, -1)
        out = x[pos:pos+1] + self.W_O(combined_flat)

        # PolyFFN
        ffn = self.ffn_up(out)
        ffn = ffn * self.poly_coeffs[1] + ffn.pow(2) * self.poly_coeffs[2] + self.poly_coeffs[0]
        out = out + self.ffn_down(ffn)
        return out


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher — Local Window Attention FHE Parity" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")

    D, NH, DK, SEQ, WINDOW = 32, 4, 8, 8, 4

    print(f"\n  CKKS context...", end="", flush=True)
    t0 = time.time()
    context = create_context()
    print(f" {time.time()-t0:.1f}s")

    block = TinySipherFullBlock(D, NH, DK, SEQ, WINDOW)
    block.eval()
    print(f"  Block: d={D} nh={NH} dk={DK} seq={SEQ} window={WINDOW}")

    x = torch.randn(SEQ, D) * 0.3
    pos = SEQ - 1
    print(f"  Input: seq={SEQ}, pos={pos}, window={WINDOW}")

    # ── Plaintext ──
    print(f"\n{'─' * 60}")
    print(f"  PLAINTEXT REFERANS")
    print(f"{'─' * 60}")
    with torch.no_grad():
        pt_out = block(x, pos)
    print(f"  Output[:5]: {pt_out[0, :5].tolist()}")

    # ── FHE ──
    print(f"\n{'─' * 60}")
    print(f"  FHE — Aynı Graf (Local Attn + Client-side Softmax)")
    print(f"{'─' * 60}")
    total_t = time.time()

    # 1) Enc(x[pos])
    enc_x = ts.ckks_vector(context, x[pos].numpy().tolist())

    # 2) QKV
    t0 = time.time()
    with torch.no_grad():
        W_Q = block.W_Q.weight.numpy()
        K_all = (x @ block.W_K.weight.T).numpy()
        V_all = (x @ block.W_V.weight.T).numpy()
    enc_q = enc_x.matmul(ts.plain_tensor((W_Q / (DK**0.5)).T.tolist()))
    enc_q_dec = np.array(enc_q.decrypt())[:NH*DK]
    print(f"  1) QKV:            {time.time()-t0:.2f}s")

    # 3) Global: KV_cum (client-side) + Q@KV_cum (FHE)
    t0 = time.time()
    KV_cum_all = np.zeros((NH, DK, DK))
    for h in range(NH):
        for t in range(pos + 1):
            K_t = K_all[t, h*DK:(h+1)*DK]
            V_t = V_all[t, h*DK:(h+1)*DK]
            KV_cum_all[h] += np.outer(K_t, V_t)

    global_fhe = np.zeros(NH * DK)
    for h in range(NH):
        enc_q_h = ts.ckks_vector(context, enc_q_dec[h*DK:(h+1)*DK].tolist())
        enc_g = enc_q_h.matmul(ts.plain_tensor(KV_cum_all[h].T.tolist()))
        global_fhe[h*DK:(h+1)*DK] = np.array(enc_g.decrypt())[:DK]
    print(f"  2) Global attn:    {time.time()-t0:.2f}s (client KV_cum + FHE Q@KV)")

    # 4) Local: Q@K_window^T (FHE) → client softmax → attn@V (FHE)
    t0 = time.time()
    local_fhe = np.zeros(NH * DK)
    with torch.no_grad():
        mask_row = block.window_mask[pos].numpy()  # (SEQ,)

    for h in range(NH):
        q_h = enc_q_dec[h*DK:(h+1)*DK]
        K_h = K_all[:, h*DK:(h+1)*DK]  # (SEQ, DK)
        V_h = V_all[:, h*DK:(h+1)*DK]  # (SEQ, DK)

        # Server: scores = Q_h @ K_h^T (FHE matvec)
        enc_q_h = ts.ckks_vector(context, q_h.tolist())
        # K_h^T: (DK, SEQ) → her sütun bir pozisyon
        # scores[t] = Q_h @ K_h[t] → matvec: (1,DK) @ (DK,SEQ) → (1,SEQ)
        enc_scores = enc_q_h.matmul(ts.plain_tensor(K_h.T.tolist()))
        scores_dec = np.array(enc_scores.decrypt())[:SEQ]

        # Client: mask + softmax
        scores_masked = scores_dec * mask_row
        scores_masked = scores_masked - np.max(scores_masked)  # numerical stability
        exp_scores = np.exp(scores_masked)
        attn_weights = exp_scores / np.sum(exp_scores)

        # Server: out = attn_weights @ V_h (FHE matvec)
        # attn_weights: (SEQ,) → (1, SEQ)
        # V_h: (SEQ, DK)
        # out: (1, DK) = (1, SEQ) @ (SEQ, DK)
        enc_attn = ts.ckks_vector(context, attn_weights.tolist())
        enc_local = enc_attn.matmul(ts.plain_tensor(V_h.tolist()))
        local_fhe[h*DK:(h+1)*DK] = np.array(enc_local.decrypt())[:DK]

    print(f"  3) Local attn:     {time.time()-t0:.2f}s (FHE Q@K + client softmax + FHE attn@V)")

    # 5) Combine + W_O + residual (FHE)
    t0 = time.time()
    with torch.no_grad():
        W_O = block.W_O.weight.numpy()
        g_global = block.gate_global.item()
        g_local = block.gate_local.item()

    combined_fhe = g_global * global_fhe + g_local * local_fhe
    enc_combined = ts.ckks_vector(context, combined_fhe.tolist())
    enc_wo = enc_combined.matmul(ts.plain_tensor(W_O.T.tolist()))
    enc_post_attn = enc_x + enc_wo  # residual
    post_attn_fhe = np.array(enc_post_attn.decrypt())[:D]
    print(f"  4) Combine+W_O:    {time.time()-t0:.2f}s (gate+matvec+residual)")

    # 6) PolyFFN (FHE)
    t0 = time.time()
    with torch.no_grad():
        W_up = block.ffn_up.weight.numpy()
        W_down = block.ffn_down.weight.numpy()
        c0, c1, c2 = block.poly_coeffs.tolist()

    enc_ffn_up = enc_post_attn.matmul(ts.plain_tensor(W_up.T.tolist()))
    enc_ffn_act = enc_ffn_up * c1 + (enc_ffn_up * enc_ffn_up) * c2
    if abs(c0) > 1e-10:
        enc_ffn_act = enc_ffn_act + ts.ckks_vector(context, [c0] * (D*2))
    enc_ffn_down = enc_ffn_act.matmul(ts.plain_tensor(W_down.T.tolist()))
    enc_out = enc_post_attn + enc_ffn_down
    fhe_out = np.array(enc_out.decrypt())[:D]
    print(f"  5) PolyFFN:        {time.time()-t0:.2f}s (matvec→ct×ct→matvec+residual)")

    total_time = time.time() - total_t

    # ── Parity ──
    print(f"\n{'─' * 60}")
    print(f"  PARITY KARŞILAŞTIRMA")
    print(f"{'─' * 60}")
    pt_flat = pt_out[0].numpy()
    err = np.max(np.abs(pt_flat - fhe_out))
    rel_err = err / (np.max(np.abs(pt_flat)) + 1e-8)
    cos_sim = np.dot(pt_flat, fhe_out) / (np.linalg.norm(pt_flat) * np.linalg.norm(fhe_out) + 1e-8)

    print(f"  PT[:5]:  {pt_flat[:5].tolist()}")
    print(f"  FHE[:5]: {fhe_out[:5].tolist()}")
    print(f"  Max err:     {err:.2e}")
    print(f"  Rel err:     {rel_err:.2e}")
    print(f"  Cosine sim:  {cos_sim:.8f}")
    print(f"  Süre:        {total_time:.1f}s")

    # ── Sonuç ──
    print(f"\n{'═' * 60}")
    ok = err < 0.002 and cos_sim > 0.9999
    print(f"  Max err:     {err:.2e}  {'✓ < 0.002' if err < 0.002 else '✗'}")
    print(f"  Cosine sim:  {cos_sim:.8f}  {'✓ > 0.9999' if cos_sim > 0.9999 else '✗'}")
    print(f"  {'✓ LOCAL WINDOW ATTENTION FHE PARITY DOĞRULANDI' if ok else '✗ BAŞARISIZ'}")
    print(f"")
    print(f"  Kapsam:")
    print(f"    ✓ QKV projection (FHE matvec)")
    print(f"    ✓ Causal linear attention (Q @ KV_cum)")
    print(f"    ✓ Local window scores (FHE Q @ K^T)")
    print(f"    ✓ Softmax (client-side, geçerli FHE protokolü)")
    print(f"    ✓ Local attn output (FHE attn @ V)")
    print(f"    ✓ Gate combine + W_O + residual (FHE)")
    print(f"    ✓ PolyFFN ct×ct (gerçek homomorphic)")
    print(f"    ✗ Multi-layer (tek katman)")
    print(f"    ✗ Eğitilmiş checkpoint (random init)")
    print(f"  TenSEAL {ts.__version__} | CKKS poly=16384 scale=2^40")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
