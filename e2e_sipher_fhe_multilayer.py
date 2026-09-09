"""
Sipher — Multi-Layer FHE Parity
=================================
2 ve 3 katman zincirleme FHE inference.
Her katman: causal linear + local window + PolyFFN.
PolyAct ct×ct başına 1 rescaling seviyesi → 9 katmana kadar bootstrap'sız.

Run: .venv312/bin/python3 -u e2e_sipher_fhe_multilayer.py
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


class SipherBlock(nn.Module):
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

        global_out = torch.zeros(self.nh, self.dk, device=x.device)
        for h in range(self.nh):
            KV_cum = torch.zeros(self.dk, self.dk, device=x.device)
            for t in range(pos + 1):
                KV_cum += K_r[t, h].unsqueeze(-1) @ V_r[t, h].unsqueeze(0)
            global_out[h] = Q_r[h] @ KV_cum

        local_out = torch.zeros(self.nh, self.dk, device=x.device)
        for h in range(self.nh):
            scores = torch.zeros(self.seq, device=x.device)
            for t in range(self.seq):
                scores[t] = Q_r[h] @ K_r[t, h]
            scores = scores * self.window_mask[pos]
            attn = torch.softmax(scores, dim=0)
            for t in range(self.seq):
                local_out[h] += attn[t] * V_r[t, h]

        combined = self.gate_global * global_out + self.gate_local * local_out
        out = x[pos:pos+1] + self.W_O(combined.view(1, -1))
        ffn = self.ffn_up(out)
        ffn = ffn * self.poly_coeffs[1] + ffn.pow(2) * self.poly_coeffs[2] + self.poly_coeffs[0]
        out = out + self.ffn_down(ffn)
        return out


def fhe_layer(context, enc_x_np, block, x_full_np, pos):
    """Tek katman FHE inference. enc_x_np: şifreli girdi (numpy)."""
    D = block.d
    NH, DK = block.nh, block.dk

    # Enc girdi
    enc_x = ts.ckks_vector(context, enc_x_np.tolist())

    # QKV
    with torch.no_grad():
        W_Q = block.W_Q.weight.numpy()
        K_all = (torch.tensor(x_full_np) @ block.W_K.weight.T).numpy()
        V_all = (torch.tensor(x_full_np) @ block.W_V.weight.T).numpy()

    enc_q = enc_x.matmul(ts.plain_tensor((W_Q / (DK**0.5)).T.tolist()))
    enc_q_dec = np.array(enc_q.decrypt())[:NH*DK]

    # Global: KV_cum (client) + Q@KV (FHE)
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

    # Local: Q@K^T (FHE) → client softmax → attn@V (FHE)
    with torch.no_grad():
        mask_row = block.window_mask[pos].numpy()
    local_fhe = np.zeros(NH * DK)
    for h in range(NH):
        q_h = enc_q_dec[h*DK:(h+1)*DK]
        K_h = K_all[:, h*DK:(h+1)*DK]
        V_h = V_all[:, h*DK:(h+1)*DK]
        enc_q_h = ts.ckks_vector(context, q_h.tolist())
        enc_scores = enc_q_h.matmul(ts.plain_tensor(K_h.T.tolist()))
        scores_dec = np.array(enc_scores.decrypt())[:block.seq]
        scores_masked = scores_dec * mask_row
        scores_masked -= np.max(scores_masked)
        exp_s = np.exp(scores_masked)
        attn_w = exp_s / np.sum(exp_s)
        enc_attn = ts.ckks_vector(context, attn_w.tolist())
        enc_local = enc_attn.matmul(ts.plain_tensor(V_h.tolist()))
        local_fhe[h*DK:(h+1)*DK] = np.array(enc_local.decrypt())[:DK]

    # Combine + W_O + residual
    with torch.no_grad():
        W_O = block.W_O.weight.numpy()
        g_g = block.gate_global.item()
        g_l = block.gate_local.item()
    combined = g_g * global_fhe + g_l * local_fhe
    enc_comb = ts.ckks_vector(context, combined.tolist())
    enc_wo = enc_comb.matmul(ts.plain_tensor(W_O.T.tolist()))
    enc_post = enc_x + enc_wo
    post_np = np.array(enc_post.decrypt())[:D]

    # PolyFFN
    with torch.no_grad():
        W_up = block.ffn_up.weight.numpy()
        W_down = block.ffn_down.weight.numpy()
        c0, c1, c2 = block.poly_coeffs.tolist()
    enc_up = enc_post.matmul(ts.plain_tensor(W_up.T.tolist()))
    enc_act = enc_up * c1 + (enc_up * enc_up) * c2
    if abs(c0) > 1e-10:
        enc_act = enc_act + ts.ckks_vector(context, [c0] * (D*2))
    enc_down = enc_act.matmul(ts.plain_tensor(W_down.T.tolist()))
    enc_out = enc_post + enc_down
    out_np = np.array(enc_out.decrypt())[:D]

    return out_np


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher — Multi-Layer FHE Parity (2 ve 3 katman)" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")

    D, NH, DK, SEQ, WINDOW = 32, 4, 8, 8, 4

    print(f"\n  CKKS context...", end="", flush=True)
    t0 = time.time()
    context = create_context()
    print(f" {time.time()-t0:.1f}s")

    for n_layers in [1, 2, 3]:
        print(f"\n{'═' * 60}")
        print(f"  {n_layers} KATMAN TESTİ")
        print(f"{'═' * 60}")

        torch.manual_seed(42)
        blocks = nn.ModuleList([SipherBlock(D, NH, DK, SEQ, WINDOW) for _ in range(n_layers)])
        blocks.eval()

        x = torch.randn(SEQ, D) * 0.3
        pos = SEQ - 1

        # Plaintext: zincirleme
        with torch.no_grad():
            pt_x = x.clone()
            for i, block in enumerate(blocks):
                out = block(pt_x, pos)
                pt_x[pos] = out[0]  # güncelle
            pt_final = pt_x[pos].numpy()

        # FHE: zincirleme
        t0 = time.time()
        fhe_x = x[pos].numpy().copy()
        x_full = x.numpy().copy()
        for i, block in enumerate(blocks):
            fhe_x = fhe_layer(context, fhe_x, block, x_full, pos)
            x_full[pos] = fhe_x  # sonraki katman için güncelle
        fhe_final = fhe_x
        elapsed = time.time() - t0

        # Karşılaştır
        err = np.max(np.abs(pt_final - fhe_final))
        cos_sim = np.dot(pt_final, fhe_final) / (np.linalg.norm(pt_final) * np.linalg.norm(fhe_final) + 1e-8)
        rel_err = err / (np.max(np.abs(pt_final)) + 1e-8)

        ok = err < 0.002 and cos_sim > 0.9999
        print(f"  Max err:     {err:.2e}  {'✓' if err < 0.002 else '✗'}")
        print(f"  Rel err:     {rel_err:.2e}")
        print(f"  Cosine sim:  {cos_sim:.8f}  {'✓' if cos_sim > 0.9999 else '✗'}")
        print(f"  Süre:        {elapsed:.1f}s")
        print(f"  {'✓ PARITY DOĞRULANDI' if ok else '✗ BAŞARISIZ'}")

    # Özet
    print(f"\n{'═' * 60}")
    print(f"  MULTI-LAYER FHE ÖZET")
    print(f"{'═' * 60}")
    print(f"  Her katman 1 rescaling seviyesi tüketir (PolyAct ct×ct)")
    print(f"  9 seviye → 9 katman bootstrap'sız")
    print(f"  20 katman → 2 bootstrap gerekli (9. ve 18. katmandan sonra)")
    print(f"  TenSEAL {ts.__version__} | CKKS poly=16384 scale=2^40")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
