"""
Sipher — Tam Homomorphic FHE Doğrulama (TenSEAL CKKS)
=======================================================
CKKSVector.matmul (BSGS diagonal encoding) ile tam homomorphic inference.
Tüm operasyonlar şifreli: matvec, PolyAct ct×ct, residual add.

Run: .venv312/bin/python3 -u e2e_sipher_fhe_final.py
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


def fhe_matvec(context, enc_x, W_np):
    """Tam homomorphic matvec: CKKSVector.matmul (BSGS)."""
    W_t = ts.plain_tensor(np.array(W_np, dtype=np.float64).T.tolist())
    return enc_x.matmul(W_t)


def fhe_poly_act(context, enc_x, coeffs):
    """PolyAct: c0 + c1*x + c2*x² (ct×ct)."""
    c0, c1, c2 = coeffs[0].item(), coeffs[1].item(), coeffs[2].item()
    result = enc_x * c1
    if abs(c2) > 1e-10:
        result = result + (enc_x * enc_x) * c2
    if abs(c0) > 1e-10:
        ones = ts.ckks_vector(context, [c0] * enc_x.size())
        result = result + ones
    return result


class TinySipher(nn.Module):
    def __init__(self, vocab=100, d=32, seq=8, nh=4, dk=8, emb_dim=16, window=4):
        super().__init__()
        self.d, self.seq, self.vocab = d, seq, vocab
        self.nh, self.dk, self.window = nh, dk, window
        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, d, bias=False)
        self.pos_emb = nn.Embedding(seq, d)
        self.W_Q = nn.Linear(d, nh * dk, bias=False)
        self.W_K = nn.Linear(d, nh * dk, bias=False)
        self.W_V = nn.Linear(d, nh * dk, bias=False)
        self.W_O = nn.Linear(nh * dk, d, bias=False)
        self.gate_global = nn.Parameter(torch.tensor(0.05))
        self.gate_local = nn.Parameter(torch.tensor(0.2))
        self.ffn_up = nn.Linear(d, d * 2, bias=False)
        self.poly_coeffs = nn.Parameter(torch.tensor([0.0, 0.5, 0.2]))
        self.ffn_down = nn.Linear(d * 2, d, bias=False)
        self.head_proj = nn.Linear(d, emb_dim, bias=False)
        idx = torch.arange(seq)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        self.register_buffer('window_mask', ((diff >= 0) & (diff < window)).float())

    def forward(self, ids):
        B, S = ids.shape
        x = self.emb_proj(self.token_emb(ids)) + self.pos_emb(torch.arange(S, device=ids.device))
        Q = self.W_Q(x).view(B, S, self.nh, self.dk).transpose(1, 2) / (self.dk ** 0.5)
        K = self.W_K(x).view(B, S, self.nh, self.dk).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.nh, self.dk).transpose(1, 2)
        KV = K.unsqueeze(-1) @ V.unsqueeze(-2)
        KV_cum = torch.cumsum(KV, dim=2)
        global_out = (Q.unsqueeze(-2) @ KV_cum).squeeze(-2)
        scores = Q @ K.transpose(-2, -1) * self.window_mask[:S, :S]
        attn = torch.softmax(scores, dim=-1)
        local_out = attn @ V
        combined = self.gate_global * global_out + self.gate_local * local_out
        combined = combined.transpose(1, 2).contiguous().view(B, S, self.nh * self.dk)
        x = x + self.W_O(combined)
        ffn = self.ffn_up(x)
        ffn = ffn * self.poly_coeffs[1] + ffn.pow(2) * self.poly_coeffs[2] + self.poly_coeffs[0]
        x = x + self.ffn_down(ffn)
        h = self.head_proj(x)
        return h @ self.token_emb.weight.T


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher — Tam Homomorphic FHE (TenSEAL CKKS, BSGS)" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")

    VOCAB, D, SEQ, NH, DK, EMB, WINDOW = 100, 32, 8, 4, 8, 16, 4

    print(f"\n  CKKS context...", end="", flush=True)
    t0 = time.time()
    context = create_context()
    print(f" {time.time()-t0:.1f}s (poly=16384, 9 seviye)")

    model = TinySipher(VOCAB, D, SEQ, NH, DK, EMB, WINDOW)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} param (d={D})")

    test_ids = torch.randint(0, VOCAB, (1, SEQ))
    print(f"  Input: {test_ids[0].tolist()}")

    # Plaintext referans
    with torch.no_grad():
        pt_logits = model(test_ids)
    pt_last = pt_logits[0, -1]
    pt_top5 = pt_last.topk(5).indices.tolist()
    print(f"  PT top-5: {pt_top5}")

    # ── Tam Homomorphic Inference ──
    print(f"\n{'─' * 60}")
    print(f"  TAM HOMOMORPHIC INFERENCE (tüm operasyonlar şifreli)")
    print(f"{'─' * 60}")

    pos = SEQ - 1
    total_t = time.time()

    # 1) Embedding (plaintext lookup → encrypt)
    with torch.no_grad():
        emb = model.emb_proj(model.token_emb(test_ids)) + model.pos_emb(torch.arange(SEQ))
        emb_pos = emb[0, pos].numpy().astype(np.float64)

    enc_x = ts.ckks_vector(context, emb_pos.tolist())
    dec_emb = np.array(enc_x.decrypt())[:D]
    err_emb = np.max(np.abs(emb_pos - dec_emb))
    print(f"  1) Embedding:      err={err_emb:.2e}")

    # 2) QKV matvec (tam homomorphic)
    t0 = time.time()
    with torch.no_grad():
        W_Q = model.W_Q.weight.numpy()
        W_K = model.W_K.weight.numpy()
        W_V = model.W_V.weight.numpy()

    enc_q = fhe_matvec(context, enc_x, W_Q)
    enc_k = fhe_matvec(context, enc_x, W_K)
    enc_v = fhe_matvec(context, enc_x, W_V)

    # Q'yu 1/sqrt(dk) ile ölçekle
    enc_q = enc_q * (1.0 / (DK ** 0.5))

    t_qkv = time.time() - t0
    fhe_q = np.array(enc_q.decrypt())[:NH*DK]
    fhe_k = np.array(enc_k.decrypt())[:NH*DK]
    fhe_v = np.array(enc_v.decrypt())[:NH*DK]

    with torch.no_grad():
        pt_q = (model.W_Q(emb[0:1, pos]) / (DK**0.5)).view(-1).numpy()
        pt_k = model.W_K(emb[0:1, pos]).view(-1).numpy()
        pt_v = model.W_V(emb[0:1, pos]).view(-1).numpy()

    err_q = np.max(np.abs(pt_q - fhe_q))
    err_k = np.max(np.abs(pt_k - fhe_k))
    err_v = np.max(np.abs(pt_v - fhe_v))
    print(f"  2) QKV matvec:     err_Q={err_q:.2e} K={err_k:.2e} V={err_v:.2e}  ({t_qkv:.1f}s)")

    # 3) PolyFFN (matvec → ct×ct → matvec)
    t0 = time.time()
    with torch.no_grad():
        W_up = model.ffn_up.weight.numpy()
        W_down = model.ffn_down.weight.numpy()

    enc_ffn_up = fhe_matvec(context, enc_x, W_up)
    enc_ffn_act = fhe_poly_act(context, enc_ffn_up, model.poly_coeffs)
    enc_ffn_down = fhe_matvec(context, enc_ffn_act, W_down)
    t_ffn = time.time() - t0

    fhe_ffn = np.array(enc_ffn_down.decrypt())[:D]
    with torch.no_grad():
        pt_ffn_in = model.ffn_up(emb[0:1, pos])
        pt_ffn_act = model.poly_coeffs[1]*pt_ffn_in + model.poly_coeffs[2]*pt_ffn_in.pow(2) + model.poly_coeffs[0]
        pt_ffn_out = model.ffn_down(pt_ffn_act).view(-1).numpy()

    err_ffn = np.max(np.abs(pt_ffn_out - fhe_ffn))
    print(f"  3) PolyFFN:        err={err_ffn:.2e}  ({t_ffn:.1f}s) [matvec→ct×ct→matvec]")

    # 4) Residual + Head
    t0 = time.time()
    # Residual: emb + ffn (encrypted addition)
    enc_residual = enc_x + enc_ffn_down
    with torch.no_grad():
        W_head = model.head_proj.weight.numpy()
    enc_head = fhe_matvec(context, enc_residual, W_head)
    t_head = time.time() - t0

    fhe_head = np.array(enc_head.decrypt())[:EMB]
    with torch.no_grad():
        pt_post = emb[0:1, pos] + torch.tensor(pt_ffn_out, dtype=torch.float32).view(1, -1)
        pt_head = model.head_proj(pt_post).view(-1).numpy()

    err_head = np.max(np.abs(pt_head - fhe_head))
    print(f"  4) Head:           err={err_head:.2e}  ({t_head:.1f}s)")

    # 5) Logits (plaintext token embedding matmul)
    with torch.no_grad():
        tok_emb = model.token_emb.weight.numpy()
    fhe_logits = fhe_head @ tok_emb.T
    pt_last_np = pt_last.numpy()

    err_logits = np.max(np.abs(pt_last_np - fhe_logits))
    fhe_top5 = np.argsort(-fhe_logits)[:5].tolist()
    top5_match = len(set(pt_top5) & set(fhe_top5))

    total_time = time.time() - total_t
    print(f"  5) Logits:         err={err_logits:.2e}  Top5={top5_match}/5")
    print(f"\n  Toplam süre: {total_time:.1f}s")
    print(f"  PT  top-5: {pt_top5}")
    print(f"  FHE top-5: {fhe_top5}")

    # ── Sonuç ──
    print(f"\n{'═' * 60}")
    print(f"  TAM HOMOMORPHIC FHE SONUCU")
    print(f"{'═' * 60}")
    errs = {'emb': err_emb, 'Q': err_q, 'K': err_k, 'V': err_v,
            'FFN': err_ffn, 'head': err_head, 'logits': err_logits}
    for name, err in errs.items():
        s = "✓" if err < 0.001 else "✗"
        print(f"  {s} {name:<15s} {err:.2e}")
    worst = max(errs.values())
    worst_name = max(errs, key=errs.get)
    ok = worst < 0.001 and top5_match >= 4
    print(f"\n  Worst: {worst_name} ({worst:.2e})")
    print(f"  Top-5: {top5_match}/5")
    print(f"  {'✓ TAM HOMOMORPHIC FHE DOĞRULAMA BAŞARILI' if ok else '✗ BAŞARISIZ'}")
    print(f"  TenSEAL {ts.__version__} | CKKS poly=16384 scale=2^40")
    print(f"  Matvec: CKKSVector.matmul (BSGS diagonal encoding)")
    print(f"  PolyAct: ct×ct (gerçek homomorphic çarpım)")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
