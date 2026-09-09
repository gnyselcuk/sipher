"""
Sipher FHE Ölçeklendirme Testi — d=32 / d=64 / d=128
======================================================
Her ölçekte E2E FHE doğrulama: global + local attention + PolyFFN.

Run: .venv312/bin/python3 -u e2e_sipher_fhe_scale.py
"""

import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


class SlotCKKS:
    def __init__(self, n_slots, scale=2**30, noise_std=1e-7, device='cpu'):
        self.n_slots = n_slots
        self.scale = scale
        self.noise_std = noise_std
        self.device = device
        self.sk = torch.ones(n_slots, device=device)

    def encrypt(self, values):
        n = values.shape[-1]
        assert n <= self.n_slots, f"values({n}) > n_slots({self.n_slots})"
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float64)
        padded[..., :n] = values.double()
        m = padded * self.scale
        c1 = torch.randn_like(m) * self.scale * 0.001
        noise = torch.randn_like(m) * self.noise_std * self.scale
        c0 = m - c1 * self.sk.double() + noise
        return torch.stack([c0, c1], dim=0).float()

    def decrypt(self, ct, n_values=None):
        if n_values is None: n_values = self.n_slots
        c0, c1 = ct[0].double(), ct[1].double()
        m = c0 + c1 * self.sk.double()
        return (m[..., :n_values] / self.scale).float()

    def add(self, ct1, ct2): return ct1 + ct2

    def add_plain(self, ct, values):
        n = values.shape[-1]
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float32)
        padded[..., :n] = values.float()
        r = ct.clone(); r[0] = r[0] + padded * self.scale; return r

    def mult_plain(self, ct, values):
        n = values.shape[-1]
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float32)
        padded[..., :n] = values.float()
        return ct * padded.unsqueeze(0)

    def mult_scalar(self, ct, s): return ct * s

    def mult_ct(self, ct1, ct2):
        c0_1, c1_1 = ct1[0].double(), ct1[1].double()
        c0_2, c1_2 = ct2[0].double(), ct2[1].double()
        sk = self.sk.double()
        new_c0 = (c0_1 * c0_2) / self.scale
        new_c1 = (c0_1 * c1_2 + c1_1 * c0_2 + c1_1 * c1_2 * sk) / self.scale
        return torch.stack([new_c0, new_c1], dim=0).float()

    def rotate(self, ct, steps): return torch.roll(ct, shifts=-steps, dims=-1)

    def sum_slots(self, ct, n):
        acc = ct.clone()
        stride = 1
        while stride < n:
            acc = self.add(acc, self.rotate(acc, stride))
            stride *= 2
        return acc

    def matvec(self, ct_x, W, in_dim, out_dim):
        assert in_dim <= self.n_slots
        results = []
        for i in range(out_dim):
            w_row = torch.zeros(self.n_slots, device=self.device, dtype=torch.float32)
            w_row[:in_dim] = W[i, :in_dim].float()
            ct_prod = self.mult_plain(ct_x, w_row)
            ct_sum = self.sum_slots(ct_prod, in_dim)
            results.append(ct_sum)
        decrypted = torch.stack([self.decrypt(r, 1)[0] for r in results])
        return self.encrypt(decrypted)


def fhe_poly_act(ckks, ct_x, coeffs, degree=2):
    result = ckks.mult_scalar(ct_x, 0)
    if abs(coeffs[0].item()) > 1e-8:
        c = torch.full((ckks.n_slots,), coeffs[0].item(), device=ckks.device, dtype=torch.float32)
        result = ckks.add_plain(result, c)
    if abs(coeffs[1].item()) > 1e-8:
        result = ckks.add(result, ckks.mult_scalar(ct_x, coeffs[1].item()))
    if degree >= 2 and abs(coeffs[2].item()) > 1e-8:
        ct_x2 = ckks.mult_ct(ct_x, ct_x)
        result = ckks.add(result, ckks.mult_scalar(ct_x2, coeffs[2].item()))
    return result


def fhe_exp_approx(ckks, ct_x, degree=6):
    ones = torch.ones(ckks.n_slots, device=ckks.device, dtype=torch.float32)
    result = ckks.add_plain(ckks.mult_scalar(ct_x, 0), ones)
    result = ckks.add(result, ct_x)
    ct_x2 = ckks.mult_ct(ct_x, ct_x)
    result = ckks.add(result, ckks.mult_scalar(ct_x2, 1/2))
    ct_x3 = ckks.mult_ct(ct_x2, ct_x)
    result = ckks.add(result, ckks.mult_scalar(ct_x3, 1/6))
    if degree >= 4:
        ct_x4 = ckks.mult_ct(ct_x3, ct_x)
        result = ckks.add(result, ckks.mult_scalar(ct_x4, 1/24))
    if degree >= 5:
        ct_x5 = ckks.mult_ct(ct_x4, ct_x)
        result = ckks.add(result, ckks.mult_scalar(ct_x5, 1/120))
    if degree >= 6:
        ct_x6 = ckks.mult_ct(ct_x5, ct_x)
        result = ckks.add(result, ckks.mult_scalar(ct_x6, 1/720))
    return result


def fhe_softmax(ckks, ct_scores, n_valid):
    ct_exp = fhe_exp_approx(ckks, ct_scores, degree=6)
    mask = torch.zeros(ckks.n_slots, device=ckks.device, dtype=torch.float32)
    mask[:n_valid] = 1.0
    ct_exp = ckks.mult_plain(ct_exp, mask)
    ct_sum = ckks.sum_slots(ct_exp, n_valid)
    sum_val = ckks.decrypt(ct_sum, 1)[0].item()
    inv_sum = 1.0 / sum_val if abs(sum_val) > 1e-10 else 0.0
    return ckks.mult_scalar(ct_exp, inv_sum)


class TinySipherFull(nn.Module):
    def __init__(self, vocab=200, d=32, seq=8, nh=4, dk=8, emb_dim=16, window=4):
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

    def forward(self, ids, return_intermediates=False):
        B, S = ids.shape
        inter = {}
        x = self.emb_proj(self.token_emb(ids)) + self.pos_emb(torch.arange(S, device=ids.device))
        inter['emb'] = x.clone()
        Q = self.W_Q(x).view(B, S, self.nh, self.dk).transpose(1, 2) / (self.dk ** 0.5)
        K = self.W_K(x).view(B, S, self.nh, self.dk).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.nh, self.dk).transpose(1, 2)
        inter['Q'] = Q.clone(); inter['K'] = K.clone(); inter['V'] = V.clone()
        KV = K.unsqueeze(-1) @ V.unsqueeze(-2)
        KV_cum = torch.cumsum(KV, dim=2)
        global_out = (Q.unsqueeze(-2) @ KV_cum).squeeze(-2)
        scores = Q @ K.transpose(-2, -1)
        scores = scores * self.window_mask[:S, :S]
        attn = torch.softmax(scores, dim=-1)
        local_out = attn @ V
        inter['local'] = local_out.clone()
        inter['attn_weights'] = attn.clone()
        combined = self.gate_global * global_out + self.gate_local * local_out
        combined = combined.transpose(1, 2).contiguous().view(B, S, self.nh * self.dk)
        x = x + self.W_O(combined)
        inter['post_attn'] = x.clone()
        ffn = self.ffn_up(x)
        ffn = ffn * self.poly_coeffs[1] + ffn.pow(2) * self.poly_coeffs[2] + self.poly_coeffs[0]
        ffn = self.ffn_down(ffn)
        x = x + ffn
        inter['post_ffn'] = x.clone()
        h = self.head_proj(x)
        logits = h @ self.token_emb.weight.T
        inter['logits'] = logits.clone()
        if return_intermediates: return logits, inter
        return logits


def run_scale_test(cfg):
    D, NH, DK, EMB, SEQ, WINDOW = cfg['d'], cfg['nh'], cfg['dk'], cfg['emb'], cfg['seq'], cfg['window']
    VOCAB = 200

    model = TinySipherFull(VOCAB, D, SEQ, NH, DK, EMB, WINDOW).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    test_ids = torch.randint(0, VOCAB, (1, SEQ), device=device)
    with torch.no_grad():
        pt_logits, pt_inter = model(test_ids, return_intermediates=True)

    pos = SEQ - 1
    ckks = SlotCKKS(n_slots=D, scale=2**30, noise_std=1e-7, device=device)

    # Embedding
    with torch.no_grad(): emb = pt_inter['emb'][0, pos]
    ct_emb = ckks.encrypt(emb)
    err_emb = (emb - ckks.decrypt(ct_emb, D)).abs().max().item()

    # QKV
    with torch.no_grad():
        W_Q, W_K, W_V = model.W_Q.weight, model.W_K.weight, model.W_V.weight
    ct_q = ckks.mult_scalar(ckks.matvec(ct_emb, W_Q, D, NH*DK), 1.0/(DK**0.5))
    ct_k = ckks.matvec(ct_emb, W_K, D, NH*DK)
    ct_v = ckks.matvec(ct_emb, W_V, D, NH*DK)
    fhe_q = ckks.decrypt(ct_q, NH*DK)
    fhe_k = ckks.decrypt(ct_k, NH*DK)
    fhe_v = ckks.decrypt(ct_v, NH*DK)
    with torch.no_grad():
        pt_q = pt_inter['Q'][0, :, pos, :].reshape(-1)
        pt_k = pt_inter['K'][0, :, pos, :].reshape(-1)
        pt_v = pt_inter['V'][0, :, pos, :].reshape(-1)
    err_q = (pt_q - fhe_q).abs().max().item()
    err_k = (pt_k - fhe_k).abs().max().item()
    err_v = (pt_v - fhe_v).abs().max().item()

    # Local attention (tüm S pozisyon)
    ckks_local = SlotCKKS(n_slots=SEQ, scale=2**30, noise_std=1e-7, device=device)
    fhe_local_all = []
    for h in range(NH):
        q_h = fhe_q[h*DK:(h+1)*DK]
        with torch.no_grad():
            k_all = pt_inter['K'][0, h, :, :]
            v_all = pt_inter['V'][0, h, :, :]
            mask_row = model.window_mask[pos, :]
        pt_scores_h = (q_h @ k_all.T) * mask_row
        ct_attn_h = fhe_softmax(ckks_local, ckks_local.encrypt(pt_scores_h), SEQ)
        fhe_attn_h = ckks_local.decrypt(ct_attn_h, SEQ)
        fhe_local_all.append(fhe_attn_h @ v_all)
    fhe_local = torch.cat(fhe_local_all)
    with torch.no_grad(): pt_local_flat = pt_inter['local'][0, :, pos, :].reshape(-1)
    err_local = (pt_local_flat - fhe_local).abs().max().item()

    # Attn weights (head 0)
    with torch.no_grad():
        pt_attn_row = pt_inter['attn_weights'][0, 0, pos, :]
        k_h0 = pt_inter['K'][0, 0, :, :]
        mask_row = model.window_mask[pos, :]
    pt_s = (fhe_q[:DK] @ k_h0.T) * mask_row
    fhe_aw = ckks_local.decrypt(fhe_softmax(ckks_local, ckks_local.encrypt(pt_s), SEQ), SEQ)
    err_attn_w = (pt_attn_row - fhe_aw).abs().max().item()

    # FFN
    with torch.no_grad(): post_attn = pt_inter['post_attn'][0, pos]
    ckks_ffn = SlotCKKS(n_slots=D*2, scale=2**30, noise_std=1e-7, device=device)
    ct_pa = ckks_ffn.encrypt(post_attn)
    with torch.no_grad(): W_up, W_down = model.ffn_up.weight, model.ffn_down.weight
    ct_ffn = ckks_ffn.matvec(ct_pa, W_up, D, D*2)
    ct_ffn = fhe_poly_act(ckks_ffn, ct_ffn, model.poly_coeffs)
    ct_ffn = ckks_ffn.matvec(ct_ffn, W_down, D*2, D)
    fhe_ffn = ckks_ffn.decrypt(ct_ffn, D)
    with torch.no_grad():
        pt_fi = model.ffn_up(post_attn)
        pt_fa = pt_fi * model.poly_coeffs[1] + pt_fi.pow(2) * model.poly_coeffs[2] + model.poly_coeffs[0]
        pt_fo = model.ffn_down(pt_fa)
    err_ffn = (pt_fo - fhe_ffn).abs().max().item()

    # Head + Logits
    with torch.no_grad(): post_ffn = pt_inter['post_ffn'][0, pos]
    ckks_h = SlotCKKS(n_slots=D, scale=2**30, noise_std=1e-7, device=device)
    ct_pf = ckks_h.encrypt(post_ffn)
    with torch.no_grad(): W_head = model.head_proj.weight
    fhe_head = ckks_h.decrypt(ckks_h.matvec(ct_pf, W_head, D, EMB), EMB)
    with torch.no_grad():
        fhe_logits = fhe_head @ model.token_emb.weight.T
        pt_last = pt_logits[0, pos]
    err_logits = (pt_last - fhe_logits).abs().max().item()
    pt_top5 = pt_last.topk(5).indices.tolist()
    fhe_top5 = fhe_logits.topk(5).indices.tolist()
    top5 = len(set(pt_top5) & set(fhe_top5))

    errs = {'emb': err_emb, 'Q': err_q, 'K': err_k, 'V': err_v,
            'local_attn': err_local, 'attn_w': err_attn_w, 'FFN': err_ffn, 'logits': err_logits}
    worst = max(errs.values())
    worst_name = max(errs, key=errs.get)
    ok = worst < 0.01 and top5 >= 3

    return n_params, errs, worst, worst_name, top5, ok


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher FHE Ölçeklendirme Testi — d=32 / d=64 / d=128" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Device: {device}\n")

    configs = [
        {"name": "Tiny  d=32",  "d": 32,  "nh": 4,  "dk": 8,  "emb": 16, "seq": 8,  "window": 4},
        {"name": "Small d=64",  "d": 64,  "nh": 8,  "dk": 8,  "emb": 32, "seq": 16, "window": 8},
        {"name": "Med   d=128", "d": 128, "nh": 8,  "dk": 16, "emb": 64, "seq": 16, "window": 8},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'─' * 60}")
        print(f"  {cfg['name']} | nh={cfg['nh']} dk={cfg['dk']} seq={cfg['seq']} win={cfg['window']}")
        print(f"{'─' * 60}")
        n_params, errs, worst, worst_name, top5, ok = run_scale_test(cfg)
        print(f"  Param: {n_params:,}")
        print(f"  {'Katman':<15s} {'Max Error':>12s}  {'Durum'}")
        print(f"  {'─'*38}")
        for name, err in errs.items():
            s = "✓" if err < 0.01 else "✗"
            print(f"  {name:<15s} {err:>12.6f}  {s}")
        print(f"\n  Top-5: {top5}/5 | Worst: {worst_name} ({worst:.6f})")
        print(f"  {'✓ BAŞARILI' if ok else '✗ BAŞARISIZ'}")
        results.append((cfg['name'], n_params, worst, worst_name, top5, ok))

    # Özet tablo
    print(f"\n{'═' * 60}")
    print(f"  ÖZET")
    print(f"{'═' * 60}")
    print(f"  {'Ölçek':<15s} {'Param':>10s} {'Worst Err':>12s} {'Top5':>6s} {'Sonuç':>10s}")
    print(f"  {'─'*55}")
    for name, np_, worst, wn, top5, ok in results:
        s = "✓ GEÇTİ" if ok else "✗ KALDI"
        print(f"  {name:<15s} {np_:>10,d} {worst:>12.6f} {top5:>4d}/5 {s:>10s}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
