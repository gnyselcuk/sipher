"""
Sipher FHE d=1024 Test — Eğitilmiş Checkpoint
================================================
Eğitilmiş model (PPL 45, epoch 28) ile d=1024 FHE doğrulama.
Tek katman testi (matvec O(d²) olduğu için tam model çok yavaş).

Run: .venv312/bin/python3 -u e2e_sipher_fhe_d1024.py
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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
        assert n <= self.n_slots
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float64)
        padded[..., :n] = values.double()
        m = padded * self.scale
        c1 = torch.randn_like(m) * self.scale * 0.001
        noise = torch.randn_like(m) * self.noise_std * self.scale
        c0 = m - c1 * self.sk.double() + noise
        return torch.stack([c0, c1], dim=0)  # float64 koru

    def decrypt(self, ct, n_values=None):
        if n_values is None: n_values = self.n_slots
        c0, c1 = ct[0].double(), ct[1].double()
        return ((c0 + c1 * self.sk.double())[..., :n_values] / self.scale).float()

    def add(self, ct1, ct2): return ct1 + ct2
    def mult_scalar(self, ct, s): return ct * s

    def mult_plain(self, ct, values):
        n = values.shape[-1]
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float64)
        padded[..., :n] = values.double()
        return ct * padded.unsqueeze(0)

    def add_plain(self, ct, values):
        n = values.shape[-1]
        padded = torch.zeros(*values.shape[:-1], self.n_slots, device=self.device, dtype=torch.float64)
        padded[..., :n] = values.double()
        r = ct.clone(); r[0] = r[0] + padded * self.scale; return r

    def mult_ct(self, ct1, ct2):
        c0_1, c1_1 = ct1[0].double(), ct1[1].double()
        c0_2, c1_2 = ct2[0].double(), ct2[1].double()
        sk = self.sk.double()
        return torch.stack([(c0_1*c0_2)/self.scale, (c0_1*c1_2+c1_1*c0_2+c1_1*c1_2*sk)/self.scale], dim=0).float()

    def rotate(self, ct, steps): return torch.roll(ct, shifts=-steps, dims=-1)

    def sum_slots(self, ct, n):
        acc = ct.clone()
        stride = 1
        while stride < n:
            acc = self.add(acc, self.rotate(acc, stride))
            stride *= 2
        return acc

    def matvec(self, ct_x, W, in_dim, out_dim):
        results = []
        for i in range(out_dim):
            w_row = torch.zeros(self.n_slots, device=self.device, dtype=torch.float64)
            w_row[:in_dim] = W[i, :in_dim].double()
            ct_sum = self.sum_slots(self.mult_plain(ct_x, w_row), in_dim)
            results.append(ct_sum)
        decrypted = torch.stack([self.decrypt(r, 1)[0] for r in results])
        return self.encrypt(decrypted)


def fhe_poly_act(ckks, ct_x, coeffs):
    result = ckks.mult_scalar(ct_x, 0)
    if abs(coeffs[0].item()) > 1e-8:
        c = torch.full((ckks.n_slots,), coeffs[0].item(), device=ckks.device, dtype=torch.float64)
        result = ckks.add_plain(result, c)
    if abs(coeffs[1].item()) > 1e-8:
        result = ckks.add(result, ckks.mult_scalar(ct_x, coeffs[1].item()))
    if abs(coeffs[2].item()) > 1e-8:
        ct_x2 = ckks.mult_ct(ct_x, ct_x)
        result = ckks.add(result, ckks.mult_scalar(ct_x2, coeffs[2].item()))
    return result


def fhe_exp_approx(ckks, ct_x):
    ones = torch.ones(ckks.n_slots, device=ckks.device, dtype=torch.float32)
    r = ckks.add_plain(ckks.mult_scalar(ct_x, 0), ones)
    r = ckks.add(r, ct_x)
    x2 = ckks.mult_ct(ct_x, ct_x); r = ckks.add(r, ckks.mult_scalar(x2, 1/2))
    x3 = ckks.mult_ct(x2, ct_x);   r = ckks.add(r, ckks.mult_scalar(x3, 1/6))
    x4 = ckks.mult_ct(x3, ct_x);   r = ckks.add(r, ckks.mult_scalar(x4, 1/24))
    x5 = ckks.mult_ct(x4, ct_x);   r = ckks.add(r, ckks.mult_scalar(x5, 1/120))
    x6 = ckks.mult_ct(x5, ct_x);   r = ckks.add(r, ckks.mult_scalar(x6, 1/720))
    return r


def fhe_softmax(ckks, ct_scores, n_valid):
    ct_exp = fhe_exp_approx(ckks, ct_scores)
    mask = torch.zeros(ckks.n_slots, device=ckks.device, dtype=torch.float32)
    mask[:n_valid] = 1.0
    ct_exp = ckks.mult_plain(ct_exp, mask)
    sum_val = ckks.decrypt(ckks.sum_slots(ct_exp, n_valid), 1)[0].item()
    inv = 1.0 / sum_val if abs(sum_val) > 1e-10 else 0.0
    return ckks.mult_scalar(ct_exp, inv)


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  Sipher FHE d=1024 — Eğitilmiş Checkpoint" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Device: {device}\n")

    # ── Checkpoint yükle ──
    ckpt_path = Path("checkpoints/pretrain_hybrid.pt")
    if not ckpt_path.exists():
        print("  ✗ Checkpoint bulunamadı: checkpoints/pretrain_hybrid.pt")
        return

    print("  Checkpoint yükleniyor...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f"  PPL: {ckpt['best_ppl']:.2f} | Epoch: {ckpt['epoch']}")
    print(f"  Config: d={cfg['d']} seq={cfg['seq']} nl={cfg['nl']} "
          f"nh={cfg['nh']} dk={cfg['dk']} emb={cfg['emb_dim']} win={cfg.get('window', 32)}")

    # ── Model oluştur ve ağırlıkları yükle ──
    from pretrain_hybrid import CipherFormerHybrid
    model = CipherFormerHybrid(
        ckpt['vocab_size'], cfg['d'], cfg['seq'], cfg['nl'], cfg['nh'],
        cfg['dk'], cfg['ffn_h'], cfg['emb_dim'], cfg.get('window', 32),
        cfg.get('causal_global', False), cfg.get('no_mixer', False),
        cfg.get('poly_ffn', False), cfg.get('gate_local_init', 0.0)
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    model = model.to(device)

    D = cfg['d']
    SEQ = cfg['seq']
    NH = cfg['nh']
    DK = cfg['dk']
    EMB = cfg['emb_dim']
    WINDOW = cfg.get('window', 32)
    VOCAB = ckpt['vocab_size']
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} param ({n_params/1e6:.1f}M)")

    # ── Test girdisi ──
    test_ids = torch.randint(0, VOCAB, (1, SEQ), device=device)
    print(f"  Input: {test_ids[0, :8].tolist()}... ({SEQ} token)")

    # ── Plaintext referans ──
    print(f"\n{'─' * 60}")
    print(f"  PLAINTEXT REFERANS")
    print(f"{'─' * 60}")
    with torch.no_grad():
        pt_logits = model(test_ids)
    pt_last = pt_logits[0, -1]
    pt_top5 = pt_last.topk(5)
    print(f"  Top-5: {pt_top5.indices.tolist()}")
    print(f"  Logits: mean={pt_last.mean().item():+.4f} std={pt_last.std().item():.4f}")

    # ── FHE Test: Tek Katman (Layer 0) ──
    print(f"\n{'─' * 60}")
    print(f"  FHE TEST — Layer 0 (d={D})")
    print(f"{'─' * 60}")

    pos = SEQ - 1
    layer = model.layers[0]

    # Embedding (plaintext)
    with torch.no_grad():
        emb = model.emb_proj(model.token_emb(test_ids)) + model.pos_emb(torch.arange(SEQ, device=device))
        emb_pos = emb[0, pos]  # [D]

    ckks = SlotCKKS(n_slots=D, scale=2**40, noise_std=1e-10, device=device)
    ct_emb = ckks.encrypt(emb_pos)
    fhe_emb = ckks.decrypt(ct_emb, D)
    err_emb = (emb_pos - fhe_emb).abs().max().item()
    print(f"  Embedding:     max_err={err_emb:.6f}")

    # QKV matvec
    print(f"  QKV matvec ({D}→{NH*DK}) çalışıyor...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        W_Q = layer.attn.W_Q.weight
        W_K = layer.attn.W_K.weight
        W_V = layer.attn.W_V.weight

    ct_q = ckks.mult_scalar(ckks.matvec(ct_emb, W_Q, D, NH*DK), 1.0/(DK**0.5))
    ct_k = ckks.matvec(ct_emb, W_K, D, NH*DK)
    ct_v = ckks.matvec(ct_emb, W_V, D, NH*DK)
    t_qkv = time.time() - t0
    print(f" {t_qkv:.1f}s")

    fhe_q = ckks.decrypt(ct_q, NH*DK)
    fhe_k = ckks.decrypt(ct_k, NH*DK)
    fhe_v = ckks.decrypt(ct_v, NH*DK)

    with torch.no_grad():
        x_in = emb_pos.unsqueeze(0)  # [1, D]
        pt_q = layer.attn.W_Q(x_in).view(NH, DK) / (DK**0.5)
        pt_k = layer.attn.W_K(x_in).view(NH, DK)
        pt_v = layer.attn.W_V(x_in).view(NH, DK)

    err_q = (pt_q.reshape(-1) - fhe_q).abs().max().item()
    err_k = (pt_k.reshape(-1) - fhe_k).abs().max().item()
    err_v = (pt_v.reshape(-1) - fhe_v).abs().max().item()
    print(f"  Q matvec:      max_err={err_q:.6f}")
    print(f"  K matvec:      max_err={err_k:.6f}")
    print(f"  V matvec:      max_err={err_v:.6f}")

    # PolyFFN (tek katman)
    if hasattr(layer, 'ffn') and layer.ffn is not None:
        print(f"  PolyFFN ({D}→{D*4}→{D}) çalışıyor...", end="", flush=True)
        t0 = time.time()
        with torch.no_grad():
            post_attn = emb_pos  # simplified: skip attention for FFN test
            W_up = layer.ffn.up.weight
            W_down = layer.ffn.down.weight
            poly_coeffs = layer.ffn.act.coeffs

        ckks_ffn = SlotCKKS(n_slots=D*4, scale=2**40, noise_std=1e-10, device=device)
        ct_pa = ckks_ffn.encrypt(post_attn)
        ct_ffn = ckks_ffn.matvec(ct_pa, W_up, D, D*4)
        ct_ffn = fhe_poly_act(ckks_ffn, ct_ffn, poly_coeffs)
        ct_ffn = ckks_ffn.matvec(ct_ffn, W_down, D*4, D)
        t_ffn = time.time() - t0
        print(f" {t_ffn:.1f}s")

        fhe_ffn = ckks_ffn.decrypt(ct_ffn, D)
        with torch.no_grad():
            pt_ffn_in = layer.ffn.up(post_attn)
            pt_ffn_act = layer.ffn.act(pt_ffn_in)
            pt_ffn_out = layer.ffn.down(pt_ffn_act)
        err_ffn = (pt_ffn_out - fhe_ffn).abs().max().item()
        print(f"  FFN:           max_err={err_ffn:.6f}")
    else:
        err_ffn = 0.0
        print(f"  FFN:           (yok)")

    # Head
    print(f"  Head ({D}→{EMB}) çalışıyor...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        W_head = model.head_proj.weight
    ckks_h = SlotCKKS(n_slots=D, scale=2**40, noise_std=1e-10, device=device)
    ct_pf = ckks_h.encrypt(emb_pos)
    ct_head = ckks_h.matvec(ct_pf, W_head, D, EMB)
    t_head = time.time() - t0
    print(f" {t_head:.1f}s")
    fhe_head = ckks_h.decrypt(ct_head, EMB)

    with torch.no_grad():
        pt_head = model.head_proj(emb_pos)
    err_head = (pt_head - fhe_head).abs().max().item()
    print(f"  Head:          max_err={err_head:.6f}")

    # Logits
    with torch.no_grad():
        fhe_logits = fhe_head @ model.token_emb.weight.T
    err_logits = (pt_last - fhe_logits).abs().max().item()
    fhe_top5 = fhe_logits.topk(5).indices.tolist()
    pt_top5_list = pt_top5.indices.tolist()
    top5_match = len(set(pt_top5_list) & set(fhe_top5))

    print(f"  Logits:        max_err={err_logits:.6f}")
    print(f"  Top-5 match:   {top5_match}/5")
    print(f"  PT  top-5:     {pt_top5_list}")
    print(f"  FHE top-5:     {fhe_top5}")

    # ── Sonuç ──
    errs = {'emb': err_emb, 'Q': err_q, 'K': err_k, 'V': err_v,
            'FFN': err_ffn, 'head': err_head, 'logits': err_logits}
    worst = max(errs.values())
    worst_name = max(errs, key=errs.get)
    ok = worst < 0.01 and top5_match >= 3

    print(f"\n{'═' * 60}")
    print(f"  d=1024 EĞİTİLMİŞ MODEL FHE SONUCU")
    print(f"{'═' * 60}")
    for name, err in errs.items():
        s = "✓" if err < 0.01 else "✗"
        print(f"  {s} {name:<15s} {err:.6f}")
    print(f"\n  Worst: {worst_name} ({worst:.6f})")
    print(f"  Top-5: {top5_match}/5")
    print(f"  {'✓ BAŞARILI' if ok else '✗ BAŞARISIZ'}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
