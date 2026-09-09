"""
CipherFormer — Hybrid Attention (Global Linear + Local Causal)
===============================================================
Key finding: linear attention learns vocabulary but NOT grammar.
Root cause: Q·(K^T·V) compresses all context into a fixed D×D matrix,
            losing sequential order information.

Solution: Add local causal dot-product attention on a sliding window.
  - Global linear attention: Q·(K^T·V) → vocabulary, semantics (unchanged)
  - Local causal attention: softmax(Q·K^T/√dk)·V → grammar, word order (NEW)
  - Learned gates combine both paths (gate_local starts at 0)

FHE compatibility:
  - Global path: matmul, polynomial (unchanged)
  - Local path: softmax → polynomial approximation in deployment
  - Window W=32: small 32×32 matrices, cheap in FHE

Run (fine-tune from pre-trained 100M):
  .venv312/bin/python3 -u pretrain_hybrid.py --epochs 5

Run (from scratch):
  .venv312/bin/python3 -u pretrain_hybrid.py --epochs 20 --from-scratch

Run (generation only, no training):
  .venv312/bin/python3 -u pretrain_hybrid.py --gen-only
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
import sys
import json
import random
import argparse
from pathlib import Path

TOKENIZER_NAME = "dbmdz/bert-base-turkish-cased"
SIPHER_TOKENIZER_PATH = Path("data/sipher_tokenizer/sipher_tokenizer.json")
BERT_EMB_PATH = Path("checkpoints/bert_embedding_128.pt")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


class _TokenShim:
    """tokenizers lib → HF-benzeri API uyumlayıcı (vocab_size/encode/decode)."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self._t = Tokenizer.from_file(str(path))

    @property
    def vocab_size(self):
        return self._t.get_vocab_size()

    def encode(self, text, add_special_tokens=True):
        return self._t.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self._t.decode(ids, skip_special_tokens=skip_special_tokens)


def _get_tokenizer(path):
    """Sipher 16K BPE tokenizer'ı lazy yükle (tokenizers lib — transformers bağımlılığı yok)."""
    return _TokenShim(path)


def repair_cfg_d(cfg, sd):
    """Eski checkpoint'lerde config['d'] yanlışlıkla token tensörü olarak kaydedilmiş
    olabilir (çoklu-veri döngüsü gölgeleme bug'ı). Gerçek d'yi state_dict'ten kurtar.
    Not: token_emb (vocab, emb_dim) — asıl model boyutu emb_proj.weight.shape[0]."""
    if not isinstance(cfg.get("d"), int):
        real_d = int(sd["emb_proj.weight"].shape[0])
        print(f"  ⚠ config['d'] bozuk (tensor) → {real_d} olarak onarıldı")
        cfg["d"] = real_d
    return cfg


class PolyAct(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor([0.0, 0.5, 0.2][:degree+1]))
        self.degree = degree
    def forward(self, x):
        powers = torch.stack([x.pow(i) for i in range(self.degree+1)], dim=0)
        return (powers.view(self.degree+1, -1).T @ self.coeffs).view(x.shape)


class TokenMixer(nn.Module):
    def __init__(self, d, seq, h):
        super().__init__()
        self.token_up = nn.Linear(seq, h)
        self.token_act = PolyAct()
        self.token_down = nn.Linear(h, seq)
    def forward(self, x):
        res = x
        x = x.transpose(1, 2)
        x = self.token_act(self.token_up(x))
        x = self.token_down(x).transpose(1, 2) + res
        return x


class PolyFFN(nn.Module):
    """Channel-wise FFN with polynomial activation. FHE-native.
    Per-position (no sequence mixing, no bidirectional leak)."""
    def __init__(self, d, mult=4):
        super().__init__()
        self.up = nn.Linear(d, d * mult)
        self.act = PolyAct()
        self.down = nn.Linear(d * mult, d)
    def forward(self, x):
        return self.down(self.act(self.up(x)))


class HybridAttention(nn.Module):
    """Global linear attention + Local causal window attention.

    Global: Q·(K^T·V) — vocabulary/semantics (bidirectional or causal)
    Local:  softmax(Q·K^T/√dk)·V with causal window — grammar/word order

    causal_global=True: uses cumulative KV sum so each position only
    attends to past tokens.  Forces the model to learn word order
    during training, matching autoregressive generation.

    FHE: global = matmul (+ cumsum if causal); local softmax → poly approx.
    """
    def __init__(self, d_model, n_heads, d_k, seq_len, window=32, causal_global=False, gate_local_init=0.0, feat_degree=2, gate_global_init=0.0):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_k
        self.scale = d_k ** 0.5
        self.window = window
        self.causal_global = causal_global
        self.feat_degree = feat_degree   # feature-map derecesi (2: φ(x)=[x; x²])
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.gate_global = nn.Parameter(torch.tensor(gate_global_init))
        self.gate_local = nn.Parameter(torch.tensor(gate_local_init))
        idx = torch.arange(seq_len)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        self.register_buffer("attn_mask", (diff >= 0) & (diff < window))
        nn.init.xavier_uniform_(self.W_Q.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_K.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_V.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_O.weight, gain=0.02)

    def _feat(self, x):
        """Polinom feature map: φ(x) = [x; x²] (derece 2) — FHE engine ile birebir."""
        if self.feat_degree < 2:
            return x, None
        return x, x * x

    def forward(self, x):
        B, S, D = x.shape
        H, dk = self.n_heads, self.d_k
        x_f = x.float()
        Q = self.W_Q(x_f).view(B, S, H, dk).transpose(1, 2) / self.scale
        K = self.W_K(x_f).view(B, S, H, dk).transpose(1, 2)
        V = self.W_V(x_f).view(B, S, H, dk).transpose(1, 2)
        Q2 = Q * Q if self.feat_degree >= 2 else None
        K2 = K * K if self.feat_degree >= 2 else None

        # Global: linear attention
        if self.causal_global:
            # Causal: each position sees only past tokens
            # KV_cum[i] = sum_{j<=i} k_j ⊗ v_j
            KV = K.unsqueeze(-1) @ V.unsqueeze(-2)       # B,H,S,dk,dk
            KV_cum = torch.cumsum(KV, dim=2)              # B,H,S,dk,dk
            global_out = (Q.unsqueeze(-2) @ KV_cum).squeeze(-2)  # B,H,S,dk
            # Feature-map derece 2: + (Σ V⊗K²) @ Q²  (FHE engine ile birebir)
            if self.feat_degree >= 2:
                KV2 = K2.unsqueeze(-1) @ V.unsqueeze(-2)
                KV2_cum = torch.cumsum(KV2, dim=2)
                global_out = global_out + (Q2.unsqueeze(-2) @ KV2_cum).squeeze(-2)
        else:
            # Bidirectional (original)
            KV = K.transpose(-2, -1) @ V                  # B,H,dk,dk
            global_out = Q @ KV                           # B,H,S,dk

        # Local: causal window attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores.masked_fill(~self.attn_mask[:S, :S], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        local_out = attn @ V

        out = self.gate_global * global_out + self.gate_local * local_out
        out = out.transpose(1, 2).contiguous().view(B, S, H * dk)
        return self.W_O(out).to(x.dtype)


class CipherFormerBlock(nn.Module):
    def __init__(self, d, seq, nh, dk, ffn_h, window, causal_global=False, no_mixer=False, poly_ffn=False, gate_local_init=0.0, feat_degree=2, gate_global_init=0.0):
        super().__init__()
        self.mixer = None if no_mixer else TokenMixer(d, seq, ffn_h)
        self.attn = HybridAttention(d, nh, dk, seq, window, causal_global, gate_local_init, feat_degree, gate_global_init)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = PolyFFN(d) if poly_ffn else None
        self.norm2 = nn.LayerNorm(d) if poly_ffn else None
    def forward(self, x):
        if self.mixer is not None:
            x = self.mixer(x)
        x = x + self.attn(self.norm1(x))
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x


class CipherFormerHybrid(nn.Module):
    def __init__(self, vocab, d, seq, nl, nh, dk, ffn_h, emb_dim=128, window=32, causal_global=False, no_mixer=False, poly_ffn=False, gate_local_init=0.0, feat_degree=2, gate_global_init=0.0):
        super().__init__()
        self.d_model, self.seq_len, self.vocab_size, self.n_layers = d, seq, vocab, nl
        self.emb_dim = emb_dim
        self.feat_degree = feat_degree
        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, d, bias=False)
        self.pos_emb = nn.Embedding(seq, d)
        self.layers = nn.ModuleList([
            CipherFormerBlock(d, seq, nh, dk, ffn_h, window, causal_global, no_mixer, poly_ffn, gate_local_init, feat_degree, gate_global_init) for _ in range(nl)
        ])
        self.head_proj = nn.Linear(d, emb_dim, bias=False)

    def forward(self, ids):
        B, S = ids.shape
        x = self.emb_proj(self.token_emb(ids)) + self.pos_emb(
            torch.arange(S, device=ids.device).unsqueeze(0)
        )
        for l in self.layers:
            x = l(x)
        h = self.head_proj(x)
        return h @ self.token_emb.weight.T

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=120, temp=0.7, rep=2.0, top_k=40):
        self.eval()
        device = next(self.parameters()).device
        tokens = list(prompt_ids)
        base_pad = list(prompt_ids)   # sol-pad: 0 yerine prompt tekrarı (model eğitimde pad görmedi)
        for _ in range(max_new):
            inp = tokens[-self.seq_len:]
            if len(inp) < self.seq_len:
                need = self.seq_len - len(inp)
                if base_pad:
                    inp = (base_pad * (need // len(base_pad) + 1))[-need:] + inp
                else:
                    inp = [0] * need + inp
            logits = self.forward(torch.tensor([inp], device=device))[0, -1]
            for t in set(tokens):
                if logits[t] > 0:
                    logits[t] /= rep
                else:
                    logits[t] *= rep
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits / temp, dim=-1)
            tokens.append(torch.multinomial(probs, 1).item())
        return tokens


def load_wiki_context(tokenizer, n_articles=200):
    ctx_ids = []
    wiki_path = Path("data/corpus_tr/wiki_tr_full.jsonl")
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_articles:
                    break
                text = json.loads(line)["text"][:500]
                ctx_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ctx_ids


def run_generation(model, tokenizer, label=""):
    prompts = [
        "Merkez Bankası Başkanı bugün yaptığı açıklamada",
        "Borsa İstanbul da işlem gören hisseler",
        "Türkiye de enflasyon oranı geçen yıla göre",
        "Yatırım yaparken dikkat edilmesi gereken en önemli",
        "Kredi faiz oranları son dönemde",
        "Bankacılık sektöründe dijital dönüşüm",
        "Altın fiyatları küresel piyasalarda",
    ]
    print(f"\n{'═' * 70}")
    print(f"  GENERATION {label}")
    print(f"{'═' * 70}", flush=True)
    for p in prompts:
        enc = tokenizer.encode(p, add_special_tokens=False)
        gen = model.generate(enc, max_new=100)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f'\n  "{p}"')
        print(f"  → {text[:350]}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--resume", type=str, default="checkpoints/pretrain_100m.pt")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--gen-only", action="store_true")
    parser.add_argument("--causal-global", action="store_true",
                        help="Make global linear attention causal (cumulative KV)")
    parser.add_argument("--no-mixer", action="store_true",
                        help="Remove TokenMixer (eliminates bidirectional leak)")
    parser.add_argument("--poly-ffn", action="store_true",
                        help="Add channel-wise PolyFFN (d→4d→d) after attention")
    parser.add_argument("--gate-local-init", type=float, default=0.0,
                        help="Initial value for gate_local (default 0)")
    parser.add_argument("--gate-global-init", type=float, default=0.0,
                        help="Initial value for gate_global (default 0 — ölü dal; 0.2 önerilir feature-map ile)")
    parser.add_argument("--freeze-gate-global", action="store_true",
                        help="gate_global'i init değerinde DONDUR (global yol zorla kullanılır — rekabeti test etmek için)")
    parser.add_argument("--feat-degree", type=int, default=2,
                        help="Feature-map derecesi (2: φ=[x;x²]; 1: kapalı — Kaggle checkpoint'leri için 1)")
    parser.add_argument("--d", type=int, default=1024, help="Model dim (küçük deneyler için: 512)")
    parser.add_argument("--nl", type=int, default=20, help="Katman sayısı (küçük deneyler için: 8)")
    parser.add_argument("--ffn-h", type=int, default=None, help="FFN hidden (default: d*4)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (yerel 8GB için BS=1 önerilir)")
    parser.add_argument("--accum", type=int, default=1,
                        help="Gradient accumulation adımı (eff_bs = batch*accum)")
    parser.add_argument("--tokenizer", type=str, default=str(SIPHER_TOKENIZER_PATH),
                        help="Tokenizer JSON yolu (varsayılan: Sipher 16K)")
    parser.add_argument("--data", type=str, default="data/pretrain_tokens_sipher16k.pt",
                        help="Eğitim token dosyası(ları): 'yol1:w1,yol2:w2' (ağırlıklı çoklu kaynak)")
    parser.add_argument("--batches-per-epoch", type=int, default=0,
                        help="Epoch başına batch (0 = auto: 2000/4000). Büyük veri için 30000+ önerilir")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("╔" + "═" * 68 + "╗")
    print("║  CipherFormer — Hybrid Attention (Linear + Local Causal)" + " " * 11 + "║")
    print("╚" + "═" * 68 + "╝", flush=True)

    tokenizer = _get_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"  Tokenizer: {Path(args.tokenizer).name} (vocab={vocab_size})")

    # ── Model config ──
    if not args.from_scratch and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = repair_cfg_d(ckpt["config"], ckpt["model"])
        d, seq, nl = cfg["d"], cfg["seq"], cfg["nl"]
        nh, dk, ffn_h, emb_dim = cfg["nh"], cfg["dk"], cfg.get("ffn_h", cfg["d"] * 4), cfg["emb_dim"]
    else:
        d, seq, nl = args.d, 256, args.nl
        dk = 64
        nh = max(1, d // dk)
        ffn_h = args.ffn_h or d * 4
        emb_dim = 128

    model = CipherFormerHybrid(
        vocab_size, d, seq, nl, nh, dk, ffn_h, emb_dim, args.window,
        args.causal_global, args.no_mixer, args.poly_ffn, args.gate_local_init,
        args.feat_degree,
        args.gate_global_init
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    causal_str = " CAUSAL" if args.causal_global else ""
    mixer_str = " NO-MIXER" if args.no_mixer else ""
    ffn_str = " POLY-FFN" if args.poly_ffn else ""
    gate_str = f" gate_local_init={args.gate_local_init}" if args.gate_local_init > 0 else ""
    print(f"  d={d} {nl}L nh={nh} dk={dk} seq={seq} emb={emb_dim} window={args.window}{causal_str}{mixer_str}{ffn_str}{gate_str}")

    # ── Load checkpoint ──
    resumed = False
    if not args.from_scratch and Path(args.resume).exists():
        old_sd = ckpt["model"]
        new_sd = {}
        for k, v in old_sd.items():
            if k.endswith(".attn.gate"):
                new_sd[k.replace(".attn.gate", ".attn.gate_global")] = v
            else:
                new_sd[k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        resumed = True
        print(f"  ✓ Resumed from {args.resume} (epoch {ckpt.get('epoch','?')}, PPL {ckpt.get('best_ppl','?')})")
        gate_g = model.layers[0].attn.gate_global.item()
        print(f"    gate_global={gate_g:.4f} (from pre-trained), gate_local=0.0000 (new)")
        if missing:
            expected_missing = [k for k in missing if "gate_local" in k or "attn_mask" in k]
            other = [k for k in missing if k not in expected_missing]
            if other:
                print(f"    ⚠ Unexpected missing keys: {other}")
    elif args.from_scratch and BERT_EMB_PATH.exists():
        bert_data = torch.load(BERT_EMB_PATH, weights_only=False)
        if bert_data["embedding"].shape[0] == vocab_size:
            with torch.no_grad():
                model.token_emb.weight.copy_(bert_data["embedding"])
            print(f"  ✓ BERT embedding init (variance: {bert_data['variance_kept']:.1%})")
        else:
            print(f"  ⚠ BERT embedding (32K) vocab'la uyuşmuyor ({vocab_size}) — init atlandı")

    # ── Generation only ──
    if args.gen_only:
        run_generation(model, tokenizer, f"— {args.resume}")
        return

    # ── Data (çoklu kaynak: "yol:w,ağırlık,yol:w") ──
    data_sources = []
    for entry in args.data.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            p, w = entry.rsplit(":", 1)
            w = float(w)
        else:
            p, w = entry, 1.0
        cache = Path(p)
        if not cache.exists():
            print(f"  ⚠ Veri bulunamadı: {cache} — atlandı")
            continue
        tok_ids = torch.load(cache, weights_only=True).long()
        data_sources.append((tok_ids, w))
        print(f"  Data: {cache.name} | Tokens: {len(tok_ids):,} | ağırlık: {w}", flush=True)
    if not data_sources:
        raise SystemExit("Veri kaynağı yok!")
    total_tok = sum(len(d) for d, _ in data_sources)
    print(f"  TOPLAM: {total_tok:,} token | {len(data_sources)} kaynak", flush=True)
    data = data_sources[0][0]   # geriye uyumluluk (PPL/print için)

    # ── Baseline generation (before training) ──
    if resumed:
        run_generation(model, tokenizer, "— BASELINE (before hybrid training)")

    # ── Training ──
    batch_size = args.batch
    batches_per_epoch = args.batches_per_epoch if args.batches_per_epoch > 0 else (2000 if resumed else 4000)
    # ── Optimizer (gate_global dondurma desteği) ──
    if args.freeze_gate_global:
        n_frozen = 0
        for name, p in model.named_parameters():
            if name.endswith("attn.gate_global"):
                p.data.fill_(args.gate_global_init)   # init değerine set + dondur (local-only: 0.0)
                p.requires_grad = False
                n_frozen += 1
        print(f"  🔒 {n_frozen} gate_global parametresi donduruldu (sabit {args.gate_global_init})", flush=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05)
    total_steps = batches_per_epoch * args.epochs
    warmup = 200 if resumed else 1000

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    best_ppl = float("inf")

    print(f"\n  Training {args.epochs} epochs (bs={batch_size}, accum={args.accum}, "
          f"eff_bs={batch_size*args.accum}, lr={args.lr}, {batches_per_epoch} batches/ep)")
    print(f"  Warmup: {warmup} steps | Grad clip: 0.1", flush=True)

    for ep in range(args.epochs):
        model.train()
        tl, nb = 0, 0
        t0 = time.time()
        opt.zero_grad()
        for step in range(batches_per_epoch):
            # Ağırlıklı kaynak seçimi (çoklu veri)
            if len(data_sources) > 1:
                wsum = sum(w for _, w in data_sources)
                r = random.random() * wsum
                acc_w = 0
                src = data_sources[0][0]
                for td, w in data_sources:
                    acc_w += w
                    if r <= acc_w:
                        src = td
                        break
            else:
                src = data
            idx = torch.randint(0, len(src) - seq - 1, (batch_size,))
            x = torch.stack([src[i : i + seq] for i in idx]).to(device)
            y = torch.stack([src[i + 1 : i + seq + 1] for i in idx]).to(device)
            do_step = ((step + 1) % args.accum == 0) or (step == batches_per_epoch - 1)
            if scaler:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = crit(model(x).view(-1, vocab_size), y.view(-1)) / args.accum
                scaler.scale(loss).backward()
                if do_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            else:
                loss = crit(model(x).view(-1, vocab_size), y.view(-1)) / args.accum
                loss.backward()
                if do_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    opt.step()
                    opt.zero_grad()
            sched.step()
            tl += loss.item() * args.accum
            nb += 1

        ppl = math.exp(min(tl / nb, 10))
        elapsed = time.time() - t0
        gate_g = model.layers[0].attn.gate_global.item()
        gate_l = model.layers[0].attn.gate_local.item()
        gate_g_last = model.layers[-1].attn.gate_global.item()
        gate_l_last = model.layers[-1].attn.gate_local.item()
        print(
            f"    Epoch {ep+1}: PPL={ppl:.2f} ({elapsed:.0f}s) "
            f"gates L0=[g:{gate_g:.3f} l:{gate_l:.3f}] "
            f"L{nl-1}=[g:{gate_g_last:.3f} l:{gate_l_last:.3f}]",
            flush=True,
        )
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep + 1,
                    "best_ppl": best_ppl,
                    "vocab_size": vocab_size,
                    "config": {
                        "d": d, "seq": seq, "nl": nl, "nh": nh,
                        "dk": dk, "ffn_h": ffn_h, "emb_dim": emb_dim,
                        "window": args.window,
                        "causal_global": args.causal_global,
                        "no_mixer": args.no_mixer,
                        "poly_ffn": args.poly_ffn,
                        "gate_local_init": args.gate_local_init,
                        "feat_degree": args.feat_degree,
                    },
                },
                CKPT_DIR / "pretrain_hybrid.pt",
            )

    # ── Post-training generation ──
    run_generation(model, tokenizer, f"— AFTER HYBRID TRAINING (PPL={best_ppl:.2f})")

    print(f"{'═' * 70}")
    print(f"  Hybrid: PPL={best_ppl:.2f} | {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  d={d} {nl}L | window={args.window} | lr={args.lr}")
    print(f"  Checkpoint: checkpoints/pretrain_hybrid.pt")
    print(f"{'═' * 70}", flush=True)


if __name__ == "__main__":
    main()
