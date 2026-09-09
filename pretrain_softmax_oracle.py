"""
Sipher — Softmax Oracle (local 8GB)
===================================
Ayrı branch: klasik causal multi-head softmax attention.
Amaç: Aynı 16K TR veri ile "softmax cümle kurar mı?" üst tavan testi.
FHE-native DEĞİL — sadece kalite oracle.

Defaults (8GB güvenli):
  d=512, 10L, 8 head, seq=256, ~50-80M param
  bs=4, accum=4 → eff_bs=16
  AMP + optional grad checkpoint

Generate: PAD-FREE (variable length) — sol-pad YOK.

Run (önerilen local):
  .venv312/bin/python3 -u pretrain_softmax_oracle.py --from-scratch --epochs 5

Daha fazla veri (RAM yeterse):
  .venv312/bin/python3 -u pretrain_softmax_oracle.py --from-scratch --epochs 3 \\
    --data data/pretrain_tokens_curated16k_clean.pt:2,data/pretrain_tokens_wiki16k.pt:1,data/pretrain_tokens_oscar16k_clean.pt:1 \\
    --batches-per-epoch 8000

Sadece generation:
  .venv312/bin/python3 -u pretrain_softmax_oracle.py --gen-only \\
    --resume checkpoints/softmax_oracle_best.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SIPHER_TOKENIZER_PATH = Path("data/sipher_tokenizer/sipher_tokenizer.json")
CKPT_DIR = Path("checkpoints")
DEFAULT_DATA = (
    "data/pretrain_tokens_curated16k_clean.pt:2,"
    "data/pretrain_tokens_wiki16k_cut15m.pt:1,"
    "data/pretrain_tokens_sipher16k.pt:1"
)

PROMPTS = [
    "Merkez Bankası Başkanı bugün yaptığı açıklamada",
    "Borsa İstanbul da işlem gören hisseler",
    "Türkiye de enflasyon oranı geçen yıla göre",
    "Yatırım yaparken dikkat edilmesi gereken en önemli",
    "Kredi faiz oranları son dönemde",
    "Bankacılık sektöründe dijital dönüşüm",
    "Bu kitap",
]


class TokWrapper:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self._t = Tokenizer.from_file(str(path))
        self.vocab_size = self._t.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._t.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, dk = self.n_heads, self.d_k
        qkv = self.W_qkv(x).view(B, S, 3, H, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        # causal mask
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        h = d_model * ffn_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SoftmaxOracleLM(nn.Module):
    """Küçük causal GPT-like — FHE yok, kalite oracle."""

    def __init__(
        self,
        vocab: int,
        d: int = 512,
        seq: int = 256,
        nl: int = 10,
        nh: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d
        self.seq_len = seq
        self.vocab_size = vocab
        self.n_layers = nl
        self.token_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq, d)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(d, nh, ffn_mult, dropout) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # weight tying
        self.head.weight = self.token_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, S = ids.shape
        if S > self.seq_len:
            ids = ids[:, -self.seq_len :]
            S = ids.shape[1]
        pos = torch.arange(S, device=ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new: int = 48,
        temp: float = 0.7,
        rep: float = 1.3,
        top_k: int = 40,
        greedy: bool = False,
    ) -> list[int]:
        """PAD-FREE generation: variable length, no left-pad zeros."""
        self.eval()
        device = next(self.parameters()).device
        tokens = list(prompt_ids)
        for _ in range(max_new):
            inp = tokens[-self.seq_len :]
            x = torch.tensor([inp], device=device, dtype=torch.long)
            logits = self.forward(x)[0, -1].float()
            # light repetition penalty
            if rep != 1.0 and tokens:
                for t in set(tokens):
                    if 0 <= t < logits.numel():
                        if logits[t] > 0:
                            logits[t] /= rep
                        else:
                            logits[t] *= rep
            if greedy or temp <= 0:
                nxt = int(logits.argmax())
            else:
                if top_k and top_k < logits.numel():
                    v, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < v[-1], float("-inf"))
                probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            tokens.append(nxt)
        return tokens


def load_data_sources(spec: str) -> list[tuple[torch.Tensor, float]]:
    sources = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            p, w = entry.rsplit(":", 1)
            try:
                w = float(w)
            except ValueError:
                p, w = entry, 1.0
        else:
            p, w = entry, 1.0
        path = Path(p)
        if not path.exists():
            print(f"  ⚠ atlandı (yok): {path}", flush=True)
            continue
        tok = torch.load(path, weights_only=True, map_location="cpu").long()
        sources.append((tok, w))
        print(f"  Data: {path.name} | {len(tok):,} token | w={w}", flush=True)
    if not sources:
        raise SystemExit("Veri yok — --data yollarını kontrol et")
    total = sum(len(t) for t, _ in sources)
    print(f"  TOPLAM: {total:,} token | {len(sources)} kaynak", flush=True)
    return sources


def sample_batch(
    sources: list[tuple[torch.Tensor, float]],
    batch: int,
    seq: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.tensor([w for _, w in sources], dtype=torch.float)
    weights = weights / weights.sum()
    xs, ys = [], []
    for _ in range(batch):
        si = int(torch.multinomial(weights, 1))
        data = sources[si][0]
        hi = len(data) - seq - 1
        if hi <= 0:
            raise SystemExit(f"Kaynak {si} çok kısa: {len(data)} < seq+1")
        i = random.randint(0, hi)
        chunk = data[i : i + seq + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.stack(ys).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def run_generation(
    model: SoftmaxOracleLM,
    tok: TokWrapper,
    label: str = "",
    greedy: bool = False,
    max_new: int = 48,
) -> None:
    model.eval()
    print(f"\n{'═' * 64}")
    print(f"  SOFTMAX ORACLE GEN {label} | pad-free | greedy={greedy}")
    print(f"{'═' * 64}", flush=True)
    for p in PROMPTS:
        ids = tok.encode(p)
        gen = model.generate(ids, max_new=max_new, temp=0.7, rep=1.5, top_k=40, greedy=greedy)
        text = tok.decode(gen)
        print(f'\n  "{p}"')
        print(f"  → {text[:320]}")
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Softmax oracle LM (local 8GB)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4, help="eff_bs = batch * accum")
    parser.add_argument("--batches-per-epoch", type=int, default=2000)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--nl", type=int, default=10)
    parser.add_argument("--nh", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=str, default=str(SIPHER_TOKENIZER_PATH))
    parser.add_argument("--resume", type=str, default="checkpoints/softmax_oracle_best.pt")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--gen-only", action="store_true")
    parser.add_argument("--greedy-gen", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("╔" + "═" * 66 + "╗")
    print("║  Softmax Oracle LM — local quality upper bound" + " " * 18 + "║")
    print("╚" + "═" * 66 + "╝", flush=True)
    print(f"  Device: {device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    tok = TokWrapper(args.tokenizer)
    vocab = tok.vocab_size
    print(f"  Tokenizer: {Path(args.tokenizer).name} | vocab={vocab}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    if not args.from_scratch and Path(args.resume).exists() and not args.gen_only:
        # gen-only also loads below; train resume
        pass

    model = SoftmaxOracleLM(
        vocab, d=args.d, seq=args.seq, nl=args.nl, nh=args.nh,
        ffn_mult=args.ffn_mult, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} ({n_params / 1e6:.1f}M) | "
          f"d={args.d} {args.nl}L nh={args.nh} seq={args.seq}", flush=True)

    start_ep = 0
    best_ppl = float("inf")
    if (not args.from_scratch) and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        start_ep = int(ckpt.get("epoch", 0))
        best_ppl = float(ckpt.get("best_ppl", best_ppl))
        print(f"  ✓ Resume {args.resume} | ep={start_ep} | best_ppl={best_ppl:.2f}", flush=True)
        if args.gen_only:
            model.to(device)
            run_generation(model, tok, f"— {args.resume}", greedy=args.greedy_gen)
            return
    elif args.gen_only:
        raise SystemExit("--gen-only için geçerli --resume ckpt lazım")

    sources = load_data_sources(args.data)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    # optimizer steps per epoch = batches_per_epoch (after accum)
    steps_per_ep = args.batches_per_epoch
    total_steps = max(steps_per_ep * args.epochs, 1)
    warmup = args.warmup

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n  Train: {args.epochs} ep | bs={args.batch}×accum={args.accum} "
          f"eff={args.batch * args.accum} | lr={args.lr} | "
          f"{steps_per_ep} opt-steps/ep", flush=True)

    global_step = 0
    for ep in range(start_ep, start_ep + args.epochs):
        model.train()
        tl, nb = 0.0, 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        # microbatches: steps_per_ep * accum
        n_micro = steps_per_ep * args.accum
        for micro in range(n_micro):
            x, y = sample_batch(sources, args.batch, args.seq, device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = crit(logits.view(-1, vocab), y.view(-1)) / args.accum
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = crit(logits.view(-1, vocab), y.view(-1)) / args.accum
                loss.backward()

            if (micro + 1) % args.accum == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

            tl += loss.item() * args.accum
            nb += 1

            if (micro + 1) % (args.accum * 200) == 0:
                avg = tl / max(nb, 1)
                ppl_r = math.exp(min(avg, 20))
                print(f"    step {global_step}: loss={avg:.3f} ppl~{ppl_r:.1f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = tl / max(nb, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {ep + 1}: PPL={ppl:.2f} loss={avg_loss:.4f} ({elapsed:.0f}s)", flush=True)

        # save best + always last
        state = {
            "model": model.state_dict(),
            "epoch": ep + 1,
            "best_ppl": min(best_ppl, ppl),
            "ppl": ppl,
            "vocab_size": vocab,
            "config": {
                "d": args.d, "seq": args.seq, "nl": args.nl, "nh": args.nh,
                "ffn_mult": args.ffn_mult, "arch": "softmax_oracle_gelu",
            },
        }
        torch.save(state, CKPT_DIR / "softmax_oracle_last.pt")
        if ppl < best_ppl:
            best_ppl = ppl
            state["best_ppl"] = best_ppl
            torch.save(state, CKPT_DIR / "softmax_oracle_best.pt")
            print(f"    ✓ best → checkpoints/softmax_oracle_best.pt (PPL={best_ppl:.2f})", flush=True)

        run_generation(model, tok, f"— ep{ep + 1} PPL={ppl:.1f}", greedy=args.greedy_gen)

    print(f"\n✓ Softmax oracle bitti | best_ppl={best_ppl:.2f}", flush=True)
    print("  ckpt: checkpoints/softmax_oracle_best.pt | last: softmax_oracle_last.pt", flush=True)


if __name__ == "__main__":
    main()
