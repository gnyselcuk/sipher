#!/usr/bin/env python3
"""
İngilizce Sipher pipeline — tokenizer eğitimi + veri tokenizasyonu.
WikiText-103 üzerinde 16K ByteLevel BPE → data/pretrain_tokens_en16k.pt + tokenizer JSON.

Kullanım:
  python3 en_prep.py --corpus /root/fhe/wiki103.txt --vocab-size 16000 \
      --out-tok data/sipher_tokenizer_en --out-tokens data/pretrain_tokens_en16k.pt
"""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Ham metin dosyası")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--out-tok", default="data/sipher_tokenizer_en", help="Tokenizer JSON dizini")
    ap.add_argument("--out-tokens", default="data/pretrain_tokens_en16k.pt")
    ap.add_argument("--max-tokens", type=int, default=0, help="Sınırla (0 = hepsi)")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_tok_dir = Path(args.out_tok)
    out_tok_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Tokenizer eğitimi (ByteLevel BPE — İngilizce için ideal) ──
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<unk>"],
        min_frequency=2,
        show_progress=True,
    )
    tok.train([str(corpus)], trainer)

    tok_path = out_tok_dir / "sipher_tokenizer_en.json"
    tok.save(str(tok_path))
    print(f"✓ Tokenizer: {tok_path} (vocab={tok.get_vocab_size()})")

    # ── 2. Corpus tokenizasyonu → 1D tensor ──
    all_ids = []
    batch = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= 10000:
                enc = tok.encode_batch(batch)
                for e in enc:
                    all_ids.extend(e.ids)
                batch = []
    if batch:
        enc = tok.encode_batch(batch)
        for e in enc:
            all_ids.extend(e.ids)

    if args.max_tokens > 0:
        all_ids = all_ids[:args.max_tokens]

    t = torch.tensor(all_ids, dtype=torch.long)
    torch.save(t, args.out_tokens)
    print(f"✓ Tokenlar: {args.out_tokens} ({len(t):,} token)")


if __name__ == "__main__":
    main()
