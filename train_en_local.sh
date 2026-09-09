#!/usr/bin/env bash
# İngilizce Sipher eğitimi — YEREL (RTX 4060, 8GB) — no-mixer reçetesi (FHE engine ile birebir)
# Kullanım: bash train_en_local.sh [epochs]
set -e
cd "$(dirname "$0")"
PY=.venv312/bin/python3
EPOCHS="${1:-3}"

# 1) Veri hazır (wiki103.txt yoksa indir + tokenize)
if [ ! -f data/pretrain_tokens_en16k.pt ]; then
    if [ ! -f wiki103.txt ]; then
        echo "== WikiText-103 indiriliyor (541MB)..."
        $PY -c "
from datasets import load_dataset
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
text = '\n'.join(ds['text'])
open('wiki103.txt','w').write(text)
print('words:', len(text.split()))
"
    fi
    echo "== Tokenizer + tokenizasyon (16K ByteLevel BPE)..."
    $PY en_prep.py --corpus wiki103.txt --vocab-size 16000 \
        --out-tok data/sipher_tokenizer_en --out-tokens data/pretrain_tokens_en16k.pt
fi

# 2) Eğitim — no-mixer (SIZINTI YOK) + feat_degree 2 + gate_local 0.5
#    bs=8 (8GB VRAM) | epoch ~2-2.5 saat (4060)
mkdir -p checkpoints
$PY pretrain_hybrid.py --from-scratch \
    --d 512 --nl 8 --ffn-h 2048 --batch 8 --accum 1 \
    --epochs "$EPOCHS" --lr 3e-4 --window 32 \
    --no-mixer --poly-ffn --feat-degree 2 \
    --gate-local-init 0.5 --gate-global-init 0.0 --freeze-gate-global \
    --tokenizer data/sipher_tokenizer_en/sipher_tokenizer_en.json \
    --data data/pretrain_tokens_en16k.pt \
    --batches-per-epoch 29000 --resume checkpoints/pretrain_hybrid.pt

echo "== TAMAM — checkpoint: checkpoints/pretrain_hybrid.pt"
echo "Sonraki: export → cpp/weights_en (export_weights.py --feat-degree 2) → FHE parity/üretim"
