"""
Sipher Türkçe BPE Tokenizer Eğitimi
=====================================
16K vocab (default), sadece Türkçe karakterler.
TEMİZ corpus: oscar_clean (filtrelenmiş) + curated_clean (finans) + diğer segmentler.
Not: corpus .jsonl dosyaları boyut/lisans gereği repo'da YOK — bu script
tokenizer'ın NASIL eğitildiğini (alfabe, BPE config, normalizer) belgeler.

Kullanım:
  .venv/bin/python3 -u train_tokenizer.py                    # 16K → data/sipher_tokenizer
  .venv/bin/python3 -u train_tokenizer.py --vocab-size 32000 --out data/sipher_tokenizer32k
"""

import json
import argparse
import tempfile
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors

OUT_DIR = Path("data/sipher_tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Türkçe alfabe (küçük + büyük)
TR_ALPHABET = (
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "0123456789"
    " .,;:!?'-()[]/\"%@#&*+=<>~^_|\n"
)


def extract_texts_for_tokenizer():
    """Tüm kaynaklardan metin çıkar → geçici dosyaya yaz."""
    sources = [
        "data/literary_corpus/cleaned/oscar_clean.jsonl",       # filtrelenmiş OSCAR
        "data/literary_corpus/cleaned/curated_clean.jsonl",     # finans (küratör)
        "data/literary_corpus/cleaned/wiki_finance_segments.jsonl",
        "data/literary_corpus/cleaned/book_segments.jsonl",
        "data/literary_corpus/cleaned/mevzuat_segments.jsonl",
        "data/literary_corpus/cleaned/textbook_segments.jsonl",
        "data/literary_corpus/cleaned/regulatory_segments.jsonl",
    ]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    total_lines = 0

    for src in sources:
        p = Path(src)
        if not p.exists():
            continue
        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    seg = json.loads(line)
                    text = seg.get("text", "").strip()
                    if text and len(text) > 50:
                        tmp.write(text + "\n")
                        count += 1
                except:
                    pass
        print(f"  {p.name:40s} {count:>8,d} satır")
        total_lines += count

    tmp.close()
    print(f"\n  Toplam: {total_lines:,} satır → {tmp.name}")
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--out", type=str, default="data/sipher_tokenizer")
    args = parser.parse_args()
    global OUT_DIR
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  Sipher Türkçe BPE Tokenizer (TEMİZ corpus)" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Vocab boyutu: {args.vocab_size:,}", flush=True)

    # 1) Metinleri çıkar
    print(f"\n  Metinler çıkarılıyor...", flush=True)
    corpus_file = extract_texts_for_tokenizer()

    # 2) Tokenizer oluştur
    print(f"\n  Tokenizer eğitiliyor...", flush=True)

    tokenizer = Tokenizer(models.BPE())

    # Normalizer: NFKC + temizlik
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Strip(),
    ])

    # Pre-tokenizer: whitespace + punctuation bazlı bölme
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])

    # Trainer: Türkçe alfabe ile BPE
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=list(TR_ALPHABET),
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
        show_progress=True,
    )

    # Eğit
    tokenizer.train([corpus_file], trainer)

    # Post-processor: [CLS] ... [SEP] formatı
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    # 3) Kaydet
    tokenizer.save(str(OUT_DIR / "sipher_tokenizer.json"))

    # 4) Test
    print(f"\n{'═' * 60}")
    print(f"  TOKENIZER TEST")
    print(f"{'═' * 60}")
    print(f"  Vocab: {tokenizer.get_vocab_size():,}")

    test_sentences = [
        "Merkez Bankası faiz oranlarını değiştirdi.",
        "Borsa İstanbul'da işlem gören hisseler yükseldi.",
        "Kredi faiz oranları son dönemde artış gösterdi.",
        "Türkiye'de enflasyon oranı geçen yıla göre yükseldi.",
        "Bankacılık sektöründe dijital dönüşüm hızlanıyor.",
        "Yatırım yaparken dikkat edilmesi gereken en önemli faktör risk yönetimidir.",
    ]

    for sent in test_sentences:
        enc = tokenizer.encode(sent)
        tokens = enc.tokens
        print(f"\n  \"{sent}\"")
        print(f"  → {len(tokens)} token: {tokens}")

    # Eski (16K) tokenizer ile karşılaştır
    print(f"\n{'═' * 60}")
    print(f"  KARŞILAŞTIRMA: Eski (16K) vs Yeni ({args.vocab_size//1000}K)")
    print(f"{'═' * 60}")
    try:
        old_tok = Tokenizer.from_file("data/sipher_tokenizer/sipher_tokenizer.json")
        for sent in test_sentences:
            old_enc = old_tok.encode(sent).ids
            new_enc = tokenizer.encode(sent).ids
            print(f"\n  \"{sent}\"")
            print(f"  Eski (16K): {len(old_enc)} token | Yeni ({args.vocab_size//1000}K): {len(new_enc)} token | Oran: {len(new_enc)/len(old_enc):.2f}x")
    except Exception as e:
        print(f"  (karşılaştırma atlandı: {e})")

    # Temizle
    Path(corpus_file).unlink(missing_ok=True)

    print(f"\n  ✓ Kaydedildi: {OUT_DIR / 'sipher_tokenizer.json'}")
    print(f"  ✓ Vocab: {tokenizer.get_vocab_size():,} token")


if __name__ == "__main__":
    main()
