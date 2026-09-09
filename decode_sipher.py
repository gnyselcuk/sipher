#!/usr/bin/env python3
"""Sipher token ID listesini decode eder. Kullanım: python3 decode_sipher.py 4412 7314 6006 ..."""
import json
import sys

TOKENIZER_PATH = "data/sipher_tokenizer/sipher_tokenizer.json"

def main():
    tok = json.load(open(TOKENIZER_PATH))
    vocab = tok.get("model", {}).get("vocab", {})
    id2tok = {v: k for k, v in vocab.items()}
    ids = [int(a) for a in sys.argv[1:]]
    for tid in ids:
        print(f"{tid}\t{id2tok.get(tid, '???')}")

if __name__ == "__main__":
    main()
