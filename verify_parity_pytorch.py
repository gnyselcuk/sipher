"""
Sipher GPU FHE Parity — PyTorch trained-model cross-check
===========================================================
C++ plaintext mirror çıktısını (GPU engine `--parity --dump` ile üretilir)
eğitilmiş PyTorch modelinin forward'ıyla karşılaştırır.

Zincir:  GPU FHE  ≈  C++ plaintext mirror  ≈  PyTorch trained model
          (--parity ile ölçülür)   (bu script ile ölçülür)

Kullanım:
  1. C++ tarafı (mirror çıktısını dök):
     cd cpp/gpu && ./sipher_fhe_gpu --parity 1 ../weights/ --dump mirror.bin
  2. Bu script:
     python3 verify_parity_pytorch.py --ckpt checkpoints/pretrain_hybrid.pt \
         --dump cpp/gpu/mirror.bin [--layers 1]

Çıktı: katman başına cos_sim / max_err + son katman logits top-5 uyumu.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pretrain_hybrid import CipherFormerHybrid

# C++ main_gpu.cpp'deki test tokenları ("Merkez Bankası faiz")
TOKENS = [101, 2847, 5523, 1024]


def load_dump(path: str):
    """C++ --parity --dump dosyasını oku: int32 n_layers, int32 dim, n_layers×dim float64."""
    with open(path, "rb") as f:
        nl = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        out = []
        for _ in range(nl):
            raw = f.read(dim * 8)
            out.append(np.frombuffer(raw, dtype=np.float64))
    return nl, dim, out


def main():
    ap = argparse.ArgumentParser(description="Sipher C++ mirror vs PyTorch trained model")
    ap.add_argument("--ckpt", default="checkpoints/pretrain_hybrid.pt")
    ap.add_argument("--dump", default="cpp/gpu/mirror.bin")
    ap.add_argument("--layers", type=int, default=0,
                    help="karşılaştırılacak katman sayısı (0 = dump'taki kadar)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocab_size"]

    model = CipherFormerHybrid(
        vocab=vocab,
        d=cfg["d"],
        seq=cfg["seq"],
        nl=cfg["nl"],
        nh=cfg["nh"],
        dk=cfg["dk"],
        ffn_h=cfg["ffn_h"],
        emb_dim=cfg.get("emb_dim", 128),
        window=cfg.get("window", 32),
        causal_global=cfg.get("causal_global", True),
        no_mixer=cfg.get("no_mixer", True),
        poly_ffn=cfg.get("poly_ffn", True),
        gate_local_init=cfg.get("gate_local_init", 0.0),
    )
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        print(f"HATA: state_dict uyuşmuyor — checkpoint farklı bayraklarla mı eğitildi?\n{e}")
        sys.exit(1)
    model.eval()

    seq = cfg["seq"]
    tokens = TOKENS + [0] * (seq - len(TOKENS))
    ids = torch.tensor([tokens], dtype=torch.long)
    pos = seq - 1

    # Her bloğun çıktısını (katman hidden state'i) pos pozisyonunda yakala
    states = []
    hooks = []
    for blk in model.layers:
        hooks.append(blk.register_forward_hook(
            lambda m, i, o: states.append(o.detach().float()[0, pos].numpy())))
    with torch.no_grad():
        logits = model(ids)
    for h in hooks:
        h.remove()

    nl, dim, dump = load_dump(args.dump)
    n = args.layers or min(nl, len(states))
    print(f"PyTorch katman: {len(states)}, C++ dump katman: {nl}, karşılaştırılan: {n}")
    if dim != cfg["d"]:
        print(f"HATA: boyut uyuşmazlığı — dump dim={dim}, model d={cfg['d']}")
        sys.exit(1)

    all_ok = True
    for l in range(n):
        ref = states[l]          # PyTorch (float32)
        mir = dump[l]            # C++ mirror (float64)
        cos = float(np.dot(ref, mir) / (np.linalg.norm(ref) * np.linalg.norm(mir)))
        err = float(np.max(np.abs(ref - mir)))
        ok = cos > 0.9999 and err < 0.05
        all_ok = all_ok and ok
        print(f"  Layer {l}: cos_sim={cos:.8f} max_err={err:.3e} {'✓' if ok else '✗'}")

    # Son katman logits karşılaştırması (C++ mirror → logits)
    h = dump[-1]
    head_proj = model.head_proj.weight.detach().numpy()   # [emb, d]
    tok_emb = model.token_emb.weight.detach().numpy()     # [vocab, emb]
    logits_mirror = (head_proj @ h) @ tok_emb.T

    logits_pt = logits[0, pos].numpy()
    cos = float(np.dot(logits_pt, logits_mirror) /
                (np.linalg.norm(logits_pt) * np.linalg.norm(logits_mirror)))
    top_pt = set(np.argsort(logits_pt)[-5:].tolist())
    top_mir = set(np.argsort(logits_mirror)[-5:].tolist())
    hit = len(top_pt & top_mir)
    print(f"  Logits: cos_sim={cos:.6f} top-5 uyumu={hit}/5")

    print(f"\n═══ PYTHON CROSS-CHECK {'OK ✓' if all_ok else 'FAIL ✗'} ═══")
    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
