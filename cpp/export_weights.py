"""
Sipher Weight Export — PyTorch → Binary (C++ için)
====================================================
Model ağırlıklarını raw binary olarak dışa aktarır.

Run: python3 export_weights.py
Output: weights/ dizininde .bin dosyaları
"""

import torch
import numpy as np
import struct
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrain_hybrid import CipherFormerHybrid, repair_cfg_d

CKPT = Path(__file__).parent.parent / "checkpoints" / "pretrain_hybrid.pt"
OUT_DIR = Path(__file__).parent / "weights"
OUT_DIR.mkdir(exist_ok=True)


def save_tensor(name, tensor):
    """Tensor'ı raw binary olarak kaydet (float64, row-major)."""
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    path = OUT_DIR / f"{name}.bin"
    arr.tofile(str(path))
    # Metadata
    meta_path = OUT_DIR / f"{name}.meta"
    with open(meta_path, "w") as f:
        f.write(f"{arr.ndim}\n")
        for s in arr.shape:
            f.write(f"{s}\n")
    print(f"  {name:40s} {str(arr.shape):20s} {arr.size*8/1024:.0f} KB")


def main():
    global OUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT),
                    help="Checkpoint yolu (varsayılan: checkpoints/pretrain_hybrid.pt)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR),
                    help="Çıktı dizini (varsayılan: cpp/weights)")
    ap.add_argument("--feat-degree", type=int, default=None,
                    help="Eski checkpoint'lerde cfg['feat_degree'] yanlış yazılmış olabilir; açıkça ver")
    args = ap.parse_args()
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    OUT_DIR = out_dir

    print("Sipher Weight Export")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {OUT_DIR}\n")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = repair_cfg_d(ckpt["config"], ckpt["model"])
    sd = ckpt["model"]
    feat_degree = args.feat_degree if args.feat_degree is not None else cfg.get("feat_degree", 1)
    # ffn_h'yi weight shape'inden çıkar — bazı checkpoint'lerde cfg['ffn_h'] yanlış
    # kayıtlı (d_model'e eşit, örn. 1024 yerine gerçek 4096). ffn.up.weight=[ffn_h, d].
    ffn_h = sd["layers.0.ffn.up.weight"].shape[0]

    print(f"  Config: d={cfg['d']} seq={cfg['seq']} nl={cfg['nl']} "
          f"nh={cfg['nh']} dk={cfg['dk']} emb={cfg['emb_dim']} ffn_h={ffn_h} feat={feat_degree}")
    print(f"  PPL: {ckpt['best_ppl']:.2f} | Epoch: {ckpt['epoch']}\n")

    # Config dosyası
    with open(OUT_DIR / "config.txt", "w") as f:
        f.write(f"vocab_size {ckpt['vocab_size']}\n")
        f.write(f"d_model {cfg['d']}\n")
        f.write(f"seq_len {cfg['seq']}\n")
        f.write(f"n_layers {cfg['nl']}\n")
        f.write(f"n_heads {cfg['nh']}\n")
        f.write(f"d_k {cfg['dk']}\n")
        f.write(f"emb_dim {cfg['emb_dim']}\n")
        f.write(f"window {cfg.get('window', 32)}\n")
        f.write(f"ffn_h {ffn_h}\n")
        f.write(f"feat_degree {feat_degree}\n")

    # Embedding
    save_tensor("token_emb", sd["token_emb.weight"])
    save_tensor("emb_proj", sd["emb_proj.weight"])
    save_tensor("pos_emb", sd["pos_emb.weight"])
    save_tensor("head_proj", sd["head_proj.weight"])

    # Her katman
    for i in range(cfg["nl"]):
        prefix = f"layers.{i}"
        save_tensor(f"{prefix}.W_Q", sd[f"{prefix}.attn.W_Q.weight"])
        save_tensor(f"{prefix}.W_K", sd[f"{prefix}.attn.W_K.weight"])
        save_tensor(f"{prefix}.W_V", sd[f"{prefix}.attn.W_V.weight"])
        save_tensor(f"{prefix}.W_O", sd[f"{prefix}.attn.W_O.weight"])
        save_tensor(f"{prefix}.gate_global", sd[f"{prefix}.attn.gate_global"])
        save_tensor(f"{prefix}.gate_local", sd[f"{prefix}.attn.gate_local"])
        save_tensor(f"{prefix}.ffn_up", sd[f"{prefix}.ffn.up.weight"])
        save_tensor(f"{prefix}.ffn_down", sd[f"{prefix}.ffn.down.weight"])
        save_tensor(f"{prefix}.poly_coeffs", sd[f"{prefix}.ffn.act.coeffs"])
        save_tensor(f"{prefix}.norm1_w", sd[f"{prefix}.norm1.weight"])
        save_tensor(f"{prefix}.norm1_b", sd[f"{prefix}.norm1.bias"])
        save_tensor(f"{prefix}.norm2_w", sd[f"{prefix}.norm2.weight"])
        save_tensor(f"{prefix}.norm2_b", sd[f"{prefix}.norm2.bias"])

    total_size = sum(f.stat().st_size for f in OUT_DIR.glob("*.bin"))
    print(f"\n  Toplam: {total_size/1e6:.0f} MB ({len(list(OUT_DIR.glob('*.bin')))} dosya)")
    print(f"  ✓ {OUT_DIR}")


if __name__ == "__main__":
    main()
