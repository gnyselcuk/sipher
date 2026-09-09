"""
CipherFormer: FHE-Native Language Model.

Architecture: 3:1 hybrid of CipherMixer (ct-pt token mixing) and
CipherAttention (linear attention for data-dependent mixing).

All operations are polynomial-computable under CKKS.
No softmax. No GELU. No standard attention.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .layers import PolyActivation, PolyLayerNorm
from .mixer import CipherMixerBlock
from .attention import CipherAttentionBlock


@dataclass
class CipherFormerConfig:
    vocab_size: int = 256          # character-level for prototype
    d_model: int = 256             # smaller for fast prototyping
    n_layers: int = 8
    n_heads: int = 2               # few heads (FHE cost scales linearly)
    seq_len: int = 128
    mixer_hidden: int = 256        # token mixing hidden dim
    ffn_mult: int = 4
    act_degree: int = 4            # polynomial activation degree
    dropout: float = 0.1
    mixer_attn_ratio: int = 3      # N mixer layers per 1 attention layer
    tie_weights: bool = True

    @property
    def layer_types(self) -> List[str]:
        """Generate layer type sequence: 3 mixer : 1 attention."""
        types = []
        for i in range(self.n_layers):
            if (i + 1) % (self.mixer_attn_ratio + 1) == 0:
                types.append("attention")
            else:
                types.append("mixer")
        return types


class CipherFormer(nn.Module):
    """
    FHE-Native Transformer.

    Replaces every FHE-hostile operation:
      softmax → eliminated (linear attention)
      GELU → polynomial activation (degree 4)
      LayerNorm → polynomial LayerNorm
      Q·K^T → φ(Q)·(φ(K)^T·V) or MLP-Mixer
    """

    def __init__(self, config: CipherFormerConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # Build layers based on 3:1 ratio
        self.layers = nn.ModuleList()
        for ltype in config.layer_types:
            if ltype == "mixer":
                self.layers.append(CipherMixerBlock(
                    d_model=config.d_model,
                    seq_len=config.seq_len,
                    mixer_hidden=config.mixer_hidden,
                    ffn_mult=config.ffn_mult,
                    act_degree=config.act_degree,
                    dropout=config.dropout,
                ))
            else:
                self.layers.append(CipherAttentionBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    act_degree=config.act_degree,
                    dropout=config.dropout,
                ))

        # Output head
        self.final_norm = PolyLayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.head(x)  # [B, S, vocab_size]
        return logits

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_mixer = sum(1 for l in self.layers if isinstance(l, CipherMixerBlock))
        n_attn = sum(1 for l in self.layers if isinstance(l, CipherAttentionBlock))
        return {
            "total": total,
            "trainable": trainable,
            "n_mixer_layers": n_mixer,
            "n_attn_layers": n_attn,
            "layer_types": self.config.layer_types,
        }

    def fhe_cost_summary(self) -> str:
        """Estimate FHE cost based on analytical model."""
        from pathlib import Path
        lines = [
            "CipherFormer FHE Cost Estimate",
            "=" * 50,
            f"  Layers: {self.config.n_layers} "
            f"({sum(1 for t in self.config.layer_types if t == 'mixer')} mixer + "
            f"{sum(1 for t in self.config.layer_types if t == 'attention')} attention)",
            f"  d_model: {self.config.d_model}",
            f"  n_heads: {self.config.n_heads}",
            f"  seq_len: {self.config.seq_len}",
            "",
            "  Per-layer FHE cost (N=32768, empirical timings):",
        ]
        # Rough estimates based on our benchmarks
        mixer_s = 25.7 * (self.config.d_model / 768) * (self.config.seq_len / 512)
        attn_hr = 1.98 * (self.config.n_heads / 12) * (self.config.seq_len / 512)
        n_mixer = sum(1 for t in self.config.layer_types if t == "mixer")
        n_attn = sum(1 for t in self.config.layer_types if t == "attention")
        total_s = n_mixer * mixer_s + n_attn * attn_hr * 3600
        lines.append(f"    CipherMixer: ~{mixer_s:.1f} s × {n_mixer} = {n_mixer * mixer_s:.1f} s")
        lines.append(f"    CipherAttn:  ~{attn_hr:.2f} hr × {n_attn} = {n_attn * attn_hr:.2f} hr")
        lines.append(f"    Total: ~{total_s / 3600:.2f} hr")
        lines.append(f"    With 21× packing: ~{total_s / 21 / 60:.1f} min")
        return "\n".join(lines)
