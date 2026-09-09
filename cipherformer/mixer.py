"""
CipherMixer: FHE-native token mixing layer.

Do not use this for causal LM training. token_up mixes across the sequence
dimension (nn.Linear(seq_len, H) after a transpose), so position i attends to
future tokens; the model just copies the next token and reports a fake PPL
around 1.0. The C++ FHE engine does not include this mixer (export omits it),
so mixer-trained checkpoints cannot run in the FHE path either.

Train with --no-mixer instead: the HybridAttention local-window path is the
causally safe, FHE-compatible one. This file is kept as the negative-result
artifact; see docs/SIPHER_V1.md.

Design, for reference: MLP-based token mixing with all mixing weights in
plaintext (ct-pt in FHE); only the activation needs ct-ct. The old "54,146x
faster than attention" figure is not meaningful, since it measured the leaky
bidirectional mixer rather than a valid causal baseline.
"""

import torch
import torch.nn as nn
from .layers import PolyActivation, PolyLayerNorm


class CipherMixerBlock(nn.Module):
    """
    MLP-Mixer block adapted for FHE.

    Token mixing:  W_mix · x^T  (ct-pt, weights plaintext)
    Channel mixing: W_chan · x   (ct-pt, weights plaintext)
    Activation:     polynomial   (ct-ct, only FHE-expensive part)

    Architecture:
      x → LN → Transpose → Linear(seq→H) → PolyAct → Linear(H→seq) → Transpose → + residual
        → LN → Linear(d→4d) → PolyAct → Linear(4d→d) → + residual
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        mixer_hidden: int = 1024,
        ffn_mult: int = 4,
        act_degree: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Token mixing
        self.norm1 = PolyLayerNorm(d_model)
        self.token_up = nn.Linear(seq_len, mixer_hidden, bias=True)
        self.token_act = PolyActivation(degree=act_degree)
        self.token_down = nn.Linear(mixer_hidden, seq_len, bias=True)
        self.drop1 = nn.Dropout(dropout)

        # Channel mixing (FFN)
        self.norm2 = PolyLayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.chan_up = nn.Linear(d_model, d_ff, bias=True)
        self.chan_act = PolyActivation(degree=act_degree)
        self.chan_down = nn.Linear(d_ff, d_model, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]

        # Token mixing
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)          # [B, d_model, seq]
        x = self.token_up(x)            # [B, d_model, H] — ct-pt
        x = self.token_act(x)           # ct-ct (polynomial)
        x = self.token_down(x)          # [B, d_model, seq] — ct-pt
        x = x.transpose(1, 2)          # [B, seq, d_model]
        x = self.drop1(x)
        x = x + residual

        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.chan_up(x)             # [B, seq, d_ff] — ct-pt
        x = self.chan_act(x)            # ct-ct (polynomial)
        x = self.chan_down(x)           # [B, seq, d_model] — ct-pt
        x = self.drop2(x)
        x = x + residual

        return x
