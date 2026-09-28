"""
EEG Conformer (Song et al. 2022, IEEE TNSRE)
  CNN front-end → Transformer encoder → classifier

Input : (B, n_channels, n_times)
Output: (B, n_classes) logits

Design notes:
  - CNN front-end mirrors EEGNet structure (temporal + depthwise spatial conv + BN)
    so AdaBN / TENT work on the BN layers exactly as with CSPNet/EEGNet
  - Transformer encoder captures long-range temporal dependencies after CNN patches
  - Global average pooling over time dimension → classifier
    makes the model input-length agnostic (Cho2017 T=257, Lee2019 T=201 both work)
  - No special fit_ initializer needed; plug straight into train_loso build_model
"""

import math
import torch
import torch.nn as nn


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 257,
        n_classes: int = 2,
        F1: int = 40,           # temporal conv filters
        D: int = 2,             # depthwise multiplier → F2 = F1*D
        temp_kern: int = 25,    # temporal kernel (samples); use odd number
        pool: int = 8,          # avg pool factor after spatial conv
        dropout: float = 0.5,
        nhead: int = 8,         # attention heads (F2 must be divisible by nhead)
        n_layers: int = 2,      # transformer encoder layers
        ff_dim: int = 256,      # transformer feed-forward hidden dim
        attn_dropout: float = 0.3,
    ):
        super().__init__()
        F2 = F1 * D
        assert F2 % nhead == 0, f"F2={F2} must be divisible by nhead={nhead}"
        self.embed_dim = F2

        # ── CNN Patch Embedding ──────────────────────────────────────────────
        pad = temp_kern // 2  # 'same' padding for odd kernel
        self.cnn = nn.Sequential(
            # Temporal conv: shape-preserving
            nn.Conv2d(1, F1, (1, temp_kern), padding=(0, pad), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv: collapse channel dim
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool)),
            nn.Dropout(dropout),
        )

        # ── Transformer Encoder ──────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=F2,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,
            batch_first=True,   # (B, T', F2)
            norm_first=True,    # pre-norm: more stable for small T'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(F2)

        # ── Classifier (GAP → FC) ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(F2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, F2) global feature"""
        x = x.unsqueeze(1)              # (B, 1, C, T)
        x = self.cnn(x)                 # (B, F2, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T', F2)
        x = self.transformer(x)         # (B, T', F2)
        x = self.norm(x).mean(dim=1)    # (B, F2)  global avg pool over time
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
