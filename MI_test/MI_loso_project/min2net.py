"""
MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification
Reference: Autthasan et al., IEEE TNSRE 2022 (doi:10.1109/TNSRE.2021.3137184)

Architecture
------------
EEGNet-based encoder -> shared bottleneck (latent_dim) -> two heads:
  1. Classifier  : latent -> n_classes          (cross-entropy)
  2. Reconstructor: latent -> C x T             (MSE, acts as regulariser)

Multi-task loss:  L = alpha * CE + (1 - alpha) * MSE_recon

Usage with train_loso.py
------------------------
    python train_loso.py --dataset cho2017 --model min2net
    python train_loso.py --dataset cho2017 --model min2net --augment
"""

import torch
import torch.nn as nn


class MIN2Net(nn.Module):
    """
    Parameters
    ----------
    n_channels  : int    number of EEG channels (C)
    n_times     : int    number of time samples  (T)
    n_classes   : int    default 2
    F1          : int    number of temporal filters  (default 8)
    D           : int    depth multiplier for spatial conv (default 2)
    F2          : int    number of separable-conv filters (default 16)
    latent_dim  : int    bottleneck size (default 64)
    dropout     : float  dropout rate (default 0.25)
    alpha       : float  weight on classification loss; (1-alpha) on recon loss (default 0.9)
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        latent_dim: int = 64,
        dropout: float = 0.25,
        alpha: float = 0.9,
    ):
        super().__init__()
        C, T = n_channels, n_times
        self.alpha = alpha
        self._C = C
        self._T = T

        # ---- Encoder --------------------------------------------------------
        # Block 1: temporal convolution (kernel ~ T/2, 'same' padding)
        kT  = T // 2
        kT  = kT if kT % 2 == 1 else kT + 1   # ensure odd for symmetric padding
        pad = kT // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kT), padding=(0, pad), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: spatial depthwise conv (across channels)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        T2 = T // 4  # time dim after pool

        # Block 3: separable conv (depthwise + pointwise)
        self.block3 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8),
                      groups=D * F1, bias=False),          # depthwise
            nn.Conv2d(D * F1, F2, 1, bias=False),          # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        T3 = max(T2 // 8, 1)   # time dim after second pool (guard against 0)
        enc_flat = F2 * T3

        # Bottleneck FC
        self.enc_fc  = nn.Linear(enc_flat, latent_dim)

        # ---- Classifier head ------------------------------------------------
        self.classifier = nn.Linear(latent_dim, n_classes)

        # ---- Reconstructor head  (MLP decoder) ------------------------------
        # Projects back to original signal space; MSE loss regularises the
        # shared representation — the core idea from the paper.
        hidden = min(512, C * T)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, C * T),
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> z: (B, latent_dim)"""
        h = x.unsqueeze(1)        # (B, 1, C, T)
        h = self.block1(h)        # (B, F1, C, T)
        h = self.block2(h)        # (B, D*F1, 1, T//4)
        h = self.block3(h)        # (B, F2, 1, T//32)
        z = self.enc_fc(h.flatten(1))
        return z

    def forward(self, x: torch.Tensor):
        """
        Training  : returns (logits, x_recon)
        Inference : returns logits
        """
        B = x.shape[0]
        z      = self.encode(x)
        logits = self.classifier(z)

        if self.training:
            x_recon = self.decoder(z).view(B, self._C, self._T)
            return logits, x_recon

        return logits
