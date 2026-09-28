"""
EEGNet: Compact CNN for EEG-based BCI.

Reference
---------
Lawhern, V.J. et al.
"EEGNet: a compact convolutional neural network for EEG-based
brain-computer interfaces."
Journal of Neural Engineering, 15(5), 056013 (2018).
https://doi.org/10.1088/1741-2552/aace8c

Architecture
------------
Input (B, C, T)
  -> unsqueeze(1)                          : (B, 1, C, T)
  -> Block 1  Temporal conv               : (B, F1, C, T)
       Conv2d(1, F1, (1, kernLen)) + BN
  -> Block 2  Depthwise spatial conv      : (B, D*F1, 1, T//4)
       DepthwiseConv2d(F1, D*F1, (C,1)) + BN + ELU + AvgPool(1,4) + Dropout
  -> Block 3  Separable conv              : (B, F2, 1, T//32)
       Depthwise(1,sepKern) + Pointwise(F2) + BN + ELU + AvgPool(1,8) + Dropout
  -> Classifier                           : (B, n_classes)
       flatten -> Linear

Default hyperparameters (paper, 2-class MI)
-------------------------------------------
  F1            = 8       temporal filters
  D             = 2       depth multiplier (spatial)
  F2            = 16      separable conv output filters
  dropout       = 0.5
  kernel_length = sfreq // 2  (~T // 4 for a 2-second epoch)

Usage in LOSO loop
------------------
  model = EEGNet(n_channels, n_times).to(device)
  # no fold-specific initialization needed (unlike CSPNet)
  # then normal Adam + epoch loop
"""

import numpy as np
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet for 2-class (or multi-class) EEG motor imagery.

    Parameters
    ----------
    n_channels    : number of EEG channels C
    n_times       : number of time samples T
    n_classes     : number of output classes (default 2)
    F1            : number of temporal filters (default 8)
    D             : depth multiplier for spatial conv (default 2)
    F2            : number of separable conv output channels (default 16)
    kernel_length : temporal conv kernel length; None -> n_times // 4
                    (equivalent to sfreq // 2 for a 2-second epoch)
    dropout       : dropout probability (default 0.5, paper value)

    Input  : (B, C, T)  -- same convention as SPDNet / CSPNet in train_loso.py
    Output : (B, n_classes) logits
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        if kernel_length is None:
            # Paper: kernel_length = sfreq // 2.
            # For a 2-second epoch: T = 2 * sfreq, so sfreq // 2 = T // 4.
            kernel_length = max(16, n_times // 4)
        if kernel_length % 2 == 0:
            kernel_length += 1          # ensure odd for symmetric 'same' padding
        pad_t = kernel_length // 2

        # Separable conv kernel: ~125 ms at 128 Hz = 16 samples
        sep_kern = max(8, n_times // 16)
        if sep_kern % 2 == 0:
            sep_kern += 1
        sep_pad = sep_kern // 2

        # ── Block 1: Temporal convolution ────────────────────────────────────
        # (B, 1, C, T) -> (B, F1, C, T)  [same-length via symmetric padding]
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── Block 2: Depthwise spatial convolution ───────────────────────────
        # (B, F1, C, T) -> (B, D*F1, 1, T//4)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # ── Block 3: Separable convolution ───────────────────────────────────
        # (B, D*F1, 1, T//4) -> (B, F2, 1, T//32)
        self.block3 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, sep_kern),
                      padding=(0, sep_pad), groups=D * F1, bias=False),  # depthwise
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),                   # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # ── Classifier ───────────────────────────────────────────────────────
        with torch.no_grad():
            n_flat = self._forward_features(
                torch.zeros(1, 1, n_channels, n_times)
            ).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    # ── Feature extraction (no classification head) ─────────────────────────

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, C, T) -> (B, n_flat)"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) -- same input convention as SPDNet / CSPNet.
        Returns logits (B, n_classes).
        """
        return self.classifier(self._forward_features(x.unsqueeze(1)))


# ─────────────────────────────────────────────────────────────────────────────
# Quick parameter/shape sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    for name, C, T in [("Cho2017", 64, 256), ("Lee2019", 62, 200)]:
        model = EEGNet(n_channels=C, n_times=T)
        x = torch.randn(8, C, T)
        logits = model(x)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:10s} C={C} T={T}  "
              f"logits={tuple(logits.shape)}  "
              f"params={n_params:,}")
