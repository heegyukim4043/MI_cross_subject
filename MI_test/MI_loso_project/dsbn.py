"""Domain-specific BatchNorm utilities for cross-dataset EEG transfer.

The BN running statistics are domain-specific, while affine scale/bias are
shared. In source-only cross-dataset training this avoids untrained target
 affine parameters while still letting target-domain unlabeled data estimate
 its own BN statistics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DomainSpecificBatchNorm2d(nn.Module):
    def __init__(self, source: nn.BatchNorm2d, num_domains: int = 2):
        super().__init__()
        self.num_domains = num_domains
        self.active_domain = 0
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(
                source.num_features,
                eps=source.eps,
                momentum=source.momentum,
                affine=False,
                track_running_stats=source.track_running_stats,
            )
            for _ in range(num_domains)
        ])
        for bn in self.bns:
            bn.running_mean.data.copy_(source.running_mean.data)
            bn.running_var.data.copy_(source.running_var.data)
            bn.num_batches_tracked.data.copy_(source.num_batches_tracked.data)

        if source.affine:
            self.weight = nn.Parameter(source.weight.detach().clone())
            self.bias = nn.Parameter(source.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def set_domain(self, domain: int) -> None:
        if domain < 0 or domain >= self.num_domains:
            raise ValueError(f"domain index out of range: {domain}")
        self.active_domain = int(domain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bns[self.active_domain](x)
        if self.weight is None:
            return y
        shape = [1, -1] + [1] * (y.dim() - 2)
        return y * self.weight.view(*shape) + self.bias.view(*shape)


def convert_batchnorm_to_dsbn(module: nn.Module, num_domains: int = 2) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DomainSpecificBatchNorm2d(child, num_domains=num_domains))
        else:
            convert_batchnorm_to_dsbn(child, num_domains=num_domains)
    return module


def set_dsbn_domain(module: nn.Module, domain: int) -> int:
    n = 0
    for child in module.modules():
        if isinstance(child, DomainSpecificBatchNorm2d):
            child.set_domain(domain)
            n += 1
    return n


@torch.no_grad()
def apply_dsbn_target_stats(
    model: nn.Module,
    X_target: np.ndarray,
    device: torch.device,
    domain: int = 1,
    batch_size: int = 64,
    n_passes: int = 3,
) -> int:
    dsbn_layers = [m for m in model.modules() if isinstance(m, DomainSpecificBatchNorm2d)]
    if not dsbn_layers:
        return 0

    set_dsbn_domain(model, domain)
    orig_momentum = {}
    target_bns = []
    for layer in dsbn_layers:
        bn = layer.bns[domain]
        target_bns.append(bn)
        orig_momentum[bn] = bn.momentum
        bn.momentum = None
        bn.reset_running_stats()

    model.train()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_target.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    for _ in range(n_passes):
        for (xb,) in loader:
            out = model(xb.to(device))
            if isinstance(out, tuple):
                out = out[0]

    for bn in target_bns:
        bn.momentum = orig_momentum[bn]
    model.eval()
    set_dsbn_domain(model, domain)
    return len(dsbn_layers)
