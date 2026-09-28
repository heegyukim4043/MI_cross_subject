"""
AdaBN: Adaptive Batch Normalization for test-time domain adaptation.

The model weights stay completely frozen. Only BN running statistics
(running_mean, running_var) are updated using unlabeled test-subject data.

This corrects for subject-level distribution shift in amplitude, baseline,
and noise level without any label or gradient.

Reference: Li et al., "Revisiting Batch Normalization for Practical
Domain Adaptation", ICLR Workshop 2017.

Usage
-----
    model.load_state_dict(best_state)
    apply_adabn(model, X_test, device)   # updates BN stats in-place
    model.eval()
    acc = evaluate(model, test_loader)   # uses adapted stats
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def apply_adabn(
    model: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    n_passes: int = 3,
) -> int:
    """
    Update all BN running statistics using test-subject data (no labels).

    Uses cumulative moving average (momentum=None) so every batch
    contributes equally regardless of n_passes.

    Parameters
    ----------
    model      : trained model with BN layers — modified in-place
    X_test     : (N, C, T) float32 test subject array, already normalised
    device     : target device
    batch_size : forward-pass batch size (default 64)
    n_passes   : passes over test data (default 3; 1 often sufficient
                 for N≥100, more passes give better stat estimates)

    Returns
    -------
    n_bn : number of BN layers found and adapted
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    if not bn_layers:
        return 0

    # Switch to cumulative MA and reset stats so only test data contributes
    orig_momentum = {}
    for bn in bn_layers:
        orig_momentum[bn] = bn.momentum
        bn.momentum = None          # cumulative: weight_k = 1/k
        bn.reset_running_stats()

    model.train()   # train mode → BN uses batch stats AND updates running stats
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for _ in range(n_passes):
        for (xb,) in loader:
            xb = xb.to(device)
            try:
                model(xb)           # output ignored; BN stats update as side-effect
            except Exception:
                pass                # e.g. tuple-output models — forward still runs BN

    # Restore momentum and return to eval mode
    for bn in bn_layers:
        bn.momentum = orig_momentum[bn]
    model.eval()

    return len(bn_layers)


def adabn_summary(bn_before: dict, model: nn.Module) -> str:
    """Return a short string showing mean shift in BN running stats."""
    shifts = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in bn_before:
            mean_shift = (m.running_mean - bn_before[name]["mean"]).abs().mean().item()
            var_shift  = (m.running_var  - bn_before[name]["var"] ).abs().mean().item()
            shifts.append((mean_shift, var_shift))
    if not shifts:
        return "no shift recorded"
    avg_mean = sum(s[0] for s in shifts) / len(shifts)
    avg_var  = sum(s[1] for s in shifts) / len(shifts)
    return f"Δmean={avg_mean:.4f}  Δvar={avg_var:.4f}  ({len(shifts)} BN layers)"


def snapshot_bn_stats(model: nn.Module) -> dict:
    """Capture current BN running stats for before/after comparison."""
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            snap[name] = {
                "mean": m.running_mean.clone(),
                "var":  m.running_var.clone(),
            }
    return snap
