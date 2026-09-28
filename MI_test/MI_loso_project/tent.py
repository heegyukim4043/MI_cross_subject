"""
TENT: Test-time Entropy Minimization for domain adaptation.

Updates only BN affine parameters (weight, bias) at test time by
minimizing prediction entropy on unlabeled test data.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
Minimization", ICLR 2021.

Difference from AdaBN
---------------------
AdaBN : updates BN running statistics only (no gradient).
TENT  : updates BN affine params (γ, β) via gradient on entropy loss.
        Running stats are also updated (model.train() mode).
        → More expressive than AdaBN, but requires a backward pass.

Usage
-----
    apply_adabn(model, X_test, device)   # optional: warm-start BN stats first
    apply_tent(model, X_test, device)    # entropy minimization on BN affine
    acc = evaluate(model, test_loader)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def collect_bn_affine_params(model: nn.Module) -> list:
    """Return BN affine parameters (weight + bias) that should be updated."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the softmax distribution. Shape: (B,) → scalar."""
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()


def apply_tent(
    model: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    n_steps: int = 1,
    lr: float = 1e-3,
    use_adabn_warmup: bool = True,
    adabn_passes: int = 3,
) -> int:
    """
    Minimise prediction entropy on test data by updating BN affine params.

    Parameters
    ----------
    model           : trained model — modified in-place
    X_test          : (N, C, T) float32 test subject array
    device          : target device
    batch_size      : mini-batch size
    n_steps         : gradient steps per pass (default 1 — one full pass)
    lr              : Adam learning rate for BN affine params
    use_adabn_warmup: if True, first run AdaBN to warm-start BN statistics
    adabn_passes    : passes for optional AdaBN warm-start

    Returns
    -------
    n_bn : number of BN layers updated
    """
    from adabn import apply_adabn

    bn_affine = collect_bn_affine_params(model)
    if not bn_affine:
        return 0

    # ── Optional AdaBN warm-start ─────────────────────────────────────────────
    if use_adabn_warmup:
        apply_adabn(model, X_test, device,
                    batch_size=batch_size, n_passes=adabn_passes)

    # ── Freeze everything, unfreeze only BN affine ────────────────────────────
    original_requires_grad = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)
    for p in bn_affine:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(bn_affine, lr=lr)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()  # BN uses batch stats + updates running stats

    for _ in range(n_steps):
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            try:
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
            except Exception:
                continue
            loss = softmax_entropy(logits)
            loss.backward()
            optimizer.step()

    # ── Restore requires_grad and switch to eval ──────────────────────────────
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad.get(name, False))
    model.eval()

    return len(bn_affine) // 2  # each BN has weight + bias → n BN layers


def tent_summary(model: nn.Module, bn_before: dict) -> str:
    """Short string showing shift in BN affine params after TENT."""
    shifts = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in bn_before:
            w_shift = (m.weight - bn_before[name]["weight"]).abs().mean().item()
            b_shift = (m.bias   - bn_before[name]["bias"]  ).abs().mean().item()
            shifts.append((w_shift, b_shift))
    if not shifts:
        return "no shift"
    aw = sum(s[0] for s in shifts) / len(shifts)
    ab = sum(s[1] for s in shifts) / len(shifts)
    return f"Δweight={aw:.4f}  Δbias={ab:.4f}  ({len(shifts)} BN layers)"


def snapshot_bn_affine(model: nn.Module) -> dict:
    """Capture BN affine params for before/after comparison."""
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            snap[name] = {
                "weight": m.weight.detach().clone() if m.weight is not None else None,
                "bias":   m.bias.detach().clone()   if m.bias   is not None else None,
            }
    return snap
