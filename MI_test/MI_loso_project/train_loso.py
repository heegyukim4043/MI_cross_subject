"""
LOSO (Leave-One-Subject-Out) training and evaluation for SPDNet.

Usage:
    python train_loso.py                    # both datasets
    python train_loso.py --dataset cho2017  # only Cho2017
    python train_loso.py --dataset lee2019  # only Lee2019_MI

Results are printed per-subject and saved to results/loso_results_*.csv.

Resume support:
  --resume will load existing results for the same run_id and skip
  already-completed subjects. Results are appended per subject.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from spd_net        import SPDNet
from riemgat_net    import RiemGATNet
from min2net        import MIN2Net
from cspnet         import CSPNet, fit_csp_layer
from eegnet         import EEGNet
from conformer      import EEGConformer
from cspnet_snn     import CSPNetSNN, fit_csp_layer_snn
from cspnet_rsnn    import CSPNetRSNN, fit_csp_layer_rsnn
from cspnet_dann         import CSPNetDANN, fit_csp_layer_dann, dann_loss, grl_lambda
from cspnet_contrastive  import CSPNetContrastive, fit_csp_layer_contrastive
from discriminator_selective import (
    TangentSpaceExtractor,
    TangentAutoEncoder,
    DiscriminatorSelectiveModel,
    discriminator_losses,
)
from eeg_augment    import EEGAugment
from selection_viz  import save_selection_plots
from trial_selection import score_trials, select_trials, score_to_weights
from adabn import apply_adabn, snapshot_bn_stats, adabn_summary
from eeg_ea import EA_EPS_MODES, apply_ea_loso
from tent  import apply_tent, snapshot_bn_affine, tent_summary
from sam          import SAM
from eeg_style_aug import build_style_aug, CovarianceStyleAug

# -----------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR = os.path.abspath(
    os.environ.get("MI_PREPROCESSED_DIR", os.path.join(_BASE_DIR, "preprocessed"))
)
RESULTS_DIR = os.path.abspath(
    os.environ.get("MI_RESULTS_DIR", os.path.join(_BASE_DIR, "results"))
)
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMON_CHO_LEE_CHANNELS = [
    "Fp1", "AF7", "AF3", "F3", "F7", "FC5", "FC3", "FC1", "C1", "C3", "C5",
    "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P7", "PO3", "O1", "Oz",
    "POz", "Pz", "CPz", "Fp2", "AF8", "AF4", "Fz", "F4", "F8", "FC6", "FC4",
    "FC2", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2",
    "P4", "P8", "PO4", "O2",
]

# Quadro RTX 6000 on this host hits illegal CUDA memory access inside cuDNN
# for CSPNet forward passes. Disabling cuDNN keeps training on GPU while using
# native PyTorch kernels for the affected ops.
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

# Hyperparameters
N_FILTERS  = 32
DROPOUT    = 0.5
LR         = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS     = 300
SEED       = int(os.environ.get("MI_SEED", "2026"))            # weight init / training randomness
SPLIT_SEED = int(os.environ.get("MI_SPLIT_SEED", str(SEED)))  # validation-subject selection


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def load_data(dataset_name: str, ch_filter: str = None):
    """
    Load preprocessed data.
    ch_filter : if set, keep only channels whose name contains this string
                e.g. ch_filter='C' -> C3, Cz, C4, FC3, CP4, ...
    """
    path = os.path.join(SAVE_DIR, f"{dataset_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run preprocess_data.py first."
        )
    d = np.load(path, allow_pickle=True)
    X        = d["X"].astype(np.float32)     # (N, C, T)
    y        = d["y"].astype(np.int64)        # (N,)
    subjects = d["subjects"].astype(np.int64) # (N,)
    sfreq    = float(d["sfreq"])
    ch_names = list(d["ch_names"])

    if ch_filter is not None:
        if ch_filter == "common_cho_lee":
            keep_channels = set(COMMON_CHO_LEE_CHANNELS)
            idx = [i for i, ch in enumerate(ch_names) if ch in keep_channels]
        elif "," in ch_filter:
            requested = {ch.strip() for ch in ch_filter.split(",") if ch.strip()}
            idx = [i for i, ch in enumerate(ch_names) if ch in requested]
        else:
            idx = [i for i, ch in enumerate(ch_names) if ch_filter in ch]
        X        = X[:, idx, :]
        ch_names = [ch_names[i] for i in idx]
        print(f"  Channel filter '{ch_filter}': {len(idx)} channels selected")
        print(f"  -> {ch_names}")

    print(f"  Loaded {dataset_name}: X={X.shape}, "
          f"{len(np.unique(subjects))} subjects, sfreq={sfreq} Hz")
    return X, y, subjects, ch_names, sfreq


def normalize_subject(X_train, X_test):
    """Channel-wise z-score using training statistics."""
    mu  = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std


# -----------------------------------------------------------------------------
# Training / Evaluation
# -----------------------------------------------------------------------------

def build_model(model_name: str, n_channels: int, n_times: int, n_filters: int,
                n_subjects: int = 51, dann_kwargs: dict | None = None):
    """Factory for all supported LOSO models."""
    dann_kwargs = dann_kwargs or {}
    if model_name == "riemgat":
        return RiemGATNet(
            n_channels=n_channels,
            n_times=n_times,
            dropout=DROPOUT,
        ).to(DEVICE)
    if model_name == "min2net":
        return MIN2Net(
            n_channels=n_channels,
            n_times=n_times,
            dropout=0.25,
        ).to(DEVICE)
    if model_name == "cspnet":
        return CSPNet(
            n_channels=n_channels,
            n_times=n_times,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
        ).to(DEVICE)
    if model_name == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            F1=8,
            D=2,
            F2=16,
            dropout=0.5,
        ).to(DEVICE)
    if model_name == "conformer":
        return EEGConformer(
            n_channels=n_channels,
            n_times=n_times,
            F1=40,
            D=2,
            temp_kern=25,
            pool=8,
            dropout=0.5,
            nhead=8,
            n_layers=2,
            ff_dim=256,
            attn_dropout=0.3,
        ).to(DEVICE)
    if model_name == "cspnetsnn":
        return CSPNetSNN(
            n_channels=n_channels,
            n_times=n_times,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
            T_sim=8,
            tau=2.0,
            v_threshold=1.0,
            encoding="direct",
        ).to(DEVICE)
    if model_name == "cspnetrsnn":
        return CSPNetRSNN(
            n_channels=n_channels,
            n_times=n_times,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
            T_sim=8,
            tau=2.0,
            v_threshold=1.0,
            encoding="direct",
            rec_gain=0.1,
        ).to(DEVICE)
    if model_name == "cspnetcontrastive":
        return CSPNetContrastive(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=2,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
            proj_dim=dann_kwargs.get("proj_dim", 64),
            proj_hidden=dann_kwargs.get("proj_hidden", 128),
            temperature=dann_kwargs.get("temperature", 0.07),
        ).to(DEVICE)
    if model_name == "cspnetdann":
        return CSPNetDANN(
            n_channels=n_channels,
            n_times=n_times,
            n_subjects=n_subjects,
            n_classes=2,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
            lambda_grl=0.0,
            qual_hidden=dann_kwargs.get("qual_hidden", 64),
            domain_hidden=dann_kwargs.get("domain_hidden", 0),
            domain_dropout=dann_kwargs.get("domain_dropout", 0.0),
            use_grl=dann_kwargs.get("use_grl", True),
        ).to(DEVICE)
    return SPDNet(
        n_channels=n_channels,
        n_filters=n_filters,
        dropout=DROPOUT,
    ).to(DEVICE)


def init_model_for_fold(model, model_name: str, X_train_prenorm, y_train_prenorm):
    """Fold-specific initialization such as CSP fitting."""
    if model_name == "cspnet":
        fit_csp_layer(model, X_train_prenorm, y_train_prenorm)
    if model_name == "cspnetsnn":
        fit_csp_layer_snn(model, X_train_prenorm, y_train_prenorm)
    if model_name == "cspnetrsnn":
        fit_csp_layer_rsnn(model, X_train_prenorm, y_train_prenorm)
    if model_name == "cspnetdann":
        fit_csp_layer_dann(model, X_train_prenorm, y_train_prenorm)
    if model_name == "cspnetcontrastive":
        fit_csp_layer_contrastive(model, X_train_prenorm, y_train_prenorm)

def _apply_sample_weights(loss_vec: torch.Tensor, sample_weight=None) -> torch.Tensor:
    """Reduce a per-sample loss vector with optional sample weights."""
    if sample_weight is None:
        return loss_vec.mean()
    weight = sample_weight.to(loss_vec.device).float().view(-1)
    weight = weight / weight.sum().clamp_min(1e-8)
    return (loss_vec.view(-1) * weight).sum()


def _classifier_features(model, xb: torch.Tensor):
    """Return encoder features and classifier for feature-level adaptation."""
    if isinstance(model, CSPNet):
        return model._forward_features(xb.unsqueeze(1)), model.classifier
    if isinstance(model, CSPNetContrastive):
        return model._encode(xb.unsqueeze(1)), model.classifier
    return None, None


def _same_class_feature_mixup_loss(model, xb: torch.Tensor, yb: torch.Tensor,
                                   alpha: float,
                                   label_smoothing: float = 0.0) -> torch.Tensor | None:
    """Mix latent representations only within the same class."""
    if alpha <= 0:
        return None
    feat, classifier = _classifier_features(model, xb)
    if feat is None:
        return None
    paired = torch.arange(len(yb), device=yb.device)
    has_pair = False
    for label in torch.unique(yb):
        idx = torch.where(yb == label)[0]
        if len(idx) > 1:
            paired[idx] = idx.roll(1)
            has_pair = True
    if not has_pair:
        return None
    lam = torch.distributions.Beta(alpha, alpha).sample().to(feat.device)
    mixed_feat = lam * feat + (1.0 - lam) * feat[paired]
    return F.cross_entropy(
        classifier(mixed_feat), yb, label_smoothing=label_smoothing
    )


def _forward_loss(model, xb, yb, wb, label_smoothing, feature_mixup_alpha, lambda_feature_mixup):
    """Single forward pass → (loss, logits). Handles MIN2Net tuple and feature mixup."""
    out = model(xb)
    if isinstance(out, tuple):
        logits, x_recon = out
        alpha = getattr(model, "alpha", 0.9)
        cls_loss = F.cross_entropy(logits, yb, reduction="none", label_smoothing=label_smoothing)
        recon_loss = F.mse_loss(x_recon, xb, reduction="none").mean(dim=(1, 2))
        loss = alpha * _apply_sample_weights(cls_loss, wb) + \
               (1 - alpha) * _apply_sample_weights(recon_loss, wb)
    else:
        logits = out
        cls_loss = F.cross_entropy(logits, yb, reduction="none", label_smoothing=label_smoothing)
        loss = _apply_sample_weights(cls_loss, wb)
    if lambda_feature_mixup > 0:
        mix_loss = _same_class_feature_mixup_loss(model, xb, yb, feature_mixup_alpha, label_smoothing)
        if mix_loss is not None:
            loss = loss + lambda_feature_mixup * mix_loss
    return loss, logits


def train_epoch(model, loader, optimizer, criterion, augment=None,
                feature_mixup_alpha: float = 0.0,
                lambda_feature_mixup: float = 0.0,
                label_smoothing: float = 0.0,
                use_sam: bool = False,
                style_aug=None):
    model.train()
    if augment is not None:
        augment.train()
    if style_aug is not None:
        style_aug.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, wb = batch
            wb = wb.to(DEVICE)
        else:
            xb, yb = batch
            wb = None
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if augment is not None:
            xb = augment(xb)
        if style_aug is not None:
            xb = style_aug(xb)

        if use_sam:
            # SAM two-step: first forward computes perturbation direction
            optimizer.zero_grad()
            loss, logits = _forward_loss(
                model, xb, yb, wb, label_smoothing, feature_mixup_alpha, lambda_feature_mixup
            )
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # Second forward at perturbed weights
            loss2, _ = _forward_loss(
                model, xb, yb, wb, label_smoothing, feature_mixup_alpha, lambda_feature_mixup
            )
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss, logits = _forward_loss(
                model, xb, yb, wb, label_smoothing, feature_mixup_alpha, lambda_feature_mixup
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


def _extract_logits(out):
    """Handle models that may return auxiliary outputs."""
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def compute_classifier_scores(model, X, y, method: str = "trueprob"):
    """
    Score training trials using a trained classifier.

    Higher is better for selection.
    """
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    out_scores = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = _extract_logits(model(xb))
        prob = torch.softmax(logits, dim=1)
        top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).values
        confidence = prob.max(dim=1).values
        trueprob = prob.gather(1, yb.unsqueeze(1)).squeeze(1)
        if top2.shape[1] == 2:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = top2[:, 0]

        if method == "confidence":
            score = confidence
        elif method == "margin":
            score = margin
        else:
            score = trueprob

        out_scores.append(score.detach().cpu().numpy())
    return np.concatenate(out_scores, axis=0).astype(np.float32)


def _minmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)


@torch.no_grad()
def compute_entropy_scores(model, X: np.ndarray) -> np.ndarray:
    """Classifier predictive entropy, normalised to [0, 1]."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    out_scores = []
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        logits = _extract_logits(model(xb))
        prob = torch.softmax(logits, dim=1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)
        entropy = entropy / np.log(prob.shape[1])
        out_scores.append(entropy.detach().cpu().numpy())
    return np.concatenate(out_scores, axis=0).astype(np.float32)


def _find_last_linear(model: nn.Module, X_probe: np.ndarray) -> nn.Linear:
    """Return the last nn.Linear that is actually called during a forward pass.

    Uses a probe forward pass so that decoder-only layers (e.g. MIN2Net's
    reconstruction head, which is skipped in eval mode) are not mistakenly
    selected over the true classification head.
    """
    all_linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not all_linears:
        raise RuntimeError("No nn.Linear layer found for rho-loss scoring.")

    last_fired: list = [None]

    def _make_hook(layer):
        def _hook(_module, _inp, _out):
            last_fired[0] = layer
        return _hook

    handles = [lin.register_forward_hook(_make_hook(lin)) for lin in all_linears]
    try:
        model.eval()
        device = next(model.parameters()).device
        x = torch.from_numpy(X_probe[:1].astype(np.float32)).to(device)
        with torch.no_grad():
            _extract_logits(model(x))
    finally:
        for h in handles:
            h.remove()

    if last_fired[0] is None:
        raise RuntimeError("No nn.Linear layer was active during forward pass.")
    return last_fired[0]


@torch.no_grad()
def _collect_last_linear_features_and_logits(model, X: np.ndarray, y: np.ndarray):
    """
    Collect penultimate features entering the final linear layer and logits.

    Returns
    -------
    feat : (N, D) float32
    logits : (N, K) float32
    y : (N,) int64
    """
    model.eval()
    last_linear = _find_last_linear(model, X)
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach())

    handle = last_linear.register_forward_hook(hook)
    try:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        feats, logits_all, labels = [], [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = _extract_logits(model(xb))
            if not captured:
                raise RuntimeError("Failed to capture last-layer features.")
            feat = captured.pop(0)
            feats.append(feat.cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            labels.append(yb.numpy())
    finally:
        handle.remove()

    return (
        np.concatenate(feats, axis=0).astype(np.float32),
        np.concatenate(logits_all, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.int64),
    )


def compute_rho_loss_scores(model, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Approximate reducible hold-out loss with cosine similarity on the final layer.

    We compute:
      score_i ≈ cos(∇_{W,b} L(x_i), ∇_{W,b} L_val)

    using the classifier's final linear layer only. This is much cheaper than
    full-parameter per-trial gradients and still aligns trial ranking with
    validation-subject improvement.
    """
    feat_tr, logits_tr, y_tr = _collect_last_linear_features_and_logits(model, X_train, y_train)
    feat_val, logits_val, yv = _collect_last_linear_features_and_logits(model, X_val, y_val)

    n_classes = logits_tr.shape[1]

    prob_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy().astype(np.float32)
    onehot_val = np.eye(n_classes, dtype=np.float32)[yv]
    delta_val = prob_val - onehot_val
    grad_w_val = (delta_val[:, :, None] * feat_val[:, None, :]).mean(axis=0)
    grad_b_val = delta_val.mean(axis=0)
    g_val = np.concatenate([grad_w_val.reshape(-1), grad_b_val.reshape(-1)], axis=0)
    g_val_norm = np.linalg.norm(g_val) + 1e-8

    prob_tr = torch.softmax(torch.from_numpy(logits_tr), dim=1).numpy().astype(np.float32)
    onehot_tr = np.eye(n_classes, dtype=np.float32)[y_tr]
    delta_tr = prob_tr - onehot_tr

    grad_w_tr = delta_tr[:, :, None] * feat_tr[:, None, :]
    grad_b_tr = delta_tr
    g_tr = np.concatenate(
        [grad_w_tr.reshape(len(X_train), -1), grad_b_tr.reshape(len(X_train), -1)],
        axis=1,
    )
    g_tr_norm = np.linalg.norm(g_tr, axis=1) + 1e-8
    cos = (g_tr @ g_val) / (g_tr_norm * g_val_norm)
    # Map to [0, 1] so that higher is always better and consistent with other selectors.
    return ((cos + 1.0) * 0.5).astype(np.float32)


def compute_dih_scores(model_name: str,
                       n_channels: int,
                       n_times: int,
                       n_filters: int,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       selector_epochs: int) -> np.ndarray:
    """
    Dynamic Instance Hardness (DIH) via epoch-averaged true-class uncertainty.

    For each trial, we track the selector model's true-class probability across
    training epochs and define hardness as:

      dih_i = mean_t (1 - p_t(y_i | x_i))

    Higher DIH means the sample stayed hard for longer during training.
    Scores are mapped to [0, 1] so they can flow through the existing
    selection / weighting pipeline unchanged.
    """
    model = build_model(model_name, n_channels, n_times, n_filters)
    init_model_for_fold(model, model_name, X_train, y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(selector_epochs, 1))

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()

    hardness_sum = np.zeros(len(y_train), dtype=np.float64)
    score_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    for _ in range(selector_epochs):
        train_epoch(model, train_loader, optimizer, criterion, augment=None)
        scheduler.step()

        model.eval()
        start = 0
        with torch.no_grad():
            for xb, yb in score_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = _extract_logits(model(xb))
                prob = torch.softmax(logits, dim=1)
                trueprob = prob.gather(1, yb.unsqueeze(1)).squeeze(1)
                batch_hardness = (1.0 - trueprob).cpu().numpy().astype(np.float64)
                end = start + len(batch_hardness)
                hardness_sum[start:end] += batch_hardness
                start = end

    scores = hardness_sum / max(selector_epochs, 1)
    scores = scores.astype(np.float32)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)


@torch.no_grad()
def compute_tangent_features(X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Compute tangent-space features from raw epochs.

    Core-set scoring is only a preprocessing step, so keep it on CPU to avoid
    CUDA instability in the SPD tangent extractor while leaving model training
    on GPU.
    """
    feature_device = torch.device("cpu")
    extractor = TangentSpaceExtractor().to(feature_device)
    extractor.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    feats = []
    for (xb,) in loader:
        xb = xb.to(feature_device)
        feats.append(extractor(xb).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def compute_coreset_scores(X_train: np.ndarray,
                           y_train: np.ndarray,
                           batch_size: int = 256) -> np.ndarray:
    """
    Rank samples by k-center greedy selection order on tangent-space features.

    We run k-center independently per class so that the downstream balanced
    top-k selection preserves the intended diversity within each label.
    Earlier-selected trials receive higher scores.
    """
    z_train = compute_tangent_features(X_train, batch_size=batch_size)
    scores = np.zeros(len(y_train), dtype=np.float32)

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        feats = z_train[cls_idx].astype(np.float32)
        n_cls = len(cls_idx)
        if n_cls == 0:
            continue
        if n_cls == 1:
            scores[cls_idx[0]] = 1.0
            continue

        norms = np.sum(feats * feats, axis=1, keepdims=True)
        d2 = np.maximum(norms + norms.T - 2.0 * (feats @ feats.T), 0.0)

        center = int(np.argmax(np.linalg.norm(feats - feats.mean(axis=0, keepdims=True), axis=1)))
        order = [center]
        selected = np.zeros(n_cls, dtype=bool)
        selected[center] = True
        min_dist = d2[center].copy()
        min_dist[center] = 0.0

        for _ in range(1, n_cls):
            candidate_scores = min_dist.copy()
            candidate_scores[selected] = -1.0
            nxt = int(np.argmax(candidate_scores))
            order.append(nxt)
            selected[nxt] = True
            min_dist = np.minimum(min_dist, d2[nxt])

        rank_scores = np.linspace(1.0, 0.0, num=n_cls, endpoint=True, dtype=np.float32)
        for rank, local_idx in enumerate(order):
            scores[cls_idx[local_idx]] = rank_scores[rank]

    return scores.astype(np.float32)


def compute_quality_diversity_scores(X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     ch_names: list,
                                     alpha: float = 0.3,
                                     batch_size: int = 256) -> np.ndarray:
    """
    Combine the existing combined quality score with tangent-space k-center rank.

    alpha=0 keeps the original combined quality score; larger alpha gives more
    weight to diversity so top-k selection is less likely to keep near-duplicates.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    quality = _minmax_scores(score_trials(X_train, ch_names, method="combined"))
    diversity = _minmax_scores(compute_coreset_scores(
        X_train, y_train, batch_size=batch_size
    ))
    return _minmax_scores((1.0 - alpha) * quality + alpha * diversity)


def compute_quality_entropy_scores(model,
                                   X_train_sel: np.ndarray,
                                   X_train_full: np.ndarray,
                                   ch_names: list,
                                   entropy_lambda: float = 0.2) -> np.ndarray:
    """
    Existing combined quality plus a small classifier-entropy bonus.

    The entropy term is deliberately additive and tunable so uncertain boundary
    trials can be retained without letting uncertainty dominate quality.
    """
    quality = _minmax_scores(score_trials(X_train_full, ch_names, method="combined"))
    uncertainty = _minmax_scores(compute_entropy_scores(model, X_train_sel))
    return _minmax_scores(quality + float(entropy_lambda) * uncertainty)


@torch.no_grad()
def compute_trueprob_scores(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convenience wrapper for true-class probabilities."""
    return compute_classifier_scores(model, X, y, method="trueprob")


def compute_learnability_scores(model_name: str,
                                n_channels: int,
                                n_times: int,
                                n_filters: int,
                                X_train_sel: np.ndarray,
                                y_train: np.ndarray,
                                X_train_full: np.ndarray,
                                X_train_prenorm: np.ndarray,
                                y_train_prenorm: np.ndarray,
                                ch_names: list,
                                warmup_epochs: int = 8,
                                learnability_lambda: float = 0.3) -> np.ndarray:
    """
    Learnability proxy from early supervised improvement on the train pool.

    We warm up the same classifier family for a small number of epochs and
    measure how quickly each trial's true-class probability improves.
    Samples that both improve quickly and end up reliable receive higher
    scores. The final score stays anchored to the existing combined quality
    heuristic so easy-to-overfit outliers do not dominate.
    """
    warmup_epochs = max(2, int(warmup_epochs))
    probe = build_model(model_name, n_channels, n_times, n_filters)
    init_model_for_fold(probe, model_name, X_train_prenorm, y_train_prenorm)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train_sel), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    first_trueprob = None
    prev_trueprob = None
    gain_accum = np.zeros(len(y_train), dtype=np.float32)

    for epoch in range(warmup_epochs):
        train_epoch(probe, train_loader, optimizer, criterion, augment=None)
        scheduler.step()
        trueprob = compute_trueprob_scores(probe, X_train_sel.astype(np.float32), y_train)
        if first_trueprob is None:
            first_trueprob = trueprob.copy()
        if prev_trueprob is not None:
            gain_accum += np.maximum(trueprob - prev_trueprob, 0.0)
        prev_trueprob = trueprob

    final_trueprob = prev_trueprob if prev_trueprob is not None else np.ones(len(y_train), dtype=np.float32)
    gain = gain_accum / max(warmup_epochs - 1, 1)
    learnability = 0.5 * _minmax_scores(final_trueprob) + 0.5 * _minmax_scores(gain)
    quality = _minmax_scores(score_trials(X_train_full, ch_names, method="combined"))
    return _minmax_scores(quality + float(learnability_lambda) * _minmax_scores(learnability))


def compute_quality_valgain_scores(model,
                                   X_train_sel: np.ndarray,
                                   y_train: np.ndarray,
                                   X_val_sel: np.ndarray,
                                   y_val: np.ndarray,
                                   X_train_full: np.ndarray,
                                   ch_names: list,
                                   valgain_lambda: float = 0.3) -> np.ndarray:
    """
    Combined heuristic quality plus validation-gain proxy from rho-loss cosine.

    The proxy is the final-layer gradient cosine with the validation loss,
    reusing the existing rho-loss machinery but treating it as a soft bonus
    instead of a hard ranking source.
    """
    quality = _minmax_scores(score_trials(X_train_full, ch_names, method="combined"))
    valgain = _minmax_scores(compute_rho_loss_scores(
        model,
        X_train_sel.astype(np.float32),
        y_train,
        X_val_sel.astype(np.float32),
        y_val,
    ))
    return _minmax_scores(quality + float(valgain_lambda) * valgain)


def train_tangent_autoencoder(z_train: np.ndarray, epochs: int = 50,
                              batch_size: int = 256, lr: float = 1e-3):
    """Train the tangent-space autoencoder for reconstruction-based quality.

    Normalises z_train to zero-mean unit-variance before training so that
    all dimensions contribute equally to the reconstruction loss.

    Returns (model, z_mean, z_std) so that compute_quality_labels can apply
    the same normalisation.
    """
    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std  = z_train.std(axis=0,  keepdims=True) + 1e-8
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)

    input_dim = z_norm.shape[1]
    hidden_dim = min(512, max(128, input_dim // 4))
    latent_dim = min(128, max(32, input_dim // 16))
    model = TangentAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.1,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for (zb,) in loader:
            zb = zb.to(DEVICE)
            optimizer.zero_grad()
            z_hat, _ = model(zb)
            loss = F.mse_loss(z_hat, zb)
            loss.backward()
            optimizer.step()
    return model, z_mean, z_std


@torch.no_grad()
def compute_quality_labels(autoencoder: TangentAutoEncoder, z_train: np.ndarray,
                           z_mean: np.ndarray, z_std: np.ndarray,
                           batch_size: int = 256) -> np.ndarray:
    """Compute soft quality labels from the trained autoencoder.

    z_train is normalised with z_mean / z_std before scoring so that
    the reconstruction error is on the same normalised scale.
    """
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)
    autoencoder.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    labels = []
    for (zb,) in loader:
        zb = zb.to(DEVICE)
        labels.append(autoencoder.quality_label(zb).cpu().numpy())
    return np.concatenate(labels, axis=0).astype(np.float32)


def compute_helpfulness_quality_labels(X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_train_prenorm: np.ndarray,
                                       y_train_prenorm: np.ndarray,
                                       n_channels: int,
                                       n_times: int,
                                       n_filters: int,
                                       epochs: int = 25) -> np.ndarray:
    """
    Approximate classifier-helpfulness with a short warm-up CSPNet.

    The proxy is the true-class probability after a brief supervised fit on the
    current training fold. It is not a full leave-one-out influence estimate,
    but it is aligned with the question "which trials become quickly useful for
    the MI classifier?" while staying cheap enough to run per LOSO fold.
    """
    probe = build_model("cspnet", n_channels, n_times, n_filters)
    init_model_for_fold(probe, "cspnet", X_train_prenorm, y_train_prenorm)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    probe.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()

    scores = compute_classifier_scores(probe, X_train, y_train, method="trueprob")
    return _minmax_scores(scores)


def compute_binary_domain_labels(X_train: np.ndarray,
                                  train_subj_ids: np.ndarray) -> tuple:
    """
    Cluster training subjects into 2 groups via k-means on mean tangent features.

    Returns (binary_labels, n_clusters=2) where binary_labels[i] ∈ {0, 1}
    is the cluster assignment for trial i, treating each subject's mean
    feature vector as the clustering input.
    """
    from sklearn.cluster import KMeans

    z_train = compute_tangent_features(X_train)
    unique_subjs = np.unique(train_subj_ids)

    subj_means = np.stack([
        z_train[train_subj_ids == s].mean(axis=0) for s in unique_subjs
    ]).astype(np.float32)

    km = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(subj_means)   # (n_subjects,) ∈ {0, 1}

    subj_to_cluster = {int(s): int(c) for s, c in zip(unique_subjs, cluster_labels)}
    binary_labels = np.array(
        [subj_to_cluster[int(s)] for s in train_subj_ids], dtype=np.int64
    )
    n0 = int((binary_labels == 0).sum())
    n1 = int((binary_labels == 1).sum())
    print(f"    [DANN/binary-domain] cluster 0: {n0} trials, cluster 1: {n1} trials",
          flush=True)
    return binary_labels, 2


def train_discriminator_selector(z_train: np.ndarray,
                                 quality_labels: np.ndarray,
                                 subject_ids: np.ndarray,
                                 epochs: int = 100,
                                 batch_size: int = 256,
                                 lr: float = 1e-3,
                                 lambda_q: float = 1.0,
                                 lambda_d: float = 1.0):
    """Train quality + GRL domain discriminator on tangent-space features.

    Normalise z_train internally so the AE/discriminator see unit-variance
    features regardless of the raw tangent-space scale.
    """
    # Normalise tangent features to zero-mean, unit-variance per dimension
    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std  = z_train.std(axis=0,  keepdims=True) + 1e-8
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)

    subj_unique = np.unique(subject_ids)
    subj_map = {sid: idx for idx, sid in enumerate(subj_unique.tolist())}
    subj_idx = np.array([subj_map[int(s)] for s in subject_ids], dtype=np.int64)

    input_dim = z_norm.shape[1]
    model = DiscriminatorSelectiveModel(
        input_dim=input_dim,
        n_subjects=len(subj_unique),
        hidden_dim=min(512, max(128, input_dim // 4)),
        embed_dim=min(256, max(64, input_dim // 8)),
        dropout=0.1,
        lambda_grl=1.0,   # GRL strength fixed at 1; lambda_d scales only the loss term
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_norm),                             # normalised
            torch.from_numpy(quality_labels.astype(np.float32)),
            torch.from_numpy(subj_idx),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for zb, qlb, sb in loader:
            zb = zb.to(DEVICE)
            qlb = qlb.to(DEVICE)
            sb = sb.to(DEVICE)
            optimizer.zero_grad()
            out = model(zb)
            losses = discriminator_losses(
                quality_logit=out["quality_logit"],
                quality_label=qlb,
                domain_logits=out["domain_logits"],
                subject_id=sb,
                lambda_q=lambda_q,
                lambda_d=lambda_d,
            )
            losses["total"].backward()
            optimizer.step()
    # Return model together with the normalisation stats so that
    # compute_discriminator_scores can apply the same transform.
    return model, z_mean, z_std


@torch.no_grad()
def compute_discriminator_scores(model: DiscriminatorSelectiveModel,
                                 z_train: np.ndarray,
                                 z_mean: np.ndarray,
                                 z_std: np.ndarray,
                                 method: str = "final",
                                 batch_size: int = 256) -> np.ndarray:
    """Score trials with discriminator outputs.

    z_train is normalised with the same z_mean / z_std used during training
    before being fed to the model.
    """
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    scores = []
    for (zb,) in loader:
        zb = zb.to(DEVICE)
        out = model(zb)
        if method == "quality":
            score = model.quality_score(out["quality_logit"])
        elif method == "domain":
            score = model.domain_score(out["domain_logits"])
        else:
            score = model.final_score(out["quality_logit"], out["domain_logits"])
        scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def _loss_acc(model, loader, criterion, n):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / n, correct / n


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL loss on feature covariances."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("coral_loss expects 2D feature tensors")
    d = source.size(1)
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cs = (source.T @ source) / max(source.size(0) - 1, 1)
    ct = (target.T @ target) / max(target.size(0) - 1, 1)
    return torch.norm(cs - ct, p="fro").pow(2) / (4.0 * (d ** 2))


def train_one_fold_contrastive(model_name, n_channels, n_times, n_filters,
                               X_train_prenorm, y_train_prenorm,
                               train_loader, val_loader, test_loader,
                               y_val, y_test, epochs=EPOCHS,
                               lambda_con=0.5, stage1_epochs=150,
                               temperature=0.07,
                               lambda_coral=0.0,
                               target_domain_loader=None,
                               feature_mixup_alpha=0.0,
                               lambda_feature_mixup=0.0,
                               label_smoothing=0.0):
    """
    Two-stage contrastive training loop.

    Stage 1 (stage1_epochs): encoder + MI head only.
    Stage 2 (epochs - stage1_epochs): MI + SupCon.
      No encoder freeze — contrastive is complementary, not adversarial.

    Joint loss (stage 2): L = L_cls + lambda_con * L_supcon(h, y)
                                 + lambda_coral * L_coral(z_src, z_tgt)
    train_loader batches: (X, y) or (X, y, sample_weight).
    """
    model = build_model(model_name, n_channels, n_times, n_filters,
                        dann_kwargs={"temperature": temperature})
    init_model_for_fold(model, model_name, X_train_prenorm, y_train_prenorm)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_log = []
    stage2_total = max(epochs - stage1_epochs, 1)

    print(f"    [Contrastive] stage1={stage1_epochs} ep (MI only) | "
          f"stage2={stage2_total} ep (MI+SupCon λ={lambda_con}, CORAL λ={lambda_coral}, "
          f"FeatMix λ={lambda_feature_mixup})", flush=True)

    # ── Stage 1: MI head only ─────────────────────────────────────────────────
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            z = model._encode(xb.unsqueeze(1))
            logits = model.classifier(z)
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)
        scheduler.step()
        model.eval()
        val_loss, val_acc   = _loss_acc(model, val_loader,  criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))
        epoch_log.append(dict(
            epoch=epoch,
            train_loss=round(total_loss / n, 6), train_acc=round(correct / n, 4),
            val_loss=round(val_loss, 6),  val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6),test_acc=round(test_acc, 4),
        ))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Stage 2: MI + SupCon, full encoder trainable ──────────────────────────
    optimizer2 = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=stage2_total
    )

    target_iter = iter(target_domain_loader) if target_domain_loader is not None else None
    for s2_epoch in range(1, stage2_total + 1):
        global_epoch = stage1_epochs + s2_epoch
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, sample_weight = batch
                sample_weight = sample_weight.to(DEVICE)
            else:
                xb, yb = batch
                sample_weight = None
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer2.zero_grad()
            z_src = model._encode(xb.unsqueeze(1))
            logits = model.classifier(z_src)
            h = F.normalize(model.proj_head(z_src), dim=1)
            cls_loss_vec = F.cross_entropy(
                logits, yb, reduction="none", label_smoothing=label_smoothing
            )
            cls_loss = _apply_sample_weights(cls_loss_vec, sample_weight)
            con_loss = model.supcon(h, yb)
            align_loss = z_src.new_zeros(())
            if lambda_coral > 0.0 and target_iter is not None:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_domain_loader)
                    target_batch = next(target_iter)
                x_tgt = target_batch[0].to(DEVICE)
                z_tgt = model._encode(x_tgt.unsqueeze(1))
                align_loss = coral_loss(z_src, z_tgt)
            mix_loss = z_src.new_zeros(())
            if lambda_feature_mixup > 0:
                candidate = _same_class_feature_mixup_loss(
                    model, xb, yb, feature_mixup_alpha, label_smoothing
                )
                if candidate is not None:
                    mix_loss = candidate
            loss = (cls_loss + lambda_con * con_loss + lambda_coral * align_loss
                    + lambda_feature_mixup * mix_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)
        scheduler2.step()
        model.eval()
        val_loss, val_acc   = _loss_acc(model, val_loader,  criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))
        epoch_log.append(dict(
            epoch=global_epoch,
            train_loss=round(total_loss / n, 6), train_acc=round(correct / n, 4),
            val_loss=round(val_loss, 6),  val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6),test_acc=round(test_acc, 4),
        ))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state or dict(model.state_dict()))
    return model, criterion, epoch_log


def _dann_train_epoch_stage1(model, train_loader, optimizer):
    """Stage 1: MI head only (encoder + classifier, no quality/domain heads)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in train_loader:
        xb, yb, ql, sid = batch
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        # Bypass auxiliary heads: directly call encoder + classifier
        z = model._encode(xb.unsqueeze(1))
        logits = model.classifier(z)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


def train_one_fold_dann(model_name, n_channels, n_times, n_filters, n_subjects,
                        X_train_prenorm, y_train_prenorm,
                        train_loader, val_loader, test_loader,
                        y_val, y_test, epochs=EPOCHS,
                        lambda_q=0.5, lambda_d=0.1,
                        stage1_epochs=150, domain_delay=10,
                        grl_exponent=5.0,
                        dann_mode="full", freeze_encoder=True,
                        target_domain_loader=None,
                        dann_domain_target="subject",
                        dann_use_grl=True,
                        dann_stage2_weighting="none",
                        dann_stage2_min_weight=0.25,
                        dann_model_kwargs=None):
    """
    Two-stage DANN training loop with ablation mode control.

    dann_mode
    ---------
    'full'         : quality + domain heads (default)
    'quality_only' : quality head only  (lambda_d forced to 0)
    'domain_only'  : domain head only   (lambda_q forced to 0)
    'no_aux'       : MI head only in both stages (two-stage effect isolation)

    freeze_encoder : if True, temporal_conv + csp_layer frozen in stage 2.
                     Set False to ablate the freeze contribution.

    train_loader batches: (X, y, quality_label, subject_id)
    """
    # Apply mode → effective lambdas
    eff_lq = lambda_q
    eff_ld = lambda_d
    if dann_mode == "quality_only":
        eff_ld = 0.0
    elif dann_mode == "domain_only":
        eff_lq = 0.0
    elif dann_mode == "no_aux":
        eff_lq = 0.0
        eff_ld = 0.0

    model = build_model(
        model_name, n_channels, n_times, n_filters,
        n_subjects=n_subjects,
        dann_kwargs=dann_model_kwargs,
    )
    init_model_for_fold(model, model_name, X_train_prenorm, y_train_prenorm)
    model.set_grl_lambda(0.0)
    model.use_grl = bool(dann_use_grl)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc, best_state = -1.0, None
    epoch_log = []
    criterion = nn.CrossEntropyLoss()

    stage2_total = max(epochs - stage1_epochs, 1)
    freeze_tag = "freeze" if freeze_encoder else "no-freeze"
    if stage1_epochs > 0:
        print(f"    [DANN/{dann_mode}/{freeze_tag}] "
              f"stage1={stage1_epochs} ep (MI only) | "
              f"stage2={stage2_total} ep "
              f"(λ_q={eff_lq} λ_d={eff_ld} delay={domain_delay} γ={grl_exponent})",
              flush=True)

    # ── Stage 1: MI head only ─────────────────────────────────────────────────
    for epoch in range(1, stage1_epochs + 1):
        tr_loss, tr_acc = _dann_train_epoch_stage1(model, train_loader, optimizer)
        scheduler.step()
        model.eval()
        val_loss, val_acc   = _loss_acc(model, val_loader,  criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))
        epoch_log.append(dict(
            epoch=epoch,
            train_loss=round(tr_loss, 6), train_acc=round(tr_acc, 4),
            val_loss=round(val_loss, 6),  val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6),test_acc=round(test_acc, 4),
        ))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Optionally freeze early encoder layers before stage 2 ────────────────
    if stage1_epochs > 0 and freeze_encoder:
        for param in model.temporal_conv.parameters():
            param.requires_grad = False
        for param in model.csp_layer.parameters():
            param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage2_total
        )
    elif stage1_epochs > 0:
        # No freeze: rebuild optimizer/scheduler scoped to stage 2 length
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage2_total
        )

    # ── Stage 2: auxiliary heads with weakly adversarial GRL ─────────────────
    target_iter = None
    for s2_epoch in range(1, stage2_total + 1):
        global_epoch = stage1_epochs + s2_epoch

        # Slow GRL ramp, delayed by domain_delay
        if eff_ld == 0.0 or s2_epoch <= domain_delay:
            lam = 0.0
        else:
            adj = (s2_epoch - domain_delay) / max(stage2_total - domain_delay, 1)
            lam = grl_lambda(adj, exponent=grl_exponent)
        model.set_grl_lambda(lam)

        model.train()
        total_loss, correct, n = 0.0, 0, 0
        if target_domain_loader is not None:
            target_iter = iter(target_domain_loader)
        for batch in train_loader:
            xb, yb, ql, sid = batch
            xb  = xb.to(DEVICE)
            yb  = yb.to(DEVICE)
            ql  = ql.to(DEVICE)
            sid = sid.to(DEVICE)
            optimizer.zero_grad()

            sample_weight = None
            if dann_stage2_weighting == "quality":
                sample_weight = dann_stage2_min_weight + (1.0 - dann_stage2_min_weight) * ql.float()

            if eff_lq == 0.0 and eff_ld == 0.0:
                # no_aux mode: MI head only (same as stage 1)
                z = model._encode(xb.unsqueeze(1))
                logits = model.classifier(z)
                cls_loss = F.cross_entropy(logits, yb, reduction="none")
                loss = _apply_sample_weights(cls_loss, sample_weight)
            else:
                if eff_ld > 0.0 and dann_domain_target == "binary":
                    z_src = model._encode(xb.unsqueeze(1))
                    logits = model.classify_from_features(z_src)
                    q_logit = model.quality_from_features(z_src)

                    try:
                        (x_tgt,) = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_domain_loader)
                        (x_tgt,) = next(target_iter)
                    x_tgt = x_tgt.to(DEVICE)
                    z_tgt = model._encode(x_tgt.unsqueeze(1))
                    dom_in = torch.cat([z_src, z_tgt], dim=0)
                    d_logits = model.domain_from_features(dom_in)
                    d_labels = torch.cat([
                        torch.zeros(len(z_src), dtype=torch.long, device=DEVICE),
                        torch.ones(len(z_tgt), dtype=torch.long, device=DEVICE),
                    ], dim=0)
                    loss, _ = dann_loss(
                        logits, yb, q_logit, ql, d_logits, d_labels,
                        lambda_q=eff_lq, lambda_d=eff_ld, sample_weight=sample_weight,
                    )
                else:
                    logits, q_logit, d_logits = model(xb)
                    loss, _ = dann_loss(
                        logits, yb, q_logit, ql, d_logits, sid,
                        lambda_q=eff_lq, lambda_d=eff_ld, sample_weight=sample_weight,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)

        scheduler.step()
        tr_loss = total_loss / n
        tr_acc  = correct / n

        model.eval()
        val_loss, val_acc   = _loss_acc(model, val_loader,  criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))
        epoch_log.append(dict(
            epoch=global_epoch,
            train_loss=round(tr_loss, 6), train_acc=round(tr_acc, 4),
            val_loss=round(val_loss, 6),  val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6),test_acc=round(test_acc, 4),
        ))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state or dict(model.state_dict()))
    return model, criterion, epoch_log


def train_one_fold_model(model_name, n_channels, n_times, n_filters,
                         X_train_prenorm, y_train_prenorm,
                         train_loader, val_loader, test_loader,
                         y_val, y_test, augment=None, epochs=EPOCHS,
                         feature_mixup_alpha=0.0, lambda_feature_mixup=0.0,
                         label_smoothing=0.0,
                         use_snapshot=False, snapshot_T0=50,
                         use_sam=False, sam_rho=0.05,
                         style_aug=None):
    """Train a fold model and return the best checkpoint and logs.

    When use_snapshot=True: uses CosineAnnealingWarmRestarts with period
    snapshot_T0 and saves a checkpoint at each cycle end.
    Returns (model, criterion, epoch_log, snapshot_states).
    snapshot_states is None when use_snapshot=False.
    """
    model = build_model(model_name, n_channels, n_times, n_filters)
    init_model_for_fold(model, model_name, X_train_prenorm, y_train_prenorm)

    if use_sam:
        optimizer = SAM(
            model.parameters(), torch.optim.Adam,
            rho=sam_rho, lr=LR, weight_decay=WEIGHT_DECAY
        )
        _sched_opt = optimizer.base_optimizer
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        _sched_opt = optimizer

    if use_snapshot:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _sched_opt, T_0=snapshot_T0, T_mult=1, eta_min=LR * 1e-2
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            _sched_opt, T_max=epochs
        )
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    snapshot_states = [] if use_snapshot else None
    epoch_log = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, augment=augment,
            feature_mixup_alpha=feature_mixup_alpha,
            lambda_feature_mixup=lambda_feature_mixup,
            label_smoothing=label_smoothing,
            use_sam=use_sam,
            style_aug=style_aug,
        )
        scheduler.step()

        val_loss, val_acc = _loss_acc(model, val_loader, criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))

        epoch_log.append(dict(
            epoch=epoch,
            train_loss=round(tr_loss, 6), train_acc=round(tr_acc, 4),
            val_loss=round(val_loss, 6), val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6), test_acc=round(test_acc, 4),
        ))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Collect snapshot at end of each cosine cycle
        if use_snapshot and epoch % snapshot_T0 == 0:
            snapshot_states.append({k: v.clone() for k, v in model.state_dict().items()})

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, criterion, epoch_log, snapshot_states


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        pred = model(xb).argmax(1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(yb.numpy())
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc  = accuracy_score(y_true, y_pred)
    bac  = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, bac, kappa, y_true, y_pred


@torch.no_grad()
def evaluate_ensemble(snapshot_states, model_name, n_channels, n_times, n_filters, loader):
    """Average softmax over snapshot checkpoints, return (acc, bac, kappa, y_true, y_pred)."""
    all_probs, all_true = None, None
    for state in snapshot_states:
        m = build_model(model_name, n_channels, n_times, n_filters)
        m.load_state_dict(state)
        m.eval()
        probs_list, true_list = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = torch.softmax(m(xb), dim=1).cpu().numpy()
            probs_list.append(p)
            true_list.extend(yb.numpy())
        probs = np.concatenate(probs_list, axis=0)
        all_probs = probs if all_probs is None else all_probs + probs
        if all_true is None:
            all_true = np.array(true_list)
    y_pred = all_probs.argmax(1)
    acc   = accuracy_score(all_true, y_pred)
    bac   = balanced_accuracy_score(all_true, y_pred)
    kappa = cohen_kappa_score(all_true, y_pred)
    return acc, bac, kappa, all_true, y_pred


@torch.no_grad()
def evaluate_ensemble_adabn(snapshot_states, model_name, n_channels, n_times,
                            n_filters, loader, X_test, adabn_passes=3):
    """Apply AdaBN to each snapshot independently, then ensemble softmax."""
    all_probs, all_true = None, None
    for state in snapshot_states:
        m = build_model(model_name, n_channels, n_times, n_filters)
        m.load_state_dict(state)
        apply_adabn(m, X_test, DEVICE, batch_size=BATCH_SIZE, n_passes=adabn_passes)
        m.eval()
        probs_list, true_list = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = torch.softmax(m(xb), dim=1).cpu().numpy()
            probs_list.append(p)
            true_list.extend(yb.numpy())
        probs = np.concatenate(probs_list, axis=0)
        all_probs = probs if all_probs is None else all_probs + probs
        if all_true is None:
            all_true = np.array(true_list)
    y_pred = all_probs.argmax(1)
    acc   = accuracy_score(all_true, y_pred)
    bac   = balanced_accuracy_score(all_true, y_pred)
    kappa = cohen_kappa_score(all_true, y_pred)
    return acc, bac, kappa, all_true, y_pred


@torch.no_grad()
def evaluate_uncertainty(model, loader, threshold: float = 0.9, n_bins: int = 10):
    """Compute confidence calibration and selective-prediction metrics."""
    model.eval()
    all_prob, all_true = [], []
    for xb, yb in loader:
        logits = _extract_logits(model(xb.to(DEVICE)))
        all_prob.append(torch.softmax(logits, dim=1).cpu())
        all_true.append(yb)
    prob = torch.cat(all_prob)
    true = torch.cat(all_true)
    conf, pred = prob.max(dim=1)
    correct = pred.eq(true).float()
    ece = torch.zeros(())
    bounds = torch.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        in_bin = (conf > lower) & (conf <= upper)
        if in_bin.any():
            ece += in_bin.float().mean() * (
                correct[in_bin].mean() - conf[in_bin].mean()
            ).abs()
    one_hot = F.one_hot(true, num_classes=prob.shape[1]).float()
    brier = (prob - one_hot).pow(2).sum(dim=1).mean()
    accepted = conf >= threshold
    coverage = accepted.float().mean()
    selective_acc = correct[accepted].mean() if accepted.any() else torch.tensor(float("nan"))
    return (float(ece), float(brier), float(coverage), float(selective_acc))


@torch.no_grad()
def evaluate_conformal(model, calibration_loader, test_loader, alpha: float = 0.1):
    """Split conformal prediction sets calibrated on the held-out validation subject."""
    model.eval()
    cal_scores = []
    for xb, yb in calibration_loader:
        prob = torch.softmax(_extract_logits(model(xb.to(DEVICE))), dim=1).cpu()
        cal_scores.append(1.0 - prob.gather(1, yb.unsqueeze(1)).squeeze(1))
    scores = torch.cat(cal_scores).numpy()
    n_cal = len(scores)
    rank = min(int(np.ceil((n_cal + 1) * (1.0 - alpha))), n_cal)
    qhat = float(np.sort(scores)[rank - 1])

    covered, sizes, n_test = 0, 0, 0
    for xb, yb in test_loader:
        prob = torch.softmax(_extract_logits(model(xb.to(DEVICE))), dim=1).cpu()
        pred_set = prob >= (1.0 - qhat)
        covered += pred_set.gather(1, yb.unsqueeze(1)).sum().item()
        sizes += pred_set.sum().item()
        n_test += len(yb)
    return covered / n_test, sizes / n_test, qhat


def adapt_pseudo_label_classifier(model, X_target: np.ndarray, threshold: float = 0.9,
                                  epochs: int = 10, lr: float = 1e-4):
    """Adapt only the final classifier on high-confidence target predictions."""
    adapted = copy.deepcopy(model)
    if not hasattr(adapted, "classifier"):
        return None, 0
    adapted.eval()
    target = torch.from_numpy(X_target.astype(np.float32))
    loader = DataLoader(TensorDataset(target), batch_size=BATCH_SIZE, shuffle=False)
    selected_x, selected_y = [], []
    with torch.no_grad():
        for (xb,) in loader:
            logits = _extract_logits(adapted(xb.to(DEVICE)))
            prob, pseudo = torch.softmax(logits, dim=1).max(dim=1)
            keep = prob >= threshold
            if keep.any():
                selected_x.append(xb[keep.cpu()])
                selected_y.append(pseudo[keep].cpu())
    if not selected_x:
        return None, 0
    for param in adapted.parameters():
        param.requires_grad_(False)
    for param in adapted.classifier.parameters():
        param.requires_grad_(True)
    pseudo_ds = TensorDataset(torch.cat(selected_x), torch.cat(selected_y))
    pseudo_loader = DataLoader(pseudo_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(adapted.classifier.parameters(), lr=lr)
    adapted.eval()
    for _ in range(epochs):
        for xb, yb in pseudo_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            feat, classifier = _classifier_features(adapted, xb)
            if feat is None:
                return None, 0
            logits = classifier(feat.detach())
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
    adapted.eval()
    return adapted, len(pseudo_ds)


# -----------------------------------------------------------------------------
# LOSO loop
# -----------------------------------------------------------------------------

def run_loso(dataset_name: str, ch_filter: str = None, model_name: str = "spdnet",
             use_augment: bool = False, resume: bool = False,
             out_dir: str = RESULTS_DIR, run_id: str = None,
             keep_ratio: float = 1.0, score_method: str = "combined",
             selection_mode: str = "hard", min_weight: float = 0.25,
             selection_source: str = "heuristic", selector_epochs: int = 100,
             lambda_q: float = 0.5, lambda_d: float = 0.1,
             lambda_con: float = 0.5, lambda_coral: float = 0.0,
             feature_mixup_alpha: float = 0.0,
             lambda_feature_mixup: float = 0.0,
             label_smoothing: float = 0.0,
             diversity_alpha: float = 0.3, uncertainty_lambda: float = 0.2,
             learnability_lambda: float = 0.3, learnability_epochs: int = 8,
             valgain_lambda: float = 0.3,
             save_selection_plots_flag: bool = False,
             selection_plot_max_points: int = 4000,
             dann_stage1_epochs: int = 150, dann_domain_delay: int = 10,
             dann_grl_exponent: float = 5.0,
             dann_mode: str = "full", dann_freeze_encoder: bool = True,
             dann_quality_mode: str = "helpfulness",
             dann_domain_mode: str = "binary",
             dann_helpfulness_epochs: int = 25,
             temperature: float = 0.07,
             use_adabn: bool = False,
             adabn_passes: int = 3,
             use_ea: bool = False,
             use_tent: bool = False,
             tent_steps: int = 1,
             tent_lr: float = 1e-3,
             use_calibration: bool = False,
             reject_threshold: float = 0.9,
             conformal_alpha: float = 0.1,
             use_pseudo_label: bool = False,
             pseudo_threshold: float = 0.9,
             pseudo_epochs: int = 10,
             pseudo_lr: float = 1e-4,
             use_snapshot: bool = False,
             snapshot_T0: int = 50,
             use_fbcsp: bool = False,
             fbcsp_bands: list = None,
             use_subject_weight: bool = False,
             subject_weight_tau: float = 1.0,
             use_sam: bool = False,
             sam_rho: float = 0.05,
             use_style_aug: bool = False,
             style_aug_p: float = 0.5,
             style_aug_alpha: float = 1.0,
             ea_eps_mode: str = None,
             checkpoint_dir: str = None,
             max_test_subjects: int = None,
             split_seed: int = SPLIT_SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tag = f" [ch_filter='{ch_filter}']" if ch_filter else ""
    aug_tag = " +aug" if use_augment else ""
    sel_tag = ""
    if keep_ratio < 1.0:
        sel_tag = (f" [sel={selection_source}/{selection_mode}:"
                   f"{int(keep_ratio*100)}% {score_method}]")
    if lambda_feature_mixup > 0:
        sel_tag += f" [feature-mixup lambda={lambda_feature_mixup}]"
    if use_pseudo_label:
        sel_tag += f" [pseudo-label threshold={pseudo_threshold}]"
    print(f"\n{'='*60}")
    print(f" LOSO - {dataset_name.upper()}{tag}{aug_tag}{sel_tag}")
    print(f"{'='*60}")

    X, y, subjects, ch_names, sfreq = load_data(dataset_name, ch_filter=ch_filter)
    all_subj_ids = np.unique(subjects)
    subj_ids = all_subj_ids
    if max_test_subjects is not None:
        subj_ids = all_subj_ids[:max_test_subjects]
        print(f"  Smoke limit: evaluating {len(subj_ids)} subjects {subj_ids.tolist()}")
    n_channels = X.shape[1]
    n_filters  = min(N_FILTERS, n_channels)   # BiMap: c_out must be <= c_in
    if n_filters != N_FILTERS:
        print(f"  n_filters capped to {n_filters} (n_channels={n_channels})")

    # -- Subject Clustering: pre-EA covariances (must come before EA transforms X) --
    subj_mean_covs = None
    if use_subject_weight:
        from pyriemann.utils.covariance import covariances as _py_covs
        from pyriemann.utils.distance import distance_riemann as _dist_riemann
        print("  Subject Clustering: computing pre-EA covariances...", flush=True)
        subj_mean_covs = {}
        for s in all_subj_ids:
            X_s = X[subjects == s]
            covs_s = _py_covs(X_s, estimator='oas')
            subj_mean_covs[s] = covs_s.mean(axis=0)
        print(f"  Subject Clustering: {len(subj_mean_covs)} subject covariances ready.")

    # -- Style Aug: pre-EA raw data kept for covariance computation -----------
    X_raw_for_style = X.copy() if use_style_aug else None
    if use_style_aug:
        print(f"  Covariance Style Aug: p={style_aug_p}, alpha={style_aug_alpha}", flush=True)

    # -- Euclidean Alignment (per-subject, before any split) ------------------
    if use_ea:
        print(f"  Euclidean Alignment: aligning {len(all_subj_ids)} subjects "
              f"(eps_mode={ea_eps_mode})...", flush=True)
        X = apply_ea_loso(X, subjects, eps_mode=ea_eps_mode)
        print(f"  EA done.")

    # -- FBCSP: concatenate band-filtered signals as extra channels -----------
    if use_fbcsp:
        from scipy.signal import butter, sosfilt
        bands = fbcsp_bands or [(8, 12), (13, 30)]
        print(f"  FBCSP: filtering into {bands} Hz bands + concatenating...", flush=True)
        X_bands = [X]
        for (lo, hi) in bands:
            sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
            Xf = sosfilt(sos, X, axis=-1).astype(np.float32)
            X_bands.append(Xf)
        X = np.concatenate(X_bands, axis=1)  # (N, (1+n_bands)*C, T)
        n_channels = X.shape[1]
        n_filters  = min(N_FILTERS, n_channels)
        print(f"  FBCSP done: n_channels={n_channels}")

    # -- Augmentation --------------------------------------------------------
    augment = None
    if use_augment:
        augment = EEGAugment(p=0.5, jitter_ms=50.0, sfreq=sfreq,
                             amp_range=(0.8, 1.2), noise_std=0.05).to(DEVICE)
        print(f"  Augmentation: {augment}")

    results   = []  # per-subject summary
    rng = np.random.default_rng(split_seed)

    import time
    t_total_start = time.time()

    # -- Incremental CSV setup ------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    ch_tag2   = f"_ch{ch_filter}" if ch_filter else ""
    aug_tag2  = "_aug" if use_augment else ""
    sel_tag2  = (f"_sel{selection_source}{selection_mode}{int(keep_ratio*100)}{score_method}"
                 if keep_ratio < 1.0 else "")
    mix_tag2 = f"_featmix{lambda_feature_mixup:g}" if lambda_feature_mixup > 0 else ""
    smooth_tag2 = f"_ls{label_smoothing:g}" if label_smoothing > 0 else ""
    pseudo_tag2 = f"_pseudo{pseudo_threshold:g}" if use_pseudo_label else ""
    cal_tag2 = f"_cal{reject_threshold:g}" if use_calibration else ""
    base_tag  = (ch_tag2 + f"_{model_name}" + aug_tag2 + sel_tag2 + mix_tag2
                 + smooth_tag2 + pseudo_tag2 + cal_tag2)
    res_fields  = ["dataset", "subject", "n_train", "n_test",
                   "acc", "bac", "kappa", "best_epoch", "best_val_acc",
                   "best_val_loss", "time_min",
                   "adabn_acc", "adabn_bac", "adabn_kappa",
                   "tent_acc", "tent_bac", "tent_kappa"]
    if use_calibration:
        res_fields.extend(["ece", "brier", "reject_coverage", "reject_acc",
                           "conformal_coverage", "conformal_set_size", "conformal_qhat"])
    if use_pseudo_label:
        res_fields.extend(["pseudo_n", "pseudo_acc", "pseudo_bac", "pseudo_kappa"])
    if use_snapshot:
        res_fields.extend(["snap_acc", "snap_bac", "snap_kappa",
                           "snap_adabn_acc", "snap_adabn_bac", "snap_adabn_kappa"])
    loss_fields = ["dataset", "subject", "epoch",
                   "train_loss", "train_acc",
                   "val_loss",   "val_acc",
                   "test_loss",  "test_acc"]

    done_subjects = set()
    res_csv = None
    loss_csv = None

    if resume:
        if run_id:
            res_csv  = os.path.join(out_dir, f"loso_results_{run_id}{base_tag}.csv")
            loss_csv = os.path.join(out_dir, f"loso_loss_{run_id}{base_tag}.csv")
            if not os.path.exists(res_csv):
                raise FileNotFoundError(
                    f"--resume requested but results not found: {res_csv}"
                )
        else:
            pattern = os.path.join(out_dir, f"loso_results_*{base_tag}.csv")
            existing = sorted(glob.glob(pattern))
            if existing:
                res_csv  = existing[-1]
                loss_csv = res_csv.replace("loso_results_", "loso_loss_")
            else:
                print("  --resume: no existing CSV found, starting fresh.")
                resume = False

    if not resume:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_csv  = os.path.join(out_dir, f"loso_results_{run_id}{base_tag}.csv")
        loss_csv = os.path.join(out_dir, f"loso_loss_{run_id}{base_tag}.csv")
        if os.path.exists(res_csv) or os.path.exists(loss_csv):
            # Allow appending when the same run_id is reused (e.g., dataset=both)
            resume = True
        else:
            with open(res_csv,  "w", newline="") as f:
                csv.DictWriter(f, fieldnames=res_fields).writeheader()
            with open(loss_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=loss_fields).writeheader()

    # If resuming, load completed subjects
    if resume and res_csv:
        with open(res_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset_name:
                    try:
                        done_subjects.add(int(row["subject"]))
                    except Exception:
                        pass
        # Ensure loss file has a header
        if loss_csv and (not os.path.exists(loss_csv)):
            with open(loss_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=loss_fields).writeheader()
        print(f"  Resuming: {len(done_subjects)} subjects done -> {res_csv}")

    print(f"  Results -> {res_csv}")

    selection_plot_dir = os.path.join(out_dir, f"selection_plots_{run_id}{base_tag}")
    if save_selection_plots_flag:
        os.makedirs(selection_plot_dir, exist_ok=True)

    for i, test_subj in enumerate(subj_ids):
        if test_subj in done_subjects:
            print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} -- skipped (already done)")
            continue
        t_subj_start = time.time()
        # -- Split: test | val (1 random train subj) | train (rest) --------
        train_pool = all_subj_ids[all_subj_ids != test_subj]
        val_subj   = rng.choice(train_pool)

        test_mask  = subjects == test_subj
        val_mask   = subjects == val_subj
        train_mask = ~test_mask & ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        train_subjects = subjects[train_mask]

        n_before = len(y_train)
        X_train_full = X_train.copy()
        y_train_full = y_train.copy()
        train_weights = None

        # -- Covariance Style Transfer augmentation (per-fold) ----------------
        fold_style_aug = None
        if use_style_aug and X_raw_for_style is not None:
            train_subj_ids = np.unique(train_subjects)
            fold_style_aug = build_style_aug(
                X_raw_for_style, subjects, train_subj_ids,
                p=style_aug_p, alpha=style_aug_alpha
            ).to(DEVICE)

        # -- Subject Clustering: weight training samples by Riemannian similarity --
        if use_subject_weight and subj_mean_covs is not None:
            test_cov = subj_mean_covs[test_subj]
            unique_train_subjs = np.unique(train_subjects)
            dists = np.array([_dist_riemann(subj_mean_covs[s], test_cov)
                              for s in unique_train_subjs])
            raw_w = np.exp(-dists / subject_weight_tau)
            raw_w /= raw_w.sum()
            subj_w_map = {int(s): float(w) for s, w in zip(unique_train_subjs, raw_w)}
            train_weights = np.array([subj_w_map[int(s)] for s in train_subjects],
                                     dtype=np.float32)
            print(f"    [SubjW tau={subject_weight_tau}] "
                  f"dist: [{dists.min():.3f}, {dists.max():.3f}]  "
                  f"w: [{train_weights.min():.4f}, {train_weights.max():.4f}]")

        # Keep pre-normalised copy for CSP filter fitting and classifier-first scoring
        X_train_prenorm = X_train.copy()
        y_train_prenorm = y_train.copy()

        if keep_ratio < 1.0:
            mu_sel = X_train_full.mean(axis=(0, 2), keepdims=True)
            std_sel = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
            X_train_sel = (X_train_full - mu_sel) / std_sel
            X_val_sel = (X_val - mu_sel) / std_sel
            X_test_sel = (X_test - mu_sel) / std_sel

            if selection_source in ("classifier", "rho_loss", "uncertainty", "val_gain"):
                # Stage 1: train a baseline classifier on the full train pool.

                selector_train_ds = TensorDataset(
                    torch.from_numpy(X_train_sel), torch.from_numpy(y_train_full)
                )
                selector_val_ds = TensorDataset(
                    torch.from_numpy(X_val_sel), torch.from_numpy(y_val)
                )
                selector_test_ds = TensorDataset(
                    torch.from_numpy(X_test_sel), torch.from_numpy(y_test)
                )

                selector_train_loader = DataLoader(selector_train_ds, batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=False)
                selector_val_loader = DataLoader(selector_val_ds, batch_size=BATCH_SIZE,
                                                 shuffle=False, drop_last=False)
                selector_test_loader = DataLoader(selector_test_ds, batch_size=BATCH_SIZE,
                                                  shuffle=False, drop_last=False)

                stage_name = selection_source if selection_source != "classifier" else "classifier"
                print(f"    [sel-{stage_name}] training selector model "
                      f"(epochs={selector_epochs}, score={score_method})")
                selector_model_name = "cspnet" if model_name == "cspnetcontrastive" else model_name
                selector_model, _, _ = train_one_fold_model(
                    model_name=selector_model_name,
                    n_channels=n_channels,
                    n_times=X.shape[2],
                    n_filters=n_filters,
                    X_train_prenorm=X_train_full,
                    y_train_prenorm=y_train_full,
                    train_loader=selector_train_loader,
                    val_loader=selector_val_loader,
                    test_loader=selector_test_loader,
                    y_val=y_val,
                    y_test=y_test,
                    augment=None,
                    epochs=selector_epochs,
                )
                if selection_source == "rho_loss":
                    if score_method != "rho_cosine":
                        print(f"    [WARN] score_method='{score_method}' not valid for "
                              f"rho_loss; using 'rho_cosine'")
                    scores = compute_rho_loss_scores(
                        selector_model,
                        X_train_sel.astype(np.float32),
                        y_train_full,
                        X_val_sel.astype(np.float32),
                        y_val,
                    )
                elif selection_source == "uncertainty":
                    if score_method != "quality_entropy":
                        print(f"    [WARN] score_method='{score_method}' not valid for "
                              f"uncertainty; using 'quality_entropy'")
                    scores = compute_quality_entropy_scores(
                        selector_model,
                        X_train_sel.astype(np.float32),
                        X_train_full.astype(np.float32),
                        ch_names,
                        entropy_lambda=uncertainty_lambda,
                    )
                elif selection_source == "val_gain":
                    if score_method != "quality_valgain":
                        print(f"    [WARN] score_method='{score_method}' not valid for "
                              f"val_gain; using 'quality_valgain'")
                    scores = compute_quality_valgain_scores(
                        selector_model,
                        X_train_sel.astype(np.float32),
                        y_train_full,
                        X_val_sel.astype(np.float32),
                        y_val,
                        X_train_full.astype(np.float32),
                        ch_names,
                        valgain_lambda=valgain_lambda,
                    )
                else:
                    scores = compute_classifier_scores(
                        selector_model, X_train_sel.astype(np.float32), y_train_full, method=score_method
                    )
            elif selection_source == "learnability":
                if score_method != "quality_learnability":
                    print(f"    [WARN] score_method='{score_method}' not valid for "
                          f"learnability; using 'quality_learnability'")
                print(f"    [sel-learnability] warm-up improvement proxy "
                      f"(epochs={learnability_epochs}, lambda={learnability_lambda:.2f})")
                scores = compute_learnability_scores(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_times=X.shape[2],
                    n_filters=n_filters,
                    X_train_sel=X_train_sel.astype(np.float32),
                    y_train=y_train_full,
                    X_train_full=X_train_full.astype(np.float32),
                    X_train_prenorm=X_train_full,
                    y_train_prenorm=y_train_full,
                    ch_names=ch_names,
                    warmup_epochs=learnability_epochs,
                    learnability_lambda=learnability_lambda,
                )
            elif selection_source == "discriminator":
                print(f"    [sel-discriminator] tangent AE + GRL discriminator "
                      f"(epochs={selector_epochs}, score={score_method})")
                z_train = compute_tangent_features(X_train_full)
                ae_epochs = max(20, selector_epochs // 2)
                autoencoder, ae_mean, ae_std = train_tangent_autoencoder(
                    z_train,
                    epochs=ae_epochs,
                    batch_size=max(BATCH_SIZE, 256),
                    lr=1e-3,
                )
                quality_labels = compute_quality_labels(
                    autoencoder,
                    z_train,
                    z_mean=ae_mean,
                    z_std=ae_std,
                    batch_size=max(BATCH_SIZE, 256),
                )
                discriminator, z_mean, z_std = train_discriminator_selector(
                    z_train=z_train,
                    quality_labels=quality_labels,
                    subject_ids=train_subjects,
                    epochs=selector_epochs,
                    batch_size=max(BATCH_SIZE, 256),
                    lr=1e-3,
                    lambda_q=lambda_q,
                    lambda_d=lambda_d,
                )
                # Validate score_method for discriminator; warn and fall back to "final"
                if score_method not in ("quality", "domain", "final"):
                    print(f"    [WARN] score_method='{score_method}' not valid for "
                          f"discriminator; using 'final'")
                    disc_method = "final"
                else:
                    disc_method = score_method
                scores = compute_discriminator_scores(
                    discriminator,
                    z_train,
                    z_mean=z_mean,
                    z_std=z_std,
                    method=disc_method,
                    batch_size=max(BATCH_SIZE, 256),
                )
            elif selection_source == "dih":
                if score_method != "dih":
                    print(f"    [WARN] score_method='{score_method}' not valid for "
                          f"dih; using 'dih'")
                print(f"    [sel-dih] dynamic hardness tracking "
                      f"(epochs={selector_epochs}, score=dih)")
                scores = compute_dih_scores(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_times=X.shape[2],
                    n_filters=n_filters,
                    X_train=X_train_sel.astype(np.float32),
                    y_train=y_train_full,
                    X_val=X_val_sel.astype(np.float32),
                    y_val=y_val,
                    X_test=X_test_sel.astype(np.float32),
                    y_test=y_test,
                    selector_epochs=selector_epochs,
                )
            elif selection_source == "core_set":
                if score_method != "kcenter":
                    print(f"    [WARN] score_method='{score_method}' not valid for "
                          f"core_set; using 'kcenter'")
                print("    [sel-core_set] tangent-space k-center greedy "
                      "(score=kcenter)")
                scores = compute_coreset_scores(
                    X_train_full.astype(np.float32),
                    y_train_full,
                    batch_size=max(BATCH_SIZE, 256),
                )
            else:
                if score_method == "quality_diversity":
                    print("    [sel-heuristic] combined quality + tangent diversity "
                          f"(alpha={diversity_alpha:.2f})")
                    scores = compute_quality_diversity_scores(
                        X_train_full.astype(np.float32),
                        y_train_full,
                        ch_names,
                        alpha=diversity_alpha,
                        batch_size=max(BATCH_SIZE, 256),
                    )
                else:
                    scores = score_trials(X_train_full, ch_names, method=score_method)

            if save_selection_plots_flag:
                stem = (
                    f"{dataset_name}_S{int(test_subj):02d}_valS{int(val_subj):02d}_"
                    f"{selection_source}_{selection_mode}_{int(keep_ratio * 100)}_{score_method}"
                )
                title_prefix = (
                    f"{dataset_name} S{int(test_subj):02d} "
                    f"{selection_source}/{selection_mode} keep={int(keep_ratio * 100)}%"
                )
                save_selection_plots(
                    scores=scores,
                    y=y_train_full,
                    keep_ratio=keep_ratio,
                    out_dir=selection_plot_dir,
                    stem=stem,
                    balanced=True,
                    title_prefix=title_prefix,
                    max_points=selection_plot_max_points,
                )

            if selection_mode == "hard":
                X_train, y_train = select_trials(
                    X_train_full, y_train_full, scores, keep_ratio, balanced=True
                )
                print(f"    [sel-{selection_source}/hard] {n_before} -> {len(y_train)} trials "
                      f"(kept {len(y_train)/n_before*100:.0f}%, method={score_method})")
            else:
                X_train, y_train = X_train_full, y_train_full
                train_weights = score_to_weights(
                    scores, y_train, keep_ratio, balanced=True, min_weight=min_weight
                )
                print(f"    [sel-{selection_source}/weighted] {n_before} trials "
                      f"(w_min={train_weights.min():.3f}, "
                      f"w_mean={train_weights.mean():.3f}, "
                      f"w_max={train_weights.max():.3f}, method={score_method})")

            X_train_prenorm = X_train.copy()
            y_train_prenorm = y_train.copy()

        # -- Normalise (per-channel z-score on training statistics) ---------
        mu  = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_train = (X_train - mu) / std
        X_val   = (X_val   - mu) / std
        X_test  = (X_test  - mu) / std

        # -- DataLoaders ----------------------------------------------------
        is_dann        = (model_name == "cspnetdann")
        is_contrastive = (model_name == "cspnetcontrastive")

        if is_dann:
            train_subj_ids = subjects[train_mask]

            # ── Quality labels ────────────────────────────────────────────────
            if dann_quality_mode == "helpfulness":
                print("    [DANN] computing helpfulness quality labels "
                      f"(cspnet warm-up {dann_helpfulness_epochs} ep)...", flush=True)
                quality_labels_dann = compute_helpfulness_quality_labels(
                    X_train, y_train,
                    X_train_prenorm, y_train_prenorm,
                    n_channels, X.shape[2], n_filters,
                    epochs=dann_helpfulness_epochs,
                )
            else:
                print("    [DANN] computing AE quality labels...", flush=True)
                z_train_dann = compute_tangent_features(X_train)
                ae_dann, ae_mean_dann, ae_std_dann = train_tangent_autoencoder(
                    z_train_dann, epochs=50,
                    batch_size=max(BATCH_SIZE, 256), lr=1e-3,
                )
                quality_labels_dann = compute_quality_labels(
                    ae_dann, z_train_dann,
                    z_mean=ae_mean_dann, z_std=ae_std_dann,
                    batch_size=max(BATCH_SIZE, 256),
                )

            # ── Domain labels ─────────────────────────────────────────────────
            if dann_domain_mode == "binary":
                subj_idx_dann, n_train_subj = compute_binary_domain_labels(
                    X_train, train_subj_ids
                )
            else:
                unique_train_subj = np.unique(train_subj_ids)
                subj_map = {int(s): i for i, s in enumerate(unique_train_subj)}
                subj_idx_dann = np.array(
                    [subj_map[int(s)] for s in train_subj_ids], dtype=np.int64
                )
                n_train_subj = len(unique_train_subj)

            train_ds = TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
                torch.from_numpy(quality_labels_dann),
                torch.from_numpy(subj_idx_dann),
            )
        elif train_weights is None:
            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        else:
            train_ds = TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
                torch.from_numpy(train_weights.astype(np.float32)),
            )
        val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
        test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)

        snapshot_states = None  # only set by train_one_fold_model below
        if is_contrastive:
            model, criterion, raw_epoch_log = train_one_fold_contrastive(
                model_name=model_name,
                n_channels=n_channels,
                n_times=X.shape[2],
                n_filters=n_filters,
                X_train_prenorm=X_train_prenorm,
                y_train_prenorm=y_train_prenorm,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                y_val=y_val,
                y_test=y_test,
                epochs=EPOCHS,
                lambda_con=lambda_con,
                lambda_coral=lambda_coral,
                stage1_epochs=dann_stage1_epochs,
                temperature=temperature,
                target_domain_loader=val_loader,
                feature_mixup_alpha=feature_mixup_alpha,
                lambda_feature_mixup=lambda_feature_mixup,
                label_smoothing=label_smoothing,
            )
        elif is_dann:
            model, criterion, raw_epoch_log = train_one_fold_dann(
                model_name=model_name,
                n_channels=n_channels,
                n_times=X.shape[2],
                n_filters=n_filters,
                n_subjects=n_train_subj,
                X_train_prenorm=X_train_prenorm,
                y_train_prenorm=y_train_prenorm,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                y_val=y_val,
                y_test=y_test,
                epochs=EPOCHS,
                lambda_q=lambda_q,
                lambda_d=lambda_d,
                stage1_epochs=dann_stage1_epochs,
                domain_delay=dann_domain_delay,
                grl_exponent=dann_grl_exponent,
                dann_mode=dann_mode,
                freeze_encoder=dann_freeze_encoder,
            )
        else:
            model, criterion, raw_epoch_log, snapshot_states = train_one_fold_model(
                model_name=model_name,
                n_channels=n_channels,
                n_times=X.shape[2],
                n_filters=n_filters,
                X_train_prenorm=X_train_prenorm,
                y_train_prenorm=y_train_prenorm,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                y_val=y_val,
                y_test=y_test,
                augment=augment,
                epochs=EPOCHS,
                feature_mixup_alpha=feature_mixup_alpha,
                lambda_feature_mixup=lambda_feature_mixup,
                label_smoothing=label_smoothing,
                use_snapshot=use_snapshot,
                snapshot_T0=snapshot_T0,
                use_sam=use_sam,
                sam_rho=sam_rho,
                style_aug=fold_style_aug,
            )
        epoch_log = [
            dict(dataset=dataset_name, subject=int(test_subj), **row)
            for row in raw_epoch_log
        ]

        acc, bac, kappa, y_true, y_pred = evaluate(model, test_loader)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{run_id}_{dataset_name}_S{int(test_subj):02d}_{model_name}.pt",
            )
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "dataset": dataset_name,
                    "test_subject": int(test_subj),
                    "validation_subject": int(val_subj),
                    "model": model_name,
                    "channels": list(ch_names),
                    "sfreq": float(sfreq),
                    "n_times": int(X.shape[2]),
                    "seed": SEED,
                    "split_seed": split_seed,
                    "epochs": EPOCHS,
                    "use_ea": use_ea,
                    "ea_eps_mode": ea_eps_mode,
                    "normalization_mean": mu,
                    "normalization_std": std,
                    "test_acc": float(acc),
                    "test_bac": float(bac),
                    "test_kappa": float(kappa),
                },
                checkpoint_path,
            )
            print(f"    Checkpoint: {checkpoint_path}", flush=True)

        row_extra = {}
        if use_calibration:
            ece, brier, coverage, reject_acc = evaluate_uncertainty(
                model, test_loader, threshold=reject_threshold
            )
            conf_coverage, conf_size, conf_qhat = evaluate_conformal(
                model, val_loader, test_loader, alpha=conformal_alpha
            )
            row_extra.update(
                ece=ece, brier=brier, reject_coverage=coverage,
                reject_acc=reject_acc,
                conformal_coverage=conf_coverage,
                conformal_set_size=conf_size,
                conformal_qhat=conf_qhat,
            )
            print(f"    [Calibration] ECE={ece:.3f} Brier={brier:.3f} "
                  f"coverage@{reject_threshold:g}={coverage:.3f} "
                  f"acc={reject_acc:.3f} conformal={conf_coverage:.3f}/"
                  f"{conf_size:.2f}", flush=True)

        if use_pseudo_label:
            pseudo_model, pseudo_n = adapt_pseudo_label_classifier(
                model, X_test, threshold=pseudo_threshold,
                epochs=pseudo_epochs, lr=pseudo_lr,
            )
            pseudo_acc = pseudo_bac = pseudo_kappa = None
            if pseudo_model is not None:
                pseudo_acc, pseudo_bac, pseudo_kappa, _, _ = evaluate(
                    pseudo_model, test_loader
                )
            row_extra.update(
                pseudo_n=pseudo_n, pseudo_acc=pseudo_acc,
                pseudo_bac=pseudo_bac, pseudo_kappa=pseudo_kappa,
            )
            print(f"    [PseudoLabel] n={pseudo_n} threshold={pseudo_threshold:g} "
                  f"acc={pseudo_acc if pseudo_acc is not None else float('nan'):.3f}",
                  flush=True)

        # ── AdaBN: test-time BN stat adaptation (no labels, no grad) ─────────
        adabn_acc = adabn_bac = adabn_kappa = None
        if use_adabn:
            bn_snap = snapshot_bn_stats(model)
            n_bn = apply_adabn(model, X_test, DEVICE,
                               batch_size=BATCH_SIZE, n_passes=adabn_passes)
            if n_bn > 0:
                adabn_acc, adabn_bac, adabn_kappa, _, _ = evaluate(model, test_loader)
                shift_str = adabn_summary(bn_snap, model)
                delta = (adabn_acc - acc) * 100
                print(f"    [AdaBN] {n_bn} BN layers  {shift_str}  "
                      f"Δacc={delta:+.1f}%  ({acc*100:.1f}%→{adabn_acc*100:.1f}%)",
                      flush=True)

        # ── TENT: entropy minimization on BN affine params (with grad) ────────
        tent_acc = tent_bac = tent_kappa = None
        if use_tent:
            affine_snap = snapshot_bn_affine(model)
            n_tent = apply_tent(model, X_test, DEVICE,
                                batch_size=BATCH_SIZE,
                                n_steps=tent_steps, lr=tent_lr,
                                use_adabn_warmup=(not use_adabn),
                                adabn_passes=adabn_passes)
            if n_tent > 0:
                tent_acc, tent_bac, tent_kappa, _, _ = evaluate(model, test_loader)
                shift_str = tent_summary(model, affine_snap)
                base = adabn_acc if adabn_acc is not None else acc
                delta = (tent_acc - base) * 100
                print(f"    [TENT]  {n_tent} BN layers  {shift_str}  "
                      f"Δacc={delta:+.1f}%  ({base*100:.1f}%→{tent_acc*100:.1f}%)",
                      flush=True)

        # ── Snapshot Ensemble ─────────────────────────────────────────────────
        snap_acc = snap_bac = snap_kappa = None
        snap_adabn_acc = snap_adabn_bac = snap_adabn_kappa = None
        if use_snapshot and snapshot_states:
            snap_acc, snap_bac, snap_kappa, _, _ = evaluate_ensemble(
                snapshot_states, model_name, n_channels, X.shape[2],
                n_filters, test_loader,
            )
            delta = (snap_acc - acc) * 100
            print(f"    [Snapshot] {len(snapshot_states)} snaps  "
                  f"Δacc={delta:+.1f}%  ({acc*100:.1f}%→{snap_acc*100:.1f}%)",
                  flush=True)
            if use_adabn:
                snap_adabn_acc, snap_adabn_bac, snap_adabn_kappa, _, _ = \
                    evaluate_ensemble_adabn(
                        snapshot_states, model_name, n_channels, X.shape[2],
                        n_filters, test_loader, X_test, adabn_passes,
                    )
                base = adabn_acc or acc
                delta2 = (snap_adabn_acc - base) * 100
                print(f"    [Snapshot+AdaBN] Δacc={delta2:+.1f}%  "
                      f"({base*100:.1f}%→{snap_adabn_acc*100:.1f}%)", flush=True)
        if use_snapshot:
            row_extra.update(
                snap_acc=snap_acc, snap_bac=snap_bac, snap_kappa=snap_kappa,
                snap_adabn_acc=snap_adabn_acc,
                snap_adabn_bac=snap_adabn_bac,
                snap_adabn_kappa=snap_adabn_kappa,
            )

        elapsed = time.time() - t_subj_start

        best_row = max(epoch_log, key=lambda x: x["val_acc"])
        results.append(dict(
            dataset=dataset_name,
            subject=int(test_subj),
            n_train=int(len(y_train)),
            n_test=int(test_mask.sum()),
            acc=acc, bac=bac, kappa=kappa,
            best_epoch=best_row["epoch"],
            best_val_acc=best_row["val_acc"],
            best_val_loss=best_row["val_loss"],
            time_min=round(elapsed / 60, 2),
            adabn_acc=adabn_acc, adabn_bac=adabn_bac, adabn_kappa=adabn_kappa,
            tent_acc=tent_acc, tent_bac=tent_bac, tent_kappa=tent_kappa,
            **row_extra,
        ))
        remaining = elapsed * (len(subj_ids) - i - 1 - len(done_subjects))
        best_ep = best_row["epoch"]
        print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} "
              f"(val=S{val_subj:02d}) | "
              f"Acc={acc*100:.1f}%  BAcc={bac*100:.1f}%  k={kappa:.3f}  "
              f"best_ep={best_ep}  "
              f"[{elapsed/60:.1f}min | ETA {remaining/60:.0f}min]")

        # -- Incremental save (write immediately after each subject) --------
        with open(res_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=res_fields).writerow(results[-1])
        with open(loss_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=loss_fields).writerows(epoch_log)

    # -- Summary ---------------------------------------------------------------
    total_time = time.time() - t_total_start
    if not results:
        print(f"\n{'-'*60}")
        print(f"  {dataset_name.upper()} LOSO Summary: all subjects already done (resume), skipping.")
        print(f"{'-'*60}\n")
        return results

    accs  = [r["acc"]  for r in results]
    bacs  = [r["bac"]  for r in results]
    kappas= [r["kappa"] for r in results]

    best_epochs = [r["best_epoch"] for r in results]
    print(f"\n{'-'*60}")
    print(f"  {dataset_name.upper()} LOSO Summary ({len(results)} subjects)")
    print(f"  Accuracy  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f} %")
    print(f"  Bal. Acc  : {np.mean(bacs)*100:.2f} ± {np.std(bacs)*100:.2f} %")
    print(f"  Cohen k   : {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    print(f"  Best epoch: {np.mean(best_epochs):.0f} ± {np.std(best_epochs):.0f} "
          f"(min={min(best_epochs)}, max={max(best_epochs)})")
    print(f"  Total time: {total_time/60:.1f} min  "
          f"({total_time/len(results)/60:.1f} min/subject)")
    print(f"{'-'*60}\n")

    return results


# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

def save_results(all_results, all_loss_logs, tag=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- Per-subject summary CSV --------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"loso_results_{ts}{tag}.csv")
    fields = ["dataset", "subject", "n_train", "n_test",
              "acc", "bac", "kappa", "best_epoch", "best_val_acc",
              "best_val_loss", "time_min"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"Results saved  -> {csv_path}")

    # -- Per-epoch loss CSV -------------------------------------------------
    loss_path = os.path.join(RESULTS_DIR, f"loso_loss_{ts}{tag}.csv")
    loss_fields = ["dataset", "subject", "epoch",
                   "train_loss", "train_acc",
                   "val_loss",   "val_acc",
                   "test_loss",  "test_acc"]
    with open(loss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=loss_fields)
        w.writeheader()
        w.writerows(all_loss_logs)
    print(f"Loss log saved -> {loss_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["cho2017", "lee2019", "both", "physionet", "bciciv2a"],
        default="both",
        help="Which dataset to run LOSO on (default: both)",
    )
    parser.add_argument(
        "--ch_filter", type=str, default=None,
        help="Keep only channels whose name contains this string (e.g. 'C')",
    )
    parser.add_argument(
        "--model", choices=["spdnet", "riemgat", "min2net", "cspnet", "eegnet",
                            "cspnetsnn", "cspnetrsnn", "cspnetdann",
                            "cspnetcontrastive", "conformer"], default="spdnet",
        help="Model architecture (default: spdnet)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable signal-level augmentation (time jitter + amp scaling + noise)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=RESULTS_DIR,
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed (default: 2026)",
    )
    parser.add_argument(
        "--split_seed", type=int, default=None,
        help="Validation split seed; defaults to MI_SPLIT_SEED or --seed",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Training epochs per fold (default: 300)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Optional directory for one pre-adaptation checkpoint per fold",
    )
    parser.add_argument(
        "--max_test_subjects", type=int, default=None,
        help="Smoke-test only: evaluate the first N LOSO test subjects",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Run id for output filenames (default: timestamp)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing CSV for the same run_id (or latest if run_id is not set)",
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=1.0,
        help="Fraction of train pool trials to keep by quality score "
             "(1.0 = no selection, 0.6 = keep top 60%%). "
             "Class balance is preserved within each fold.",
    )
    parser.add_argument(
        "--score_method",
        choices=["band_power", "laterality", "cov_quality", "combined",
                 "confidence", "trueprob", "margin",
                 "quality", "domain", "final", "rho_cosine",
                 "dih", "kcenter", "quality_diversity", "quality_entropy",
                 "quality_learnability", "quality_valgain"],
        default="combined",
        help="Trial scoring method for --keep_ratio selection. "
             "Heuristic source: band_power/laterality/cov_quality/combined. "
             "Classifier source: confidence/trueprob/margin. "
             "Discriminator source: quality/domain/final. "
             "RHO-LOSS source: rho_cosine. "
             "DIH source: dih. "
             "Core-set source: kcenter. "
             "Quality-diversity source: quality_diversity. "
             "Uncertainty source: quality_entropy. "
             "Learnability source: quality_learnability. "
             "Validation-gain source: quality_valgain. "
             "(default: combined)",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["hard", "weighted"],
        default="hard",
        help="How to use suitability scores when keep_ratio < 1.0 "
             "(default: hard)",
    )
    parser.add_argument(
        "--min_weight", type=float, default=0.25,
        help="Minimum per-trial weight for weighted selection mode "
             "(default: 0.25)",
    )
    parser.add_argument(
        "--selection_source",
        choices=["heuristic", "classifier", "discriminator", "rho_loss", "dih",
                 "core_set", "uncertainty", "learnability", "val_gain"],
        default="heuristic",
        help="Where trial-selection scores come from when keep_ratio < 1.0 "
             "(default: heuristic)",
    )
    parser.add_argument(
        "--selector_epochs", type=int, default=100,
        help="Pretraining epochs for classifier/discriminator-based trial scoring "
             "(default: 100)",
    )
    parser.add_argument(
        "--lambda_con", type=float, default=0.5,
        help="SupCon loss weight for cspnetcontrastive (default: 0.5)",
    )
    parser.add_argument(
        "--lambda_coral", type=float, default=0.0,
        help="CORAL alignment weight for cspnetcontrastive (default: 0.0)",
    )
    parser.add_argument(
        "--feature_mixup_alpha", type=float, default=0.0,
        help="Beta(alpha, alpha) parameter for same-class feature-space MixUp "
             "(0 disables; suggested: 0.4)",
    )
    parser.add_argument(
        "--lambda_feature_mixup", type=float, default=0.0,
        help="Feature-space MixUp classification loss weight (default: 0.0)",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="Cross-entropy label smoothing factor (default: 0.0)",
    )
    parser.add_argument(
        "--con_temperature", type=float, default=0.07,
        help="SupCon temperature for cspnetcontrastive (default: 0.07)",
    )
    parser.add_argument(
        "--lambda_q", type=float, default=0.5,
        help="Quality-loss weight for DANN/discriminator (default: 0.5)",
    )
    parser.add_argument(
        "--lambda_d", type=float, default=0.1,
        help="Domain-loss / GRL weight for DANN/discriminator (default: 0.1 weakly adversarial)",
    )
    parser.add_argument(
        "--dann_stage1_epochs", type=int, default=150,
        help="DANN two-stage: epochs for MI-only stage 1 (default: 150, 0=skip)",
    )
    parser.add_argument(
        "--dann_domain_delay", type=int, default=10,
        help="DANN: stage-2 epochs to skip domain loss before GRL ramp (default: 10)",
    )
    parser.add_argument(
        "--dann_grl_exponent", type=float, default=5.0,
        help="DANN: GRL sigmoid exponent (default: 5.0; original was 10.0)",
    )
    parser.add_argument(
        "--dann_mode",
        choices=["full", "quality_only", "domain_only", "no_aux"],
        default="full",
        help="DANN ablation mode: full=quality+domain, quality_only, domain_only, "
             "no_aux=MI-only two-stage (default: full)",
    )
    parser.add_argument(
        "--dann_no_freeze", action="store_true",
        help="DANN: skip encoder freeze in stage 2 (ablate freeze contribution)",
    )
    parser.add_argument(
        "--dann_quality_mode",
        choices=["ae", "helpfulness"],
        default="helpfulness",
        help="DANN quality label source: ae=tangent AE recon, "
             "helpfulness=CSPNet warm-up trueprob (default: helpfulness)",
    )
    parser.add_argument(
        "--dann_domain_mode",
        choices=["subject", "binary"],
        default="binary",
        help="DANN domain label: subject=full subject-ID, "
             "binary=k-means 2-cluster (default: binary)",
    )
    parser.add_argument(
        "--dann_helpfulness_epochs", type=int, default=25,
        help="Epochs for CSPNet warm-up in helpfulness quality label (default: 25)",
    )
    parser.add_argument(
        "--diversity_alpha", type=float, default=0.3,
        help="Diversity weight for score_method=quality_diversity (default: 0.3)",
    )
    parser.add_argument(
        "--uncertainty_lambda", type=float, default=0.2,
        help="Entropy bonus for selection_source=uncertainty (default: 0.2)",
    )
    parser.add_argument(
        "--learnability_lambda", type=float, default=0.3,
        help="Learnability bonus for selection_source=learnability (default: 0.3)",
    )
    parser.add_argument(
        "--learnability_epochs", type=int, default=8,
        help="Warm-up epochs for learnability scoring (default: 8)",
    )
    parser.add_argument(
        "--valgain_lambda", type=float, default=0.3,
        help="Validation-gain proxy bonus for selection_source=val_gain (default: 0.3)",
    )
    parser.add_argument(
        "--save_selection_plots", action="store_true",
        help="Save ranked/distribution PNGs for fold-level selection scores "
             "(only used when keep_ratio < 1.0)",
    )
    parser.add_argument(
        "--selection_plot_max_points", type=int, default=4000,
        help="Maximum ranked points to draw in selection plots (default: 4000)",
    )
    parser.add_argument(
        "--adabn", action="store_true",
        help="Apply AdaBN (test-time BN stat adaptation) after training",
    )
    parser.add_argument(
        "--adabn_passes", type=int, default=3,
        help="Number of forward passes over test data for AdaBN (default: 3)",
    )
    parser.add_argument(
        "--ea", action="store_true",
        help="Apply Euclidean Alignment per-subject before LOSO training",
    )
    parser.add_argument(
        "--ea_eps_mode", choices=EA_EPS_MODES,
        default=os.environ.get("MI_EA_FLOOR", "relative"),
        help="EA eigenvalue handling (default: MI_EA_FLOOR or relative)",
    )
    parser.add_argument(
        "--tent", action="store_true",
        help="Apply TENT (entropy minimization on BN affine) after training",
    )
    parser.add_argument(
        "--tent_steps", type=int, default=1,
        help="TENT gradient steps per test pass (default: 1)",
    )
    parser.add_argument(
        "--tent_lr", type=float, default=1e-3,
        help="TENT Adam learning rate for BN affine params (default: 1e-3)",
    )
    parser.add_argument(
        "--calibration", action="store_true",
        help="Save subject-level ECE, Brier score and confidence-rejection metrics",
    )
    parser.add_argument(
        "--reject_threshold", type=float, default=0.9,
        help="Confidence threshold for calibration reject metrics (default: 0.9)",
    )
    parser.add_argument(
        "--conformal_alpha", type=float, default=0.1,
        help="Target error rate for split conformal prediction sets (default: 0.1)",
    )
    parser.add_argument(
        "--pseudo_label", action="store_true",
        help="Fine-tune only the final classifier on high-confidence unlabeled test trials",
    )
    parser.add_argument(
        "--pseudo_threshold", type=float, default=0.9,
        help="Confidence threshold for pseudo-label selection (default: 0.9)",
    )
    parser.add_argument(
        "--pseudo_epochs", type=int, default=10,
        help="Classifier-only pseudo-label adaptation epochs (default: 10)",
    )
    parser.add_argument(
        "--pseudo_lr", type=float, default=1e-4,
        help="Classifier-only pseudo-label adaptation learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--snapshot_ensemble", action="store_true",
        help="Snapshot Ensemble: save model at each cosine cycle end, ensemble at test time",
    )
    parser.add_argument(
        "--snapshot_T0", type=int, default=50,
        help="Cosine cycle length in epochs for snapshot ensemble (default: 50 → 6 snapshots per 300ep)",
    )
    parser.add_argument(
        "--fbcsp", action="store_true",
        help="Filterbank CSP: concatenate original + mu(8-12Hz) + beta(13-30Hz) band signals",
    )
    parser.add_argument(
        "--subject_weight", action="store_true",
        help="Subject Clustering: weight training samples by Riemannian distance to test subject",
    )
    parser.add_argument(
        "--subject_weight_tau", type=float, default=1.0,
        help="Temperature for subject similarity softmax (default: 1.0)",
    )
    parser.add_argument(
        "--sam", action="store_true",
        help="Use Sharpness-Aware Minimization (SAM) optimizer instead of plain Adam",
    )
    parser.add_argument(
        "--sam_rho", type=float, default=0.05,
        help="SAM neighbourhood size rho (default: 0.05)",
    )
    parser.add_argument(
        "--style_aug", action="store_true",
        help="Covariance Style Transfer augmentation (reverse of EA — re-colors whitened data)",
    )
    parser.add_argument(
        "--style_aug_p", type=float, default=0.5,
        help="Probability of applying style aug per batch (default: 0.5)",
    )
    parser.add_argument(
        "--style_aug_alpha", type=float, default=1.0,
        help="Covariance interpolation strength [0,1]; 1.0=full transfer (default: 1.0)",
    )
    args = parser.parse_args()

    if args.split_seed is None:
        args.split_seed = int(os.environ.get("MI_SPLIT_SEED", str(args.seed)))
    SEED = args.seed
    SPLIT_SEED = args.split_seed
    EPOCHS = args.epochs

    ch_tag = f"_ch{args.ch_filter}" if args.ch_filter else ""
    model_tag = f"_{args.model}"
    print(f"\nDevice    : {DEVICE}")
    print(f"Model     : {args.model}")
    print(f"Augment   : {'ON (time_jitter+amp_scale+noise)' if args.augment else 'OFF'}")
    print(f"Selection : "
          f"{'source=' + args.selection_source + ' mode=' + args.selection_mode + ' keep=' + str(int(args.keep_ratio*100)) + '% method=' + args.score_method if args.keep_ratio < 1.0 else 'OFF (keep_ratio=1.0)'}")
    print(f"Config    : n_filters={N_FILTERS}, lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}")
    if args.ch_filter:
        print(f"Ch filter : '{args.ch_filter}' channels only")
    print()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    dann_kwargs = dict(
        dann_stage1_epochs=args.dann_stage1_epochs,
        dann_domain_delay=args.dann_domain_delay,
        dann_grl_exponent=args.dann_grl_exponent,
        dann_mode=args.dann_mode,
        dann_freeze_encoder=not args.dann_no_freeze,
        dann_quality_mode=args.dann_quality_mode,
        dann_domain_mode=args.dann_domain_mode,
        dann_helpfulness_epochs=args.dann_helpfulness_epochs,
        temperature=args.con_temperature,
        ea_eps_mode=args.ea_eps_mode,
        checkpoint_dir=args.checkpoint_dir,
        max_test_subjects=args.max_test_subjects,
        split_seed=args.split_seed,
    )

    if args.dataset in ("cho2017", "both"):
        run_loso("cho2017", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 lambda_con=args.lambda_con, lambda_coral=args.lambda_coral,
                 feature_mixup_alpha=args.feature_mixup_alpha,
                 lambda_feature_mixup=args.lambda_feature_mixup,
                 label_smoothing=args.label_smoothing,
                 diversity_alpha=args.diversity_alpha,
                 uncertainty_lambda=args.uncertainty_lambda,
                 learnability_lambda=args.learnability_lambda,
                 learnability_epochs=args.learnability_epochs,
                 valgain_lambda=args.valgain_lambda,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points,
                 use_adabn=args.adabn, adabn_passes=args.adabn_passes,
                 use_ea=args.ea,
                 use_tent=args.tent, tent_steps=args.tent_steps, tent_lr=args.tent_lr,
                 use_calibration=args.calibration,
                 reject_threshold=args.reject_threshold,
                 conformal_alpha=args.conformal_alpha,
                 use_pseudo_label=args.pseudo_label,
                 pseudo_threshold=args.pseudo_threshold,
                 pseudo_epochs=args.pseudo_epochs,
                 pseudo_lr=args.pseudo_lr,
                 use_snapshot=args.snapshot_ensemble,
                 snapshot_T0=args.snapshot_T0,
                 use_fbcsp=args.fbcsp,
                 use_subject_weight=args.subject_weight,
                 subject_weight_tau=args.subject_weight_tau,
                 use_sam=args.sam,
                 sam_rho=args.sam_rho,
                 use_style_aug=args.style_aug,
                 style_aug_p=args.style_aug_p,
                 style_aug_alpha=args.style_aug_alpha,
                 **dann_kwargs)

    if args.dataset == "physionet":
        run_loso("physionet", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 lambda_con=args.lambda_con, lambda_coral=args.lambda_coral,
                 feature_mixup_alpha=args.feature_mixup_alpha,
                 lambda_feature_mixup=args.lambda_feature_mixup,
                 label_smoothing=args.label_smoothing,
                 diversity_alpha=args.diversity_alpha,
                 uncertainty_lambda=args.uncertainty_lambda,
                 learnability_lambda=args.learnability_lambda,
                 learnability_epochs=args.learnability_epochs,
                 valgain_lambda=args.valgain_lambda,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points,
                 use_adabn=args.adabn, adabn_passes=args.adabn_passes,
                 use_ea=args.ea,
                 use_tent=args.tent, tent_steps=args.tent_steps, tent_lr=args.tent_lr,
                 use_calibration=args.calibration,
                 reject_threshold=args.reject_threshold,
                 conformal_alpha=args.conformal_alpha,
                 use_pseudo_label=args.pseudo_label,
                 pseudo_threshold=args.pseudo_threshold,
                 pseudo_epochs=args.pseudo_epochs,
                 pseudo_lr=args.pseudo_lr,
                 use_snapshot=args.snapshot_ensemble,
                 snapshot_T0=args.snapshot_T0,
                 use_fbcsp=args.fbcsp,
                 use_subject_weight=args.subject_weight,
                 subject_weight_tau=args.subject_weight_tau,
                 use_sam=args.sam,
                 sam_rho=args.sam_rho,
                 use_style_aug=args.style_aug,
                 style_aug_p=args.style_aug_p,
                 style_aug_alpha=args.style_aug_alpha,
                 **dann_kwargs)

    if args.dataset == "bciciv2a":
        run_loso("bciciv2a", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 lambda_con=args.lambda_con, lambda_coral=args.lambda_coral,
                 feature_mixup_alpha=args.feature_mixup_alpha,
                 lambda_feature_mixup=args.lambda_feature_mixup,
                 label_smoothing=args.label_smoothing,
                 diversity_alpha=args.diversity_alpha,
                 uncertainty_lambda=args.uncertainty_lambda,
                 learnability_lambda=args.learnability_lambda,
                 learnability_epochs=args.learnability_epochs,
                 valgain_lambda=args.valgain_lambda,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points,
                 use_adabn=args.adabn, adabn_passes=args.adabn_passes,
                 use_ea=args.ea,
                 use_tent=args.tent, tent_steps=args.tent_steps, tent_lr=args.tent_lr,
                 use_calibration=args.calibration,
                 reject_threshold=args.reject_threshold,
                 conformal_alpha=args.conformal_alpha,
                 use_pseudo_label=args.pseudo_label,
                 pseudo_threshold=args.pseudo_threshold,
                 pseudo_epochs=args.pseudo_epochs,
                 pseudo_lr=args.pseudo_lr,
                 use_snapshot=args.snapshot_ensemble,
                 snapshot_T0=args.snapshot_T0,
                 use_fbcsp=args.fbcsp,
                 use_subject_weight=args.subject_weight,
                 subject_weight_tau=args.subject_weight_tau,
                 use_sam=args.sam,
                 sam_rho=args.sam_rho,
                 use_style_aug=args.style_aug,
                 style_aug_p=args.style_aug_p,
                 style_aug_alpha=args.style_aug_alpha,
                 **dann_kwargs)

    if args.dataset in ("lee2019", "both"):
        run_loso("lee2019", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 lambda_con=args.lambda_con, lambda_coral=args.lambda_coral,
                 feature_mixup_alpha=args.feature_mixup_alpha,
                 lambda_feature_mixup=args.lambda_feature_mixup,
                 label_smoothing=args.label_smoothing,
                 diversity_alpha=args.diversity_alpha,
                 uncertainty_lambda=args.uncertainty_lambda,
                 learnability_lambda=args.learnability_lambda,
                 learnability_epochs=args.learnability_epochs,
                 valgain_lambda=args.valgain_lambda,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points,
                 use_adabn=args.adabn, adabn_passes=args.adabn_passes,
                 use_ea=args.ea,
                 use_tent=args.tent, tent_steps=args.tent_steps, tent_lr=args.tent_lr,
                 use_calibration=args.calibration,
                 reject_threshold=args.reject_threshold,
                 conformal_alpha=args.conformal_alpha,
                 use_pseudo_label=args.pseudo_label,
                 pseudo_threshold=args.pseudo_threshold,
                 pseudo_epochs=args.pseudo_epochs,
                 pseudo_lr=args.pseudo_lr,
                 use_snapshot=args.snapshot_ensemble,
                 snapshot_T0=args.snapshot_T0,
                 use_fbcsp=args.fbcsp,
                 use_subject_weight=args.subject_weight,
                 subject_weight_tau=args.subject_weight_tau,
                 use_sam=args.sam,
                 sam_rho=args.sam_rho,
                 use_style_aug=args.style_aug,
                 style_aug_p=args.style_aug_p,
                 style_aug_alpha=args.style_aug_alpha,
                 **dann_kwargs)
