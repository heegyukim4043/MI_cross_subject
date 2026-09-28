"""
Cross-dataset evaluation using common channels only.

Train on one dataset (all subjects), test on the other (per-subject).

Common channels: 48 channels shared between Cho2017 (64ch) and Lee2019 (62ch).
n_times: truncated to min(257, 201) = 201 for both.

Usage
-----
    # Cho2017 → Lee2019
    CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --train cho2017 --test lee2019

    # Lee2019 → Cho2017
    CUDA_VISIBLE_DEVICES=2 python cross_dataset.py --train lee2019 --test cho2017

    # Both directions
    CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --both

    # With contrastive model
    CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --both --model cspnetcontrastive
"""

import os, sys, argparse, copy, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from cspnet            import CSPNet, fit_csp_layer
from cspnet_contrastive import CSPNetContrastive, SupConLoss, fit_csp_layer_contrastive
from cspnet_dann      import CSPNetDANN, fit_csp_layer_dann, grl_lambda
from eegnet            import EEGNet
from conformer         import EEGConformer
from adabn             import apply_adabn, snapshot_bn_stats, adabn_summary
from eeg_ea            import EA_EPS_MODES, apply_ea_loso, euclidean_align
from tent              import apply_tent, snapshot_bn_affine, tent_summary
from dsbn              import DomainSpecificBatchNorm2d, convert_batchnorm_to_dsbn, set_dsbn_domain, apply_dsbn_target_stats

# ─────────────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = os.path.abspath(
    os.environ.get(
        "MI_PREPROCESSED_DIR",
        os.path.join(os.path.dirname(__file__), "..", "preprocessed"),
    )
)
RESULTS_DIR = os.path.abspath(
    os.environ.get(
        "MI_RESULTS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "results"),
    )
)
os.makedirs(RESULTS_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
EPOCHS       = 300
N_TIMES      = int(os.environ.get("MI_N_TIMES", "201"))   # default: min(257, 201)
SEED         = int(os.environ.get("MI_SEED", "2026"))            # weight init / training randomness
SPLIT_SEED   = int(os.environ.get("MI_SPLIT_SEED", str(SEED)))  # validation-subject selection

STANDARD_MI_CHANNELS = [
    "Fz",
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "Oz",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_common_channels(channel_set="common"):
    cho = np.load(os.path.join(SAVE_DIR, "cho2017.npz"), allow_pickle=True)
    lee = np.load(os.path.join(SAVE_DIR, "lee2019.npz"), allow_pickle=True)
    cho_ch = list(cho["ch_names"])
    lee_ch = list(lee["ch_names"])
    if channel_set == "standard_mi":
        common = [c for c in STANDARD_MI_CHANNELS if c in cho_ch and c in lee_ch]
        missing = [c for c in STANDARD_MI_CHANNELS if c not in cho_ch or c not in lee_ch]
        if missing:
            print(f"  Standard MI channels missing and skipped: {missing}")
    elif channel_set == "common":
        common = [c for c in cho_ch if c in lee_ch]
    else:
        raise ValueError(f"Unknown channel_set: {channel_set}")
    cho_idx = [cho_ch.index(c) for c in common]
    lee_idx = [lee_ch.index(c) for c in common]
    return common, cho_idx, lee_idx


def load_dataset(name: str, ch_idx: list, n_times: int = N_TIMES):
    d = np.load(os.path.join(SAVE_DIR, f"{name}.npz"), allow_pickle=True)
    X = d["X"].astype(np.float32)[:, ch_idx, :n_times]  # (N, C, T)
    y = d["y"].astype(np.int64)
    subjects = d["subjects"].astype(np.int64)
    print(f"  {name}: X={X.shape}, {len(np.unique(subjects))} subjects")
    return X, y, subjects


def normalize(X_train, X_test):
    mu  = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std


def psd_power_normalize_dataset(X, eps=1e-8):
    """Normalize each channel by its dataset-level RMS power.

    This is label-free and computed independently per dataset. It removes
    hardware/protocol scale differences before EA/global z-score without
    using target labels or source statistics.
    """
    rms = np.sqrt(np.mean(X ** 2, axis=(0, 2), keepdims=True) + eps)
    return (X / rms).astype(np.float32)


def apply_dataset_ea(X, eps_mode=None):
    """Apply one EA transform using the full dataset covariance."""
    return euclidean_align(X, eps_mode=eps_mode)

def apply_lee_session_ea(X, subjects, dataset_name, eps_mode=None):
    """Apply Lee2019 session-level EA using the saved two-session trial order."""
    if dataset_name != "lee2019":
        return X
    out = X.copy()
    for subj in np.unique(subjects):
        idx = np.flatnonzero(subjects == subj)
        for part in np.array_split(idx, 2):
            if len(part) >= 4:
                out[part] = euclidean_align(X[part], eps_mode=eps_mode)
    return out


def make_loader(X, y, shuffle=False, sample_weights=None):
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    if sample_weights is not None:
        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def _mean_cov(X, eps=1e-6):
    cov = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for trial in X.astype(np.float64):
        trial = trial - trial.mean(axis=1, keepdims=True)
        cc = trial @ trial.T
        tr = np.trace(cc)
        if tr > 1e-12:
            cov += cc / tr
    cov /= max(len(X), 1)
    cov += eps * np.eye(cov.shape[0])
    return cov


def _log_spd(C, eps=1e-8):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def source_subject_trial_weights(X_train, subjects, X_target, tau=0.5):
    target_log = _log_spd(_mean_cov(X_target))
    subj_ids = np.unique(subjects)
    dists = []
    for subj in subj_ids:
        subj_log = _log_spd(_mean_cov(X_train[subjects == subj]))
        dists.append(np.linalg.norm(target_log - subj_log, ord="fro"))
    dists = np.asarray(dists, dtype=np.float64)
    logits = -dists / max(tau, 1e-8)
    logits -= logits.max()
    subj_weights = np.exp(logits)
    subj_weights /= subj_weights.mean()
    mapping = {subj: weight for subj, weight in zip(subj_ids, subj_weights)}
    trial_weights = np.asarray([mapping[s] for s in subjects], dtype=np.float64)
    print("  Source subject weighting:")
    print(f"    tau={tau}  dist min/mean/max={dists.min():.3f}/{dists.mean():.3f}/{dists.max():.3f}")
    print(f"    weight min/mean/max={subj_weights.min():.3f}/{subj_weights.mean():.3f}/{subj_weights.max():.3f}")
    return trial_weights



class SupConBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, kind: str, feature_dim: int,
                 proj_dim: int = 64, proj_hidden: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        self.backbone = backbone
        self.kind = kind
        self.proj_head = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )
        self.supcon = SupConLoss(temperature=temperature)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "eegnet":
            return self.backbone._forward_features(x.unsqueeze(1))
        if self.kind == "conformer":
            return self.backbone.forward_features(x)
        raise ValueError(f"Unknown SupCon backbone kind: {self.kind}")

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "eegnet":
            return self.backbone.classifier(z)
        if self.kind == "conformer":
            return self.backbone.head(z)
        raise ValueError(f"Unknown SupCon backbone kind: {self.kind}")

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        logits = self.classify(z)
        if self.training:
            h = F.normalize(self.proj_head(z), dim=1)
            return logits, h
        return logits


def _infer_feature_dim(backbone: nn.Module, kind: str, n_channels: int, n_times: int) -> int:
    with torch.no_grad():
        probe = torch.zeros(1, n_channels, n_times)
        if kind == "eegnet":
            z = backbone._forward_features(probe.unsqueeze(1))
        elif kind == "conformer":
            z = backbone.forward_features(probe)
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
    return int(z.shape[1])

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name, n_channels, n_times, temperature=0.07, n_domains=2):
    if model_name == "cspnet":
        return CSPNet(
            n_channels=n_channels, n_times=n_times,
            n_csp=8, F1=8, F2=16, dropout=0.25, trainable_csp=True,
        ).to(DEVICE)
    if model_name == "cspnetcontrastive":
        return CSPNetContrastive(
            n_channels=n_channels, n_times=n_times,
            n_csp=8, F1=8, F2=16, dropout=0.25, trainable_csp=True,
            proj_dim=64, proj_hidden=128, temperature=temperature,
        ).to(DEVICE)
    if model_name == "cspnetdann":
        return CSPNetDANN(
            n_channels=n_channels, n_times=n_times, n_subjects=n_domains,
            n_csp=8, F1=8, F2=16, dropout=0.25, trainable_csp=True,
            lambda_grl=0.0, qual_hidden=64, domain_hidden=0,
            domain_dropout=0.0, use_grl=True,
        ).to(DEVICE)
    if model_name in ("eegnet", "eegnetcontrastive"):
        backbone = EEGNet(
            n_channels=n_channels, n_times=n_times,
            F1=8, D=2, F2=16, dropout=0.5,
        )
        if model_name == "eegnetcontrastive":
            feature_dim = _infer_feature_dim(backbone, "eegnet", n_channels, n_times)
            return SupConBackbone(backbone, "eegnet", feature_dim, temperature=temperature).to(DEVICE)
        return backbone.to(DEVICE)
    if model_name in ("conformer", "conformercontrastive"):
        backbone = EEGConformer(
            n_channels=n_channels, n_times=n_times,
            F1=40, D=2, temp_kern=25, pool=8,
            dropout=0.5, nhead=8, n_layers=2,
            ff_dim=256, attn_dropout=0.3,
        )
        if model_name == "conformercontrastive":
            feature_dim = _infer_feature_dim(backbone, "conformer", n_channels, n_times)
            return SupConBackbone(backbone, "conformer", feature_dim, temperature=temperature).to(DEVICE)
        return backbone.to(DEVICE)
    raise ValueError(f"Unknown model: {model_name}")


def init_csp(model, model_name, X_train, y_train):
    if model_name == "cspnet":
        fit_csp_layer(model, X_train, y_train)
    elif model_name == "cspnetcontrastive":
        fit_csp_layer_contrastive(model, X_train, y_train)
    elif model_name == "cspnetdann":
        fit_csp_layer_dann(model, X_train, y_train)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, model_name, lambda_con=0.5):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        if isinstance(out, tuple):
            logits, h = out
            loss = F.cross_entropy(logits, yb) + lambda_con * model.supcon(h, yb)
        else:
            logits = out
            loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n



@torch.no_grad()
def evaluate_snapshot_ensemble(snapshot_states, model_name, n_channels, n_times, loader,
                               X_adabn=None, adabn_passes=3):
    """Average softmax over snapshot checkpoints, optionally with per-subject AdaBN."""
    if not snapshot_states:
        return None, None, None, None, None
    all_probs, all_true = None, None
    for state in snapshot_states:
        m = build_model(model_name, n_channels, n_times)
        m.load_state_dict(state)
        if X_adabn is not None:
            apply_adabn(m, X_adabn, DEVICE, batch_size=BATCH_SIZE, n_passes=adabn_passes)
        m.eval()
        probs_list, true_list = [], []
        for xb, yb in loader:
            out = m(xb.to(DEVICE))
            logits = out[0] if isinstance(out, tuple) else out
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            true_list.extend(yb.numpy())
        probs = np.concatenate(probs_list, axis=0)
        all_probs = probs if all_probs is None else all_probs + probs
        if all_true is None:
            all_true = np.asarray(true_list)
    all_probs /= len(snapshot_states)
    y_pred = all_probs.argmax(1)
    return (accuracy_score(all_true, y_pred),
            balanced_accuracy_score(all_true, y_pred),
            cohen_kappa_score(all_true, y_pred),
            all_true, y_pred)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            logits = out[0] if isinstance(out, tuple) else out
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    y_true, y_pred = np.array(trues), np.array(preds)
    return (accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred),
            y_true, y_pred)


def pseudo_label_finetune(model, X_target, threshold=0.85, epochs=20, lr=5e-4):
    if any(isinstance(m, DomainSpecificBatchNorm2d) for m in model.modules()):
        set_dsbn_domain(model, 1)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_target.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb_dev = xb.to(DEVICE)
            out = model(xb_dev)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            keep = conf >= threshold
            if keep.any():
                xs.append(xb[keep.cpu()])
                ys.append(pred[keep].cpu())
    if not xs:
        print(f"  Pseudo-label FT skipped: no samples above threshold {threshold}")
        return 0
    X_pl = torch.cat(xs).float()
    y_pl = torch.cat(ys).long()
    print(f"  Pseudo-label FT: selected {len(y_pl)}/{len(X_target)} samples at threshold {threshold}")

    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, module in model.named_modules():
        if name.endswith("classifier") or isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, DomainSpecificBatchNorm2d)):
            for p in module.parameters(recurse=True):
                p.requires_grad = True
                trainable.append(p)
    if not trainable:
        print("  Pseudo-label FT skipped: no trainable BN/classifier parameters")
        return len(y_pl)

    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-5)
    ft_loader = DataLoader(TensorDataset(X_pl, y_pl), batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in ft_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        if ep in (1, epochs):
            print(f"    pseudo ep={ep:02d}/{epochs} loss={total/max(n,1):.4f}")
    model.eval()
    return len(y_pl)


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage training (Stage1: MI only, Stage2: MI + contrastive)
# ─────────────────────────────────────────────────────────────────────────────


def make_unlabeled_loader(X, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False,
    )


def train_dann_model(model, train_loader, val_loader, target_loader,
                     epochs=EPOCHS, stage1_ratio=0.5,
                     lambda_d=0.1, domain_delay=10,
                     grl_exponent=5.0):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stage1_epochs = int(epochs * stage1_ratio)
    stage2_total = max(epochs - stage1_epochs, 1)
    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        if ep <= stage1_epochs:
            model.set_grl_lambda(0.0)
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                z = model._encode(xb.unsqueeze(1))
                logits = model.classifier(z)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
        else:
            s2_epoch = ep - stage1_epochs
            if s2_epoch <= domain_delay:
                lam = 0.0
            else:
                adj = (s2_epoch - domain_delay) / max(stage2_total - domain_delay, 1)
                lam = grl_lambda(adj, exponent=grl_exponent)
            model.set_grl_lambda(lam)
            target_iter = iter(target_loader)
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                try:
                    (x_tgt,) = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    (x_tgt,) = next(target_iter)
                x_tgt = x_tgt.to(DEVICE)

                optimizer.zero_grad()
                z_src = model._encode(xb.unsqueeze(1))
                z_tgt = model._encode(x_tgt.unsqueeze(1))
                logits = model.classifier(z_src)
                cls_loss = F.cross_entropy(logits, yb)
                dom_in = torch.cat([z_src, z_tgt], dim=0)
                dom_logits = model.domain_from_features(dom_in)
                dom_labels = torch.cat([
                    torch.zeros(len(z_src), dtype=torch.long, device=DEVICE),
                    torch.ones(len(z_tgt), dtype=torch.long, device=DEVICE),
                ])
                dom_loss = F.cross_entropy(dom_logits, dom_labels)
                loss = cls_loss + lambda_d * dom_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                n += len(yb)

        scheduler.step()
        val_acc, _, _, _, _ = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            stage = "S1" if ep <= stage1_epochs else "S2"
            print(f"    ep={ep:3d} [{stage}]  train_acc={correct/max(n,1)*100:.1f}%  val_acc={val_acc*100:.1f}%  best={best_val*100:.1f}%", flush=True)

    model.load_state_dict(best_state)
    return best_val, None


def train_model(model, model_name, train_loader, val_loader,
                epochs=EPOCHS, lambda_con=0.5, stage1_ratio=0.5,
                use_snapshot=False, snapshot_T0=50):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if use_snapshot:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=snapshot_T0, T_mult=1, eta_min=LR * 1e-2
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    stage1_epochs = int(epochs * stage1_ratio) if model_name in ("cspnetcontrastive", "eegnetcontrastive", "conformercontrastive") else epochs

    best_val = -1.0
    best_state = None
    snapshot_states = [] if use_snapshot else None

    for ep in range(1, epochs + 1):
        # Stage 1: MI head only (for contrastive model)
        if model_name in ("cspnetcontrastive", "eegnetcontrastive", "conformercontrastive") and ep <= stage1_epochs:
            model.train()
            total_loss, correct, n = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                if model_name == "cspnetcontrastive":
                    z = model._encode(xb.unsqueeze(1))
                    logits = model.classifier(z)
                else:
                    z = model.encode(xb)
                    logits = model.classify(z)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
        else:
            train_epoch(model, train_loader, optimizer, model_name, lambda_con)

        scheduler.step()

        val_acc, _, _, _, _ = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if use_snapshot and ep % snapshot_T0 == 0:
            snapshot_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            print(f"    [Snapshot] saved cycle {len(snapshot_states)} at epoch {ep}", flush=True)

        if ep % 50 == 0:
            stage = "S1" if ep <= stage1_epochs else "S2"
            print(f"    ep={ep:3d} [{stage}]  val_acc={val_acc*100:.1f}%  best={best_val*100:.1f}%",
                  flush=True)

    model.load_state_dict(best_state)
    return best_val, snapshot_states


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset run
# ─────────────────────────────────────────────────────────────────────────────

def run_cross(train_name, test_name, model_name="cspnet",
              lambda_con=0.5, temperature=0.07, use_adabn=False,
              use_ea=False, use_tent=False, tent_steps=1, tent_lr=1e-3,
              use_psd_norm=False, use_dataset_ea=False, ea_order="dataset_subject",
              use_dsbn=False, channel_set="common",
              use_pseudo_label=False, pseudo_threshold=0.85, pseudo_epochs=20,
              use_source_weighting=False, source_weight_tau=0.5,
              use_snapshot=False, snapshot_T0=50, use_session_ea=False,
              out_dir=RESULTS_DIR, run_id=None, ea_eps_mode=None,
              epochs=EPOCHS, seed=SEED, split_seed=SPLIT_SEED, checkpoint_dir=None,
              max_train_subjects=None, max_test_subjects=None):

    global SEED
    SEED = seed  # train_model/train_dann_model seed from the module-level SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"cross_{train_name}_to_{test_name}_{model_name}"
    res_csv = os.path.join(out_dir, f"loso_results_{run_id}_{tag}.csv")

    print(f"\n{'='*60}")
    print(f"  Cross-dataset: TRAIN={train_name.upper()} → TEST={test_name.upper()}")
    print(f"  Model: {model_name}  EA: {use_ea}  DatasetEA: {use_dataset_ea}  PSDNorm: {use_psd_norm}")
    print(f"  EA order: {ea_order}  eps_mode: {ea_eps_mode}  DSBN: {use_dsbn}")
    print(f"  AdaBN: {use_adabn}  TENT: {use_tent}  PseudoLabel: {use_pseudo_label}")
    print(f"  SourceWeighting: {use_source_weighting} tau={source_weight_tau}")
    print(f"  SnapshotEnsemble: {use_snapshot} T0={snapshot_T0}")
    print(f"  SessionEA: {use_session_ea}")
    print(f"  n_times={N_TIMES}, channel_set={channel_set}")
    print(f"{'='*60}")

    # ── Find common channels ──────────────────────────────────────────────────
    common_ch, cho_idx, lee_idx = find_common_channels(channel_set=channel_set)
    n_ch = len(common_ch)
    train_idx = cho_idx if train_name == "cho2017" else lee_idx
    test_idx  = cho_idx if test_name  == "cho2017" else lee_idx
    print(f"  Common channels: {n_ch}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train_all, y_train_all, subj_train = load_dataset(train_name, train_idx)
    X_test_all,  y_test_all,  subj_test  = load_dataset(test_name,  test_idx)

    def limit_subjects(X, y, subjects, limit, label):
        if limit is None:
            return X, y, subjects
        keep_ids = np.unique(subjects)[:limit]
        keep = np.isin(subjects, keep_ids)
        print(f"  Smoke limit {label}: {len(keep_ids)} subjects {keep_ids.tolist()}")
        return X[keep], y[keep], subjects[keep]

    X_train_all, y_train_all, subj_train = limit_subjects(
        X_train_all, y_train_all, subj_train, max_train_subjects, "train"
    )
    X_test_all, y_test_all, subj_test = limit_subjects(
        X_test_all, y_test_all, subj_test, max_test_subjects, "test"
    )

    if use_psd_norm:
        print("  Applying dataset-independent channel power normalization")
        X_train_all = psd_power_normalize_dataset(X_train_all)
        X_test_all = psd_power_normalize_dataset(X_test_all)

    if ea_order not in ("dataset_subject", "subject_dataset"):
        raise ValueError(f"Unknown EA order: {ea_order}")

    def apply_subject_ea_pair(X_src, X_tgt):
        print("  Applying subject-level Euclidean Alignment independently to source and target subjects")
        return (
            apply_ea_loso(X_src, subj_train, eps_mode=ea_eps_mode),
            apply_ea_loso(X_tgt, subj_test, eps_mode=ea_eps_mode),
        )

    def apply_dataset_ea_pair(X_src, X_tgt):
        print("  Applying dataset-level Euclidean Alignment independently to source and target")
        return (
            apply_dataset_ea(X_src, eps_mode=ea_eps_mode),
            apply_dataset_ea(X_tgt, eps_mode=ea_eps_mode),
        )

    if ea_order == "subject_dataset":
        if use_ea:
            X_train_all, X_test_all = apply_subject_ea_pair(X_train_all, X_test_all)
        if use_dataset_ea:
            X_train_all, X_test_all = apply_dataset_ea_pair(X_train_all, X_test_all)
    else:
        if use_dataset_ea:
            X_train_all, X_test_all = apply_dataset_ea_pair(X_train_all, X_test_all)
        if use_ea:
            X_train_all, X_test_all = apply_subject_ea_pair(X_train_all, X_test_all)

    if use_session_ea:
        print("  Applying Lee2019 session-level EA after dataset/subject alignment")
        X_train_all = apply_lee_session_ea(
            X_train_all, subj_train, train_name, eps_mode=ea_eps_mode
        )
        X_test_all = apply_lee_session_ea(
            X_test_all, subj_test, test_name, eps_mode=ea_eps_mode
        )

    # ── Global normalize (fit on train, apply to test) ────────────────────────
    X_train_norm, X_test_norm = normalize(X_train_all, X_test_all)

    # ── Hold out 10% of train subjects for validation ─────────────────────────
    train_subj_ids = np.unique(subj_train)
    np.random.seed(split_seed)
    n_val = max(1, int(len(train_subj_ids) * 0.1))
    val_subj_ids = np.random.choice(train_subj_ids, n_val, replace=False)
    train_only_mask = ~np.isin(subj_train, val_subj_ids)
    val_mask        =  np.isin(subj_train, val_subj_ids)

    X_tr = X_train_norm[train_only_mask]
    y_tr = y_train_all[train_only_mask]
    X_val = X_train_norm[val_mask]
    y_val = y_train_all[val_mask]

    print(f"  Train: {X_tr.shape[0]} trials ({len(train_subj_ids)-n_val} subjects)")
    print(f"  Val  : {X_val.shape[0]} trials ({n_val} subjects) — {val_subj_ids}")
    print(f"  Test : {X_test_all.shape[0]} trials ({len(np.unique(subj_test))} subjects)")

    train_weights = None
    if use_source_weighting:
        all_weights = source_subject_trial_weights(X_train_all, subj_train, X_test_all, tau=source_weight_tau)
        train_weights = all_weights[train_only_mask]
    train_loader = make_loader(X_tr, y_tr, shuffle=True, sample_weights=train_weights)
    val_loader   = make_loader(X_val, y_val)
    target_loader = make_unlabeled_loader(X_test_norm, shuffle=True) if model_name == "cspnetdann" else None

    # ── Build & init model ────────────────────────────────────────────────────
    model = build_model(model_name, n_ch, N_TIMES, temperature=temperature)
    if use_dsbn:
        convert_batchnorm_to_dsbn(model, num_domains=2)
        model.to(DEVICE)
        n_dsbn = set_dsbn_domain(model, 0)
        print(f"  DSBN enabled: {n_dsbn} BN layers converted; source domain=0, target domain=1")
    init_csp(model, model_name, X_tr, y_tr)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    if model_name == "cspnetdann":
        best_val, snapshot_states = train_dann_model(
            model, train_loader, val_loader, target_loader, epochs=epochs,
        )
    else:
        best_val, snapshot_states = train_model(
            model, model_name, train_loader, val_loader,
            epochs=epochs, lambda_con=lambda_con,
            use_snapshot=use_snapshot, snapshot_T0=snapshot_T0,
        )
    elapsed = time.time() - t0
    print(f"  Training done: {elapsed/60:.1f} min  best_val={best_val*100:.1f}%")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_{tag}.pt")
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model": model_name,
                "train_dataset": train_name,
                "test_dataset": test_name,
                "channels": common_ch,
                "n_times": N_TIMES,
                "seed": seed,
                "split_seed": split_seed,
                "epochs": epochs,
                "use_ea": use_ea,
                "use_dataset_ea": use_dataset_ea,
                "ea_order": ea_order,
                "ea_eps_mode": ea_eps_mode,
                "normalization_mean": X_train_all.mean(axis=(0, 2), keepdims=True),
                "normalization_std": X_train_all.std(axis=(0, 2), keepdims=True) + 1e-8,
            },
            checkpoint_path,
        )
        print(f"  Checkpoint: {checkpoint_path}")

    if use_dsbn:
        n_dsbn = apply_dsbn_target_stats(model, X_test_norm, DEVICE, domain=1,
                                         batch_size=BATCH_SIZE, n_passes=3)
        print(f"  DSBN target stats calibrated on unlabeled {test_name}: {n_dsbn} layers")

    if use_pseudo_label:
        n_pl = pseudo_label_finetune(model, X_test_norm, threshold=pseudo_threshold,
                                     epochs=pseudo_epochs)
        print(f"  Pseudo-label FT done: {n_pl} samples used")

    # ── Per-subject test evaluation ───────────────────────────────────────────
    test_subj_ids = np.unique(subj_test)
    results = []
    res_fields = ["train_ds", "test_ds", "model", "subject",
                  "n_test", "acc", "bac", "kappa",
                  "adabn_acc", "adabn_bac", "adabn_kappa",
                  "tent_acc", "tent_bac", "tent_kappa",
                  "snap_acc", "snap_bac", "snap_kappa",
                  "snap_adabn_acc", "snap_adabn_bac", "snap_adabn_kappa"]

    with open(res_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=res_fields).writeheader()

    print(f"\n  Per-subject results on {test_name.upper()}:")
    for s in test_subj_ids:
        mask = subj_test == s
        X_s = X_test_norm[mask]
        y_s = y_test_all[mask]
        loader_s = make_loader(X_s, y_s)
        if use_dsbn:
            set_dsbn_domain(model, 1)
        acc, bac, kappa, _, _ = evaluate(model, loader_s)

        adabn_acc = adabn_bac = adabn_kappa = None
        tent_acc = tent_bac = tent_kappa = None
        snap_acc = snap_bac = snap_kappa = None
        snap_adabn_acc = snap_adabn_bac = snap_adabn_kappa = None
        model_adapt = copy.deepcopy(model) if (use_adabn or use_tent) else None

        if use_adabn and model_adapt is not None:
            snap = snapshot_bn_stats(model_adapt)
            n_bn = apply_adabn(model_adapt, X_s, DEVICE,
                               batch_size=BATCH_SIZE, n_passes=3)
            if n_bn > 0:
                adabn_acc, adabn_bac, adabn_kappa, _, _ = evaluate(model_adapt, loader_s)
                shift = adabn_summary(snap, model_adapt)
                delta = (adabn_acc - acc) * 100
                print(f"    S{s:02d}: acc={acc*100:.1f}%  AdaBN={adabn_acc*100:.1f}% "
                      f"(Δ{delta:+.1f}%)  {shift}")
            else:
                print(f"    S{s:02d}: acc={acc*100:.1f}%  (no BN layers)")

        if use_tent and model_adapt is not None:
            affine_snap = snapshot_bn_affine(model_adapt)
            n_tent = apply_tent(
                model_adapt,
                X_s,
                DEVICE,
                batch_size=BATCH_SIZE,
                n_steps=tent_steps,
                lr=tent_lr,
                use_adabn_warmup=(not use_adabn),
                adabn_passes=3,
            )
            if n_tent > 0:
                tent_acc, tent_bac, tent_kappa, _, _ = evaluate(model_adapt, loader_s)
                shift = tent_summary(model_adapt, affine_snap)
                base = adabn_acc if adabn_acc is not None else acc
                delta = (tent_acc - base) * 100
                print(f"    S{s:02d}: acc={acc*100:.1f}%  TENT={tent_acc*100:.1f}% "
                      f"(Δ{delta:+.1f}%)  {shift}")

        if use_snapshot and snapshot_states:
            snap_acc, snap_bac, snap_kappa, _, _ = evaluate_snapshot_ensemble(
                snapshot_states, model_name, n_ch, N_TIMES, loader_s
            )
            snap_adabn_acc, snap_adabn_bac, snap_adabn_kappa, _, _ = evaluate_snapshot_ensemble(
                snapshot_states, model_name, n_ch, N_TIMES, loader_s, X_adabn=X_s
            )
            print(f"    S{s:02d}: acc={acc*100:.1f}%  Snap={snap_acc*100:.1f}%  "
                  f"Snap+AdaBN={snap_adabn_acc*100:.1f}%  snaps={len(snapshot_states)}")

        if not use_adabn and not use_tent and not use_snapshot:
            print(f"    S{s:02d}: acc={acc*100:.1f}%  bac={bac*100:.1f}%  κ={kappa:.3f}")

        row = dict(train_ds=train_name, test_ds=test_name, model=model_name,
                   subject=int(s), n_test=int(mask.sum()),
                   acc=round(acc*100, 1), bac=round(bac*100, 1),
                   kappa=round(kappa, 3),
                   adabn_acc=round(adabn_acc*100, 1) if adabn_acc is not None else "",
                   adabn_bac=round(adabn_bac*100, 1) if adabn_bac is not None else "",
                   adabn_kappa=round(adabn_kappa, 3) if adabn_kappa is not None else "",
                   tent_acc=round(tent_acc*100, 1) if tent_acc is not None else "",
                   tent_bac=round(tent_bac*100, 1) if tent_bac is not None else "",
                   tent_kappa=round(tent_kappa, 3) if tent_kappa is not None else "",
                   snap_acc=round(snap_acc*100, 1) if snap_acc is not None else "",
                   snap_bac=round(snap_bac*100, 1) if snap_bac is not None else "",
                   snap_kappa=round(snap_kappa, 3) if snap_kappa is not None else "",
                   snap_adabn_acc=round(snap_adabn_acc*100, 1) if snap_adabn_acc is not None else "",
                   snap_adabn_bac=round(snap_adabn_bac*100, 1) if snap_adabn_bac is not None else "",
                   snap_adabn_kappa=round(snap_adabn_kappa, 3) if snap_adabn_kappa is not None else "")
        results.append(row)
        with open(res_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=res_fields).writerow(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    accs  = [r["acc"]  for r in results]
    bacs  = [r["bac"]  for r in results]
    kaps  = [r["kappa"] for r in results]
    print(f"\n  {'─'*50}")
    print(f"  {train_name.upper()} → {test_name.upper()} | {model_name}")
    print(f"  Accuracy : {np.mean(accs):.2f} ± {np.std(accs):.2f} %")
    print(f"  Bal. Acc : {np.mean(bacs):.2f} ± {np.std(bacs):.2f} %")
    print(f"  Cohen κ  : {np.mean(kaps):.3f} ± {np.std(kaps):.3f}")
    print(f"  Saved    : {res_csv}")

    return np.mean(accs), np.mean(kaps)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  choices=["cho2017", "lee2019"], default="cho2017")
    parser.add_argument("--test",   choices=["cho2017", "lee2019"], default="lee2019")
    parser.add_argument("--both",   action="store_true",
                        help="Run both directions: cho→lee and lee→cho")
    parser.add_argument("--model",  choices=["cspnet", "cspnetcontrastive", "cspnetdann",
                                           "eegnet", "eegnetcontrastive",
                                           "conformer", "conformercontrastive"],
                        default="cspnet")
    parser.add_argument("--lambda_con", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--ea", action="store_true")
    parser.add_argument(
        "--ea_eps_mode", choices=EA_EPS_MODES,
        default=os.environ.get("MI_EA_FLOOR", "relative"),
        help="EA eigenvalue handling (default: MI_EA_FLOOR or relative)",
    )
    parser.add_argument("--tent", action="store_true")
    parser.add_argument("--tent_steps", type=int, default=1)
    parser.add_argument("--tent_lr", type=float, default=1e-3)
    parser.add_argument("--psd_norm", action="store_true",
                        help="Normalize each dataset/channel by label-free RMS power before EA")
    parser.add_argument("--dataset_ea", action="store_true",
                        help="Apply one dataset-level EA transform before optional subject-level EA")
    parser.add_argument("--ea_order", choices=["dataset_subject", "subject_dataset"],
                        default="dataset_subject",
                        help="Order when both dataset-level EA and subject-level EA are enabled")
    parser.add_argument("--dsbn", action="store_true",
                        help="Use domain-specific BN stats: source-domain train, target-domain unlabeled calibration")
    parser.add_argument("--channel_set", choices=["common", "standard_mi"], default="common",
                        help="Channel ordering: all common channels or fixed standard MI 10-20 subset")
    parser.add_argument("--pseudo_label", action="store_true",
                        help="Fine-tune BN/classifier on confident target pseudo-labels")
    parser.add_argument("--pseudo_threshold", type=float, default=0.85)
    parser.add_argument("--pseudo_epochs", type=int, default=20)
    parser.add_argument("--source_weighting", action="store_true",
                        help="Weight source subjects by covariance similarity to target dataset")
    parser.add_argument("--source_weight_tau", type=float, default=0.5)
    parser.add_argument("--snapshot_ensemble", action="store_true",
                        help="Save snapshots at cosine warm-restart cycle ends and ensemble softmax at test time")
    parser.add_argument("--snapshot_T0", type=int, default=50,
                        help="Cosine cycle length; default 50 gives 6 snapshots for 300 epochs")
    parser.add_argument("--session_ea", action="store_true",
                        help="Apply Lee2019 session-level EA after dataset/subject EA")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Validation split seed; defaults to MI_SPLIT_SEED or --seed")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_train_subjects", type=int, default=None,
                        help="Smoke-test only: retain the first N source subjects")
    parser.add_argument("--max_test_subjects", type=int, default=None,
                        help="Smoke-test only: evaluate the first N target subjects")
    args = parser.parse_args()

    if args.split_seed is None:
        args.split_seed = int(os.environ.get("MI_SPLIT_SEED", str(args.seed)))
    SEED = args.seed
    SPLIT_SEED = args.split_seed
    EPOCHS = args.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device : {DEVICE}")
    print(f"Model  : {args.model}")
    print(f"Common channels: 48  n_times: {N_TIMES}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    runtime_kwargs = dict(
        out_dir=args.out_dir,
        ea_eps_mode=args.ea_eps_mode,
        epochs=args.epochs,
        seed=args.seed,
        split_seed=args.split_seed,
        checkpoint_dir=args.checkpoint_dir,
        max_train_subjects=args.max_train_subjects,
        max_test_subjects=args.max_test_subjects,
    )

    if args.both:
        r1 = run_cross("cho2017", "lee2019", model_name=args.model,
                       lambda_con=args.lambda_con, temperature=args.temperature,
                       use_adabn=args.adabn, use_ea=args.ea,
                       use_tent=args.tent, tent_steps=args.tent_steps,
                       tent_lr=args.tent_lr, use_psd_norm=args.psd_norm,
                       use_dataset_ea=args.dataset_ea, ea_order=args.ea_order,
                       use_dsbn=args.dsbn, channel_set=args.channel_set,
                       use_pseudo_label=args.pseudo_label, pseudo_threshold=args.pseudo_threshold,
                       pseudo_epochs=args.pseudo_epochs,
                       use_source_weighting=args.source_weighting,
                       source_weight_tau=args.source_weight_tau,
                       use_snapshot=args.snapshot_ensemble,
                       snapshot_T0=args.snapshot_T0,
                       use_session_ea=args.session_ea, run_id=run_id,
                       **runtime_kwargs)
        r2 = run_cross("lee2019", "cho2017", model_name=args.model,
                       lambda_con=args.lambda_con, temperature=args.temperature,
                       use_adabn=args.adabn, use_ea=args.ea,
                       use_tent=args.tent, tent_steps=args.tent_steps,
                       tent_lr=args.tent_lr, use_psd_norm=args.psd_norm,
                       use_dataset_ea=args.dataset_ea, ea_order=args.ea_order,
                       use_dsbn=args.dsbn, channel_set=args.channel_set,
                       use_pseudo_label=args.pseudo_label, pseudo_threshold=args.pseudo_threshold,
                       pseudo_epochs=args.pseudo_epochs,
                       use_source_weighting=args.source_weighting,
                       source_weight_tau=args.source_weight_tau,
                       use_snapshot=args.snapshot_ensemble,
                       snapshot_T0=args.snapshot_T0,
                       use_session_ea=args.session_ea, run_id=run_id,
                       **runtime_kwargs)
        print(f"\n{'='*60}")
        print(f"  Cho→Lee : {r1[0]:.2f}%  κ={r1[1]:.3f}")
        print(f"  Lee→Cho : {r2[0]:.2f}%  κ={r2[1]:.3f}")
    else:
        run_cross(args.train, args.test, model_name=args.model,
                  lambda_con=args.lambda_con, temperature=args.temperature,
                  use_adabn=args.adabn, use_ea=args.ea,
                  use_tent=args.tent, tent_steps=args.tent_steps,
                  tent_lr=args.tent_lr, use_psd_norm=args.psd_norm,
                  use_dataset_ea=args.dataset_ea, ea_order=args.ea_order,
                  use_dsbn=args.dsbn, channel_set=args.channel_set,
                  use_pseudo_label=args.pseudo_label, pseudo_threshold=args.pseudo_threshold,
                  pseudo_epochs=args.pseudo_epochs,
                  use_source_weighting=args.source_weighting,
                  source_weight_tau=args.source_weight_tau,
                  use_snapshot=args.snapshot_ensemble,
                  snapshot_T0=args.snapshot_T0,
                  use_session_ea=args.session_ea, run_id=run_id,
                  **runtime_kwargs)
