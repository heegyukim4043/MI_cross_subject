"""
Discriminator-based selective training modules for LOSO MI EEG.

This file provides the first-stage building blocks for the planned
discriminator-based selective training pipeline:

  EEG epoch
    -> tangent-space feature extractor
    -> tangent autoencoder
    -> discriminator with:
         - quality head
         - domain head through GRL

The current repository does not wire this module into train_loso.py yet.
It is intended as a reusable component for future integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spd_net import CovarianceLayer, LogMapLayer, VectorizeLayer


class _GradientReversalFn(torch.autograd.Function):
    """Gradient reversal for domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    """Applies identity in forward pass and sign-flipped gradient in backward."""

    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = float(lambda_grl)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.lambda_grl)


class TangentSpaceExtractor(nn.Module):
    """
    EEG -> covariance -> LogMap -> vectorized tangent feature.

    Input:
      x: (B, C, T)

    Output:
      z_tangent: (B, C * (C + 1) // 2)
    """

    def __init__(self, cov_eps: float = 1e-5, log_eps: float = 1e-7):
        super().__init__()
        self.cov = CovarianceLayer(eps=cov_eps)
        self.logmap = LogMapLayer(eps=log_eps)
        self.vec = VectorizeLayer()

    def forward(self, x):
        x = self.cov(x)
        x = self.logmap(x)
        return self.vec(x)


class TangentAutoEncoder(nn.Module):
    """
    Autoencoder for self-supervised tangent-space reconstruction quality.

    Returns:
      z_hat : reconstructed tangent-space vector
      h     : latent embedding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z_tangent):
        h = self.encoder(z_tangent)
        z_hat = self.decoder(h)
        return z_hat, h

    @torch.no_grad()
    def reconstruction_error(self, z_tangent):
        """Per-sample MSE reconstruction error."""
        z_hat, _ = self.forward(z_tangent)
        return F.mse_loss(z_hat, z_tangent, reduction="none").mean(dim=1)

    @torch.no_grad()
    def quality_label(self, z_tangent):
        """
        Continuous self-supervised quality label in (0, 1].

        quality_label = 1 / (1 + reconstruction_error)
        """
        err = self.reconstruction_error(z_tangent)
        return 1.0 / (1.0 + err)


class DiscriminatorSelectiveModel(nn.Module):
    """
    Shared encoder with quality and domain heads.

    The domain branch uses a Gradient Reversal Layer (GRL) so that the
    shared encoder learns features that are less predictive of subject ID.
    """

    def __init__(
        self,
        input_dim: int,
        n_subjects: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.2,
        lambda_grl: float = 1.0,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )
        self.quality_head = nn.Linear(embed_dim, 1)
        self.grl = GradientReversal(lambda_grl=lambda_grl)
        self.domain_head = nn.Linear(embed_dim, n_subjects)

    def forward(self, z_tangent):
        h = self.shared(z_tangent)
        quality_logit = self.quality_head(h).squeeze(-1)
        domain_logits = self.domain_head(self.grl(h))
        return {
            "embedding": h,
            "quality_logit": quality_logit,
            "domain_logits": domain_logits,
        }

    @staticmethod
    def quality_score(quality_logit):
        """Map quality logit to [0, 1]."""
        return torch.sigmoid(quality_logit)

    @staticmethod
    def domain_score(domain_logits):
        """
        Higher is better for selection.

        domain_score = 1 - max_softmax(subject_logits)
        """
        prob = torch.softmax(domain_logits, dim=-1)
        conf = prob.max(dim=-1).values
        return 1.0 - conf

    def final_score(self, quality_logit, domain_logits):
        """Combined score = quality * domain-invariance."""
        q = self.quality_score(quality_logit)
        d = self.domain_score(domain_logits)
        return q * d


class TangentSelectiveDiscriminator(nn.Module):
    """
    Full convenience wrapper:

      EEG epoch -> tangent-space feature -> discriminator

    This module is useful when the training code wants to feed raw epochs
    directly and avoid managing the tangent extractor separately.
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.2,
        lambda_grl: float = 1.0,
        cov_eps: float = 1e-5,
        log_eps: float = 1e-7,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.extractor = TangentSpaceExtractor(cov_eps=cov_eps, log_eps=log_eps)
        input_dim = n_channels * (n_channels + 1) // 2
        self.discriminator = DiscriminatorSelectiveModel(
            input_dim=input_dim,
            n_subjects=n_subjects,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            lambda_grl=lambda_grl,
        )

    def forward(self, x):
        z_tangent = self.extractor(x)
        out = self.discriminator(z_tangent)
        out["z_tangent"] = z_tangent
        out["final_score"] = self.discriminator.final_score(
            out["quality_logit"], out["domain_logits"]
        )
        return out


def discriminator_losses(
    class_logits=None,
    y_true=None,
    quality_logit=None,
    quality_label=None,
    domain_logits=None,
    subject_id=None,
    lambda_q: float = 1.0,
    lambda_d: float = 1.0,
):
    """
    Utility to compute the planned loss terms for integration code.

    Returns a dict of scalar losses. Missing terms are omitted.
    """
    losses = {}
    total = 0.0

    if class_logits is not None and y_true is not None:
        l_class = F.cross_entropy(class_logits, y_true)
        losses["class"] = l_class
        total = total + l_class

    if quality_logit is not None and quality_label is not None:
        l_quality = F.mse_loss(torch.sigmoid(quality_logit), quality_label.float())
        losses["quality"] = l_quality
        total = total + float(lambda_q) * l_quality

    if domain_logits is not None and subject_id is not None:
        l_domain = F.cross_entropy(domain_logits, subject_id)
        losses["domain"] = l_domain
        total = total + float(lambda_d) * l_domain

    losses["total"] = total if isinstance(total, torch.Tensor) else torch.tensor(total)
    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cho2017-style sanity check
    B, C, T = 8, 64, 257
    n_subjects = 52
    x = torch.randn(B, C, T, device=device)

    extractor = TangentSpaceExtractor().to(device)
    z = extractor(x)
    print(f"Tangent feature : {tuple(z.shape)}")

    ae = TangentAutoEncoder(input_dim=z.shape[1]).to(device)
    z_hat, h = ae(z)
    q_label = ae.quality_label(z)
    print(f"AE recon       : {tuple(z_hat.shape)}")
    print(f"AE latent      : {tuple(h.shape)}")
    print(f"Quality label  : {tuple(q_label.shape)}")

    disc = TangentSelectiveDiscriminator(
        n_channels=C,
        n_subjects=n_subjects,
        hidden_dim=256,
        embed_dim=128,
        lambda_grl=1.0,
    ).to(device)
    out = disc(x)
    print(f"Embedding      : {tuple(out['embedding'].shape)}")
    print(f"Quality logit  : {tuple(out['quality_logit'].shape)}")
    print(f"Domain logits  : {tuple(out['domain_logits'].shape)}")
    print(f"Final score    : {tuple(out['final_score'].shape)}")
