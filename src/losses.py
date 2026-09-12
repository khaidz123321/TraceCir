"""Loss functions from the paper (Eq. 1, 2, 3, 4, 5).

L_cos    (Eq. 1): cosine loss pulling the pseudo-word prompt features
                  towards the reference image features.
L_gpt    (Eq. 2): GPT-powered contextualized regularization that keeps the
                  pseudo-word token on the CLIP token-embedding manifold.
L_distil (Eq. 4): symmetric contrastive (SimCLR-style) distillation loss
                  between phi's predicted tokens and the OTI pre-generated
                  targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L_cos = 1 - cos(a, b), averaged over the batch. Eq. (1) and Eq. (2)."""
    return (1 - F.cosine_similarity(a, b, dim=-1)).mean()


def distillation_contrastive_loss(
    v_pred: torch.Tensor, v_target: torch.Tensor, temperature: float = 0.25
) -> torch.Tensor:
    """L_distil, Eq. (4): symmetric InfoNCE between predicted and target
    pseudo-word tokens within a batch.

    v_pred, v_target: (B, d_w), assumed L2-normalizable feature vectors.
    """
    v_pred = F.normalize(v_pred, dim=-1)
    v_target = F.normalize(v_target, dim=-1)

    logits_pred_target = v_pred @ v_target.t() / temperature  # (B, B)
    logits_pred_pred = v_pred @ v_pred.t() / temperature
    logits_target_target = v_target @ v_target.t() / temperature
    logits_target_pred = v_target @ v_pred.t() / temperature

    batch_size = v_pred.shape[0]
    labels = torch.arange(batch_size, device=v_pred.device)
    eye = torch.eye(batch_size, dtype=torch.bool, device=v_pred.device)

    # For "predicted as anchor": positives are pred<->target on the diagonal,
    # negatives are all target rows plus the other predicted rows (off-diag).
    neg_pred = logits_pred_pred.masked_fill(eye, float("-inf"))
    logits_a = torch.cat([logits_pred_target, neg_pred], dim=1)
    loss_a = F.cross_entropy(logits_a, labels)

    neg_target = logits_target_target.masked_fill(eye, float("-inf"))
    logits_b = torch.cat([logits_target_pred, neg_target], dim=1)
    loss_b = F.cross_entropy(logits_b, labels)

    return loss_a + loss_b


def oti_loss(
    l_cos: torch.Tensor, l_gpt: torch.Tensor, lambda_cos: float = 1.0, lambda_gpt: float = 0.5
) -> torch.Tensor:
    """L_OTI, Eq. (3)."""
    return lambda_cos * l_cos + lambda_gpt * l_gpt


def phi_loss(
    l_distil: torch.Tensor, l_gpt: torch.Tensor, lambda_distil: float = 1.0, lambda_gpt: float = 0.75
) -> torch.Tensor:
    """L_phi, Eq. (5)."""
    return lambda_distil * l_distil + lambda_gpt * l_gpt
