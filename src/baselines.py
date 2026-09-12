"""Training-free baselines the paper compares SEARLE against (Sec. 5.1):
Image-only, Text-only, and Image+Text. All three only need a frozen CLIP,
no OTI / Phi training at all.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.clip_utils import tokenize


@torch.no_grad()
def image_only_query_features(clip_model, reference_images: torch.Tensor) -> torch.Tensor:
    """Retrieve images most similar to the reference image, ignoring the caption."""
    feats = clip_model.encode_image(reference_images).float()
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def text_only_query_features(clip_model, relative_captions: list[str], device) -> torch.Tensor:
    """Retrieve using only the relative caption's CLIP text features."""
    tokens = tokenize(relative_captions).to(device)
    feats = clip_model.encode_text(tokens).float()
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def image_plus_text_query_features(
    clip_model, reference_images: torch.Tensor, relative_captions: list[str], device
) -> torch.Tensor:
    """Sum of (normalized) reference-image and relative-caption CLIP features."""
    image_feats = F.normalize(clip_model.encode_image(reference_images).float(), dim=-1)
    tokens = tokenize(relative_captions).to(device)
    text_feats = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
    return F.normalize(image_feats + text_feats, dim=-1)
