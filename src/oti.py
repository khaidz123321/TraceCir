"""Optimization-based Textual Inversion (OTI), Sec. 3.1.

For a batch of images, directly optimize a pseudo-word token embedding
v* per image (no shared network / weights) so that:
  - "a photo of v*" matches the image in CLIP's joint space (L_cos, Eq. 1)
  - a GPT-generated elaboration of the image's concept, with the concept
    word swapped for v*, still matches the same elaboration with the real
    concept word (L_gpt, Eq. 2)

This is slow (run once per image, ~30s/image for ViT-L/14 on an A100 per
Appendix A) and is only meant to be run offline to build the pool of
pseudo-word targets V* that `train_phi.py` later distills into the Phi
network.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .concepts import sample_gpt_pair
from .losses import cosine_loss, oti_loss
from .models.clip_utils import (
    OTI_TEMPLATES,
    build_oti_prompt,
    encode_with_pseudo_tokens,
    tokenize,
)


@dataclass
class OTIConfig:
    steps: int = 350
    lr: float = 2e-2
    lambda_cos: float = 1.0
    lambda_gpt: float = 0.5
    ema_decay: float = 0.99
    weight_decay: float = 0.01


def run_oti_for_batch(
    clip_model,
    images: torch.Tensor,
    image_names: list[str],
    concepts: dict[str, list[str]],
    gpt_phrases: dict[str, list[str]],
    placeholder_id: int,
    device,
    config: OTIConfig = OTIConfig(),
) -> torch.Tensor:
    """Run OTI for a batch of images and return their pseudo-word tokens.

    Args:
        images: (B, C, H, W) CLIP-preprocessed images.
        image_names: names used to look up `concepts` (top-k concept words
            assigned by `concepts.assign_concepts`) and, transitively,
            `gpt_phrases`.
        gpt_phrases: mapping concept -> list of pre-generated GPT-Neo
            continuations of "a photo of {concept}" (see concepts.py).

    Returns:
        (B, d_w) tensor of optimized pseudo-word token embeddings, detached.
    """
    batch_size = images.shape[0]
    d_w = clip_model.token_embedding.embedding_dim

    with torch.no_grad():
        image_features = F.normalize(clip_model.encode_image(images).float(), dim=-1)

    v_star = torch.randn(batch_size, d_w, device=device) * 0.02
    v_star.requires_grad_(True)
    v_star_ema = v_star.detach().clone()

    optimizer = AdamW([v_star], lr=config.lr, weight_decay=config.weight_decay)

    for _step in range(config.steps):
        optimizer.zero_grad()

        template = random.choice(OTI_TEMPLATES)
        prompt = build_oti_prompt(template)
        text_tokens = tokenize([prompt] * batch_size).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, text_tokens, v_star, placeholder_id)
        text_features = F.normalize(text_features, dim=-1)
        l_cos = cosine_loss(image_features, text_features)

        gpt_texts, gpt_star_texts = sample_gpt_pair(image_names, concepts, gpt_phrases)

        gpt_tokens = tokenize(gpt_texts).to(device)
        with torch.no_grad():
            t_hat = F.normalize(clip_model.encode_text(gpt_tokens).float(), dim=-1)

        gpt_star_tokens = tokenize(gpt_star_texts).to(device)
        t_hat_star = encode_with_pseudo_tokens(clip_model, gpt_star_tokens, v_star, placeholder_id)
        t_hat_star = F.normalize(t_hat_star, dim=-1)
        l_gpt = cosine_loss(t_hat, t_hat_star)

        loss = oti_loss(l_cos, l_gpt, config.lambda_cos, config.lambda_gpt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            v_star_ema.mul_(config.ema_decay).add_(v_star.detach(), alpha=1 - config.ema_decay)

    return v_star_ema.detach()
