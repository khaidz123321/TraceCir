"""OpenCLIP loading + global/local visual feature extraction for TAPR/P0
(Sec. IV-A of the TAPR paper; Sec. 4 of the P0 protocol).

Candidate evidence bank per image I (TAPR Eq. 3):
    E(I) = (v0(I), Vl(I)),   Vl(I) = {v1(I), ..., vM(I)}, M = 64
    ||v0(I)||_2 = ||vj(I)||_2 = 1

v0 is the usual pooled/global CLIP image embedding. Vl are *local* (patch)
embeddings projected into the same joint embedding space as v0, so they can
be compared with text probe vectors using the same dot product. OpenCLIP's
ViT exposes pre-projection patch tokens via `output_tokens=True`; we apply
the vision tower's own projection matrix to those tokens (a standard
"dense CLIP" trick) and average-pool the native patch grid down to an
8x8 = 64 grid, matching the paper's setting for ViT-L/14 at 224 resolution
(16x16 native grid -> 8x8 pooled).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import open_clip


def load_openclip(
    model_name: str = "ViT-L-14",
    pretrained: str = "laion2b_s32b_b82k",
    device: str | torch.device = "cpu",
):
    """Load a frozen OpenCLIP model + its preprocessing transform + tokenizer."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Ask the vision tower to also return per-patch tokens alongside the
    # pooled embedding (see module docstring).
    model.visual.output_tokens = True
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_image_global_local(
    model, images: torch.Tensor, local_grid: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of images into (global, local) visual vectors.

    Args:
        images: (B, C, H, W) preprocessed images.
        local_grid: target side length of the local vector grid (8 -> M=64).

    Returns:
        v0: (B, d) L2-normalized global visual vectors.
        v_local: (B, M, d) L2-normalized local visual vectors, M = local_grid**2.
    """
    pooled, tokens = model.visual(images)  # pooled: (B, d); tokens: (B, N, pool_dim)
    proj = model.visual.proj
    if proj is not None:
        tokens = tokens @ proj  # project patch tokens into the same joint space as pooled

    batch_size, num_tokens, dim = tokens.shape
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        raise ValueError(
            f"Expected a square patch grid, got {num_tokens} tokens "
            f"(not a perfect square) -- check the model/input resolution."
        )

    grid = tokens.transpose(1, 2).reshape(batch_size, dim, side, side)  # (B, d, side, side)
    pooled_grid = F.adaptive_avg_pool2d(grid, (local_grid, local_grid))  # (B, d, g, g)
    v_local = pooled_grid.flatten(2).transpose(1, 2)  # (B, g*g, d)

    v0 = F.normalize(pooled.float(), dim=-1)
    v_local = F.normalize(v_local.float(), dim=-1)
    return v0, v_local


@torch.no_grad()
def encode_text(model, tokenizer, texts: list[str], device: str | torch.device) -> torch.Tensor:
    """L2-normalized OpenCLIP text-tower features for a list of phrases."""
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens).float()
    return F.normalize(features, dim=-1)
