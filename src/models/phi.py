"""Textual Inversion Network (phi).

Architecture from Table 5 of the paper "Zero-Shot Composed Image Retrieval
with Textual Inversion" (SEARLE, arXiv:2303.15247):

    Input  -> nn.Linear(d, d*4)
    GELU
    Dropout(0.5)
    Hidden -> nn.Linear(d*4, d*4)
    GELU
    Dropout(0.5)
    Output -> nn.Linear(d*4, d_w)

`d` is the CLIP joint embedding dimension (image/text feature size),
`d_w` is the dimension of the CLIP token embedding space W.
For ViT-B/32, d == d_w == 512. For ViT-L/14, d == d_w == 768.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Phi(nn.Module):
    """MLP that maps a CLIP image feature to a pseudo-word token embedding."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None,
                 output_dim: int | None = None, dropout: float = 0.5):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """image_features: (B, d) CLIP image features -> (B, d_w) pseudo tokens."""
        return self.layers(image_features)
