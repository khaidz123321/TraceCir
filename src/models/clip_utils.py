"""Utilities to load CLIP and run text encoding with an injected pseudo-word
token embedding (v*) at inference / training time.

OpenAI's CLIP `encode_text` only accepts token ids and looks them up in a
fixed embedding table, so it cannot ingest an arbitrary vector as one of the
"words" in the sentence. `encode_with_pseudo_tokens` reimplements the text
tower forward pass so that, after building the normal token embeddings, it
overwrites the embedding of a placeholder token ("$") with the externally
computed pseudo-token vector(s) before running the transformer + projection.
This is exactly what turns "a photo of $ that <caption>" into
"a photo of S* that <caption>" as described in Sec. 3 of the paper.
"""
from __future__ import annotations

from typing import Iterable

import clip
import torch
import torch.nn as nn

PSEUDO_TOKEN = "$"
CONTEXT_LENGTH = 77


def load_clip(clip_model_name: str = "ViT-B/32", device: str | torch.device = "cpu"):
    """Load a frozen CLIP model and its preprocessing transform."""
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model = model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


def get_placeholder_token_id(device: str | torch.device = "cpu") -> int:
    """Token id CLIP's BPE tokenizer assigns to the single placeholder char."""
    tokens = clip.tokenize([PSEUDO_TOKEN]).to(device)[0]
    # tokens[0] is <|startoftext|>; tokens[1] is the placeholder's own token.
    return int(tokens[1].item())


def tokenize(texts: Iterable[str], context_length: int = CONTEXT_LENGTH) -> torch.Tensor:
    return clip.tokenize(list(texts), context_length=context_length, truncate=True)


def encode_with_pseudo_tokens(
    clip_model: nn.Module,
    text_tokens: torch.Tensor,
    pseudo_tokens: torch.Tensor,
    placeholder_id: int,
) -> torch.Tensor:
    """Run CLIP's text tower, substituting the placeholder token embedding.

    Args:
        clip_model: a loaded (and frozen) CLIP model.
        text_tokens: (B, context_length) long tensor from `tokenize`, where
            each row contains exactly one occurrence of `placeholder_id`
            (the "$" in a template such as "a photo of $ that ...").
        pseudo_tokens: (B, d_w) tensor, one pseudo-word embedding per row,
            typically produced by OTI or by the Phi network.
        placeholder_id: token id returned by `get_placeholder_token_id`.

    Returns:
        (B, d) text features, in the same space as CLIP's normal
        `encode_text` output.
    """
    dtype = clip_model.dtype
    x = clip_model.token_embedding(text_tokens).type(dtype)  # (B, L, d_w)

    mask = text_tokens == placeholder_id  # (B, L)
    if not torch.all(mask.sum(dim=1) == 1):
        raise ValueError(
            "Each prompt must contain the placeholder token "
            f"'{PSEUDO_TOKEN}' exactly once."
        )
    batch_idx, pos_idx = mask.nonzero(as_tuple=True)
    x = x.clone()
    x[batch_idx, pos_idx] = pseudo_tokens.type(dtype)[batch_idx]

    x = x + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(dtype)

    eot_idx = text_tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot_idx] @ clip_model.text_projection
    return x


# A handful of generic templates used to sample T when running OTI, following
# the same spirit as PALAVRA [8] / CLIP's zero-shot prompt ensembling.
OTI_TEMPLATES = [
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
    "a close-up photo of {}",
    "a cropped photo of {}",
    "a good photo of {}",
    "{} in a photo",
]


def build_oti_prompt(template: str = "a photo of {}") -> str:
    return template.format(PSEUDO_TOKEN)


def build_cir_prompt(relative_caption: str) -> str:
    """Template used at CIR inference time: "a photo of S* that <caption>".

    Real dataset captions occasionally contain a literal "$" (e.g. a price
    mention: "it looks $ 10 cheaper"). Since PSEUDO_TOKEN=="$" is what
    `encode_with_pseudo_tokens` uses to locate the single pseudo-word
    position, an un-escaped "$" in the caption would create a second
    occurrence and trip its "exactly once" check. Replace any literal
    placeholder characters in the caption with a plain word first so only
    the one we insert remains.
    """
    safe_caption = relative_caption.replace(PSEUDO_TOKEN, "dollar")
    return f"a photo of {PSEUDO_TOKEN} that {safe_caption}"
