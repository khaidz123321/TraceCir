#!/usr/bin/env python
"""Sec. 7 sanity check: where does a source phrase ground in the reference image?

For randomly chosen compiled queries, draws the grounding weights
alpha_k = softmax_k(u^T v_k^r / tau_g) of each REMOVE / PRESERVE / REPLACE source
phrase as a heatmap over the 8x8 local grid of the reference image. The grid
covers the centre crop CLIP sees (shorter side resized to 224, then a 224 centre
crop), so the overlay is drawn on that crop.

Example:
    python tools/make_grounding_heatmaps.py --data-root /workspace/data/CIRCO \
        --feature-cache-dir /workspace/features/circo \
        --compiled-queries-path /workspace/outputs/circo_compiled_q36.jsonl \
        --out /workspace/outputs/grounding_heatmaps.png
"""
from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import CIRCODataset  # noqa: E402
from src.models.openclip_utils import encode_text, load_openclip  # noqa: E402
from src.p0.compiler import load_compiled_queries  # noqa: E402
from src.p0.feature_cache import FeatureCache  # noqa: E402
from src.p0.scoring_transition import ground_source  # noqa: E402

TILE = 224
GRID = 8


def center_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = TILE / min(w, h)
    image = image.convert("RGB").resize((max(TILE, round(w * scale)), max(TILE, round(h * scale))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - TILE) // 2, (h - TILE) // 2
    return image.crop((left, top, left + TILE, top + TILE))


def overlay(image: Image.Image, alpha_grid: np.ndarray) -> Image.Image:
    heat = alpha_grid.reshape(GRID, GRID)
    heat = heat / max(heat.max(), 1e-12)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((TILE, TILE), Image.BILINEAR)
    heat_arr = np.asarray(heat_img, dtype=np.float32)[..., None] / 255.0
    base = np.asarray(image, dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    return Image.fromarray((base * (1 - 0.65 * heat_arr) + red * 0.65 * heat_arr).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--compiled-queries-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tau-g", type=float, default=0.02)
    parser.add_argument("--num", type=int, default=12, help="Number of phrases to draw.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _, tokenizer = load_openclip("ViT-L-14", "laion2b_s32b_b82k", device)
    cache = FeatureCache(args.feature_cache_dir)
    compiled = load_compiled_queries(args.compiled_queries_path)
    items = CIRCODataset(args.data_root, args.split, "relative", preprocess=lambda image: image)

    candidates = []  # (item index, operation, source phrase)
    for index in range(len(items)):
        spec = compiled.get(str(items.annotations[index]["reference_img_id"])) if hasattr(items, "annotations") else None
        if spec is None:
            continue
        for atom in spec.atoms:
            if atom.operation in ("REMOVE", "PRESERVE", "REPLACE") and atom.source_state:
                candidates.append((index, atom.operation, atom.source_state))
    random.Random(args.seed).shuffle(candidates)
    picked = candidates[: args.num]

    tiles = []
    for index, operation, phrase in picked:
        item = items[index]
        reference_id = item["reference_img_id"]
        u = encode_text(model, tokenizer, [phrase], device)
        ref_local = cache.local_vectors(reference_id).float().to(device)
        alpha = torch.softmax(u @ ref_local.t() / args.tau_g, dim=-1)[0].cpu().numpy()
        tile = Image.new("RGB", (TILE, TILE + 46), "white")
        tile.paste(overlay(center_crop(item["reference_image"]), alpha), (0, 0))
        draw = ImageDraw.Draw(tile)
        caption = f"{operation}: {phrase}"
        draw.multiline_text((3, TILE + 2), "\n".join(textwrap.wrap(caption, 36)[:3]), fill="black")
        # z- is the alpha-weighted mean of the local vectors: report how peaked alpha is.
        tiles.append((tile, float(alpha.max())))
        print(f"{index:3d} {operation:8s} max_alpha={alpha.max():.2f} | {phrase}")

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + 46)), "white")
    for i, (tile, _) in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * TILE, (i // cols) * (TILE + 46)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
