#!/usr/bin/env python
"""Headless feature-cache builder: same logic as the notebook's Bước 3, for
running under nohup/tmux on a rented GPU machine.

Resume-safe. "Complete" means `_progress.json` records every image; the size
of the .npy files proves nothing because they are preallocated at full size.

Example:
    nohup python -m src.p0.cache_cli \
        --dataset circo --data-root /workspace/data/CIRCO \
        --output-dir /workspace/features/circo \
        --batch-size 128 --num-workers 6 > /workspace/logs/cache.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Subset

from ..data.datasets import CIRCODataset, CIRRDataset
from ..models.openclip_utils import load_openclip
from ..seed import set_seed
from .feature_cache import build_feature_cache

FILES = ["global.npy", "local.npy", "image_ids.json", "_progress.json"]


def cache_completed(cache_dir: str) -> int:
    if not all(os.path.exists(f"{cache_dir}/{f}") for f in FILES):
        return 0
    with open(f"{cache_dir}/_progress.json", "r", encoding="utf-8") as f:
        return json.load(f)["completed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Only cache the first N images (smoke test).")
    parser.add_argument("--openclip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip-pretrained", type=str, default="laion2b_s32b_b82k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model, preprocess, _ = load_openclip(args.openclip_model_name, args.openclip_pretrained, device)

    if args.dataset == "circo":
        dataset = CIRCODataset(args.data_root, args.split, "classic", preprocess)
        id_key = "image_id"
    else:
        dataset = CIRRDataset(args.data_root, args.split, "classic", preprocess)
        dataset.image_ids = dataset.image_names
        id_key = "image_name"

    if args.limit is not None:
        ids = list(dataset.image_ids[: args.limit])
        dataset = Subset(dataset, list(range(args.limit)))
        dataset.image_ids = ids

    total = len(dataset)
    done = cache_completed(args.output_dir)
    print(f"Tien do hien tai: {done}/{total}", flush=True)
    if done >= total:
        print("Cache da xong day du, bo qua tinh toan.", flush=True)
        return

    build_feature_cache(
        model, dataset,
        output_dir=args.output_dir,
        id_key=id_key,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    done = cache_completed(args.output_dir)
    assert done == total, f"Cache chua xong: {done}/{total}"
    print(f"CACHE_DONE {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
