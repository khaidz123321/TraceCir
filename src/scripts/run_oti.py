#!/usr/bin/env python
"""Stage 1 (offline, run once): run OTI over the whole pre-training image
set to build the pool of pseudo-word targets V* = {v*_1, ..., v*_N} that
`train_phi.py` will later distill into the Phi network.

Also writes the concepts.json file (image_name -> assigned concept words)
that `train_phi.py` requires as --concepts-path, since both stages must
sample concepts/GPT phrases for the exact same images.

Example:
    python -m src.scripts.run_oti \
        --image-dir /data/ImageNet1K/test \
        --vocab-path data/open_images_v7_classes.txt \
        --gpt-phrases-path data/gpt_phrases.jsonl \
        --output-path data/oti_targets.pt \
        --concepts-output-path data/concepts.json \
        --clip-model-name ViT-B/32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..concepts import assign_concepts, embed_vocab, load_gpt_phrases, load_open_images_vocab, save_concepts
from ..data.datasets import UnlabeledImageFolder
from ..models.clip_utils import get_placeholder_token_id, load_clip
from ..oti import OTIConfig, run_oti_for_batch
from ..seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--gpt-phrases-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--concepts-output-path", type=str, required=True,
                         help="Where to save the image_name -> concepts JSON for train_phi.py's --concepts-path.")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--oti-steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    clip_model, preprocess = load_clip(args.clip_model_name, device)
    placeholder_id = get_placeholder_token_id(device)

    dataset = UnlabeledImageFolder(args.image_dir, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vocab = load_open_images_vocab(args.vocab_path)
    vocab_embeddings = embed_vocab(clip_model, vocab, device)
    concepts = assign_concepts(clip_model, loader, vocab, vocab_embeddings, device, top_k=args.top_k)
    save_concepts(concepts, args.concepts_output_path)
    print(f"Saved concept assignments for {len(concepts)} images to {args.concepts_output_path}")
    gpt_phrases = load_gpt_phrases(args.gpt_phrases_path)

    config = OTIConfig(steps=args.oti_steps, lr=args.lr)

    all_names: list[str] = []
    all_targets: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Running OTI"):
        images = batch["image"].to(device)
        names = batch["image_name"]
        v_star = run_oti_for_batch(
            clip_model, images, names, concepts, gpt_phrases, placeholder_id, device, config
        )
        all_names.extend(names)
        all_targets.append(v_star.cpu())

    targets = torch.cat(all_targets, dim=0)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": all_names, "targets": targets}, output_path)
    print(f"Saved {len(all_names)} OTI pseudo-word targets to {output_path}")


if __name__ == "__main__":
    main()
