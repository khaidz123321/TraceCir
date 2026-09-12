#!/usr/bin/env python
"""Stage 3: run zero-shot CIR inference + evaluation with a trained Phi
network (SEARLE) on FashionIQ, CIRR, or CIRCO.

Example:
    python -m src.scripts.validate --dataset cirr --split val \
        --data-root /data/CIRR --phi-checkpoint checkpoints/phi_b32/phi_final.pt \
        --clip-model-name ViT-B/32
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import CIRCODataset, CIRRDataset, FashionIQDataset
from ..eval import mean_average_precision_at_k, recall_at_k
from ..models.clip_utils import build_cir_prompt, encode_with_pseudo_tokens, get_placeholder_token_id, load_clip, tokenize
from ..models.phi import Phi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fashioniq", "cirr", "circo"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dress-types", nargs="+", default=["dress", "shirt", "toptee"])
    parser.add_argument("--phi-checkpoint", type=str, required=True)
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def extract_index_features(clip_model, loader, device, id_key: str):
    feats, ids = [], []
    for batch in tqdm(loader, desc="Indexing"):
        images = batch["image"].to(device)
        image_feats = F.normalize(clip_model.encode_image(images).float(), dim=-1)
        feats.append(image_feats.cpu())
        ids.extend(batch[id_key] if isinstance(batch[id_key], list) else batch[id_key].tolist())
    return torch.cat(feats, dim=0), ids


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    clip_model, preprocess = load_clip(args.clip_model_name, device)
    placeholder_id = get_placeholder_token_id(device)
    feature_dim = clip_model.token_embedding.embedding_dim

    phi = Phi(input_dim=feature_dim, output_dim=feature_dim).to(device)
    phi.load_state_dict(torch.load(args.phi_checkpoint, map_location=device))
    phi.eval()

    if args.dataset == "fashioniq":
        index_ds = FashionIQDataset(args.data_root, args.split, args.dress_types, "classic", preprocess)
        query_ds = FashionIQDataset(args.data_root, args.split, args.dress_types, "relative", preprocess)
        id_key = "image_name"
    elif args.dataset == "cirr":
        index_ds = CIRRDataset(args.data_root, args.split, "classic", preprocess)
        query_ds = CIRRDataset(args.data_root, args.split, "relative", preprocess)
        id_key = "image_name"
    else:
        index_ds = CIRCODataset(args.data_root, args.split, "classic", preprocess)
        query_ds = CIRCODataset(args.data_root, args.split, "relative", preprocess)
        id_key = "image_id"

    index_loader = DataLoader(index_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    index_features, index_ids = extract_index_features(clip_model, index_loader, device, id_key)
    id_to_pos = {name: i for i, name in enumerate(index_ids)}

    def _keep_variable_length(batch: list[dict]) -> dict:
        """Collate that stacks tensors normally but leaves list-valued /
        variable-length fields (e.g. CIRCO's per-query gt_img_ids) as plain
        Python lists instead of trying to stack them into a tensor."""
        from torch.utils.data._utils.collate import default_collate

        keys = batch[0].keys()
        out: dict = {}
        for key in keys:
            values = [sample[key] for sample in batch]
            if key in ("gt_img_ids", "relative_captions", "member_set"):
                out[key] = values
            else:
                out[key] = default_collate(values)
        return out

    query_loader = DataLoader(
        query_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=_keep_variable_length,
    )

    all_query_feats = []
    target_indices = []
    gt_lists: list[list[int]] = []

    with torch.no_grad():
        for batch in tqdm(query_loader, desc="Querying"):
            ref_images = batch["reference_image"].to(device)
            ref_feats = clip_model.encode_image(ref_images).float()
            v_star = phi(ref_feats)

            if args.dataset == "fashioniq":
                captions = [" and ".join(c) for c in batch["relative_captions"]]
            else:
                captions = batch["relative_caption"]

            prompts = [build_cir_prompt(c) for c in captions]
            text_tokens = tokenize(prompts).to(device)
            text_feats = encode_with_pseudo_tokens(clip_model, text_tokens, v_star, placeholder_id)
            text_feats = F.normalize(text_feats, dim=-1)
            all_query_feats.append(text_feats.cpu())

            if args.dataset != "circo" and args.split != "test1":
                target_indices.extend(id_to_pos[name] for name in batch["target_name"])
            if args.dataset == "circo":
                gt_lists.extend(
                    [id_to_pos[i] for i in gts if i in id_to_pos]
                    for gts in batch["gt_img_ids"]
                )

    query_features = torch.cat(all_query_feats, dim=0)
    similarity = query_features @ index_features.t()

    if args.dataset == "circo":
        print("Note: CIRCO ground-truth batching for mAP needs per-sample gt_img_ids; "
              "see docs/DATASETS.md for the exact collate function used in the paper's evaluation server.")
        results = mean_average_precision_at_k(similarity, gt_lists, [5, 10, 25, 50])
        print({f"mAP@{k}": v for k, v in results.items()})
    else:
        k_values = [10, 50] if args.dataset == "fashioniq" else [1, 5, 10, 50]
        results = recall_at_k(similarity, torch.tensor(target_indices), k_values)
        print({f"R@{k}": v for k, v in results.items()})


if __name__ == "__main__":
    main()
