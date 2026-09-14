#!/usr/bin/env python
"""P0 Step 0 (protocol Sec. 3): run the TAPR-style E0 baseline on CIRCO
and/or CIRR validation and report the metrics required by Table 4
(CIRCO mAP@5/@10; CIRR Recall@1/@5/@10).

Prerequisites (run once, before this script):
  1. `src.p0.feature_cache.build_feature_cache` for the dataset's classic
     split -> features/<dataset>/{global.npy,local.npy,image_ids.json}.
  2. `src.p0.compiler.run_compiler_batch` over the dataset's relative
     (query) split -> a JSONL cache of TransitionSpec per query, AUDITED
     per Sec. 5.3 before trusting full-benchmark numbers.

Example:
    python -m src.p0.run_e0 \
        --dataset circo --split val --data-root data/CIRCO \
        --feature-cache-dir features/circo \
        --compiled-queries-path data/circo_compiled.jsonl \
        --openclip-pretrained laion2b_s32b_b82k
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from ..data.datasets import CIRCODataset, CIRRDataset
from ..eval import mean_average_precision_at_k, recall_at_k, recall_subset_at_k
from ..models.openclip_utils import encode_text, load_openclip
from ..seed import set_seed
from .compiler import load_compiled_queries
from .scoring import E0Weights, e0_score
from .feature_cache import FeatureCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--feature-cache-dir", type=str, required=True)
    parser.add_argument("--compiled-queries-path", type=str, required=True)
    parser.add_argument("--openclip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip-pretrained", type=str, default="laion2b_s32b_b82k")
    parser.add_argument("--tau-local", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _query_id(dataset: str, item: dict) -> str:
    if dataset == "circo":
        return str(item["reference_img_id"])
    return item["reference_name"] + "|" + str(item["pair_id"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model, preprocess, tokenizer = load_openclip(
        args.openclip_model_name, args.openclip_pretrained, device
    )
    cache = FeatureCache(args.feature_cache_dir)
    compiled = load_compiled_queries(args.compiled_queries_path)

    parse_failures = sum(1 for spec in compiled.values() if not spec.parse_ok)
    if parse_failures:
        print(
            f"Warning: {parse_failures}/{len(compiled)} compiler outputs failed to "
            "parse (see Sec. 5.3 manual audit) -- their queries will fall back to "
            "Target-only scoring."
        )

    if args.dataset == "circo":
        query_ds = CIRCODataset(args.data_root, args.split, "relative", preprocess)
    else:
        query_ds = CIRRDataset(args.data_root, args.split, "relative", preprocess)

    weights = E0Weights(tau_local=args.tau_local)

    similarity_rows: list[torch.Tensor] = []
    target_indices: list[int] = []
    gt_lists: list[list[int]] = []
    member_indices: list[list[int]] = []

    for item in tqdm(query_ds, desc=f"E0 scoring ({args.dataset} {args.split})"):
        query_id = _query_id(args.dataset, item)
        spec = compiled.get(query_id)

        with torch.no_grad():
            target_vec = encode_text(model, tokenizer, [spec.target if spec and spec.parse_ok else item["relative_caption"]], device)[0]

        reference_id = item["reference_img_id"] if args.dataset == "circo" else item["reference_name"]
        reference_local = cache.local_vectors(reference_id).to(device)

        def probe_vecs(operation: str, field: str) -> torch.Tensor | None:
            if spec is None:
                return None
            phrases = [getattr(a, field) for a in spec.atoms_by_operation(operation) if getattr(a, field)]
            if not phrases:
                return None
            with torch.no_grad():
                return encode_text(model, tokenizer, phrases, device)

        add_probes = probe_vecs("ADD", "target_state")
        preserve_probes = probe_vecs("PRESERVE", "source_state")
        remove_probes = probe_vecs("REMOVE", "source_state")

        scores = torch.empty(len(cache))
        for row in range(len(cache)):
            cand_global = cache.global_features[row].to(device)
            cand_local = cache.local_features[row].to(device)
            scores[row] = e0_score(
                target_vec, add_probes, preserve_probes, remove_probes,
                reference_local, cand_global, cand_local, weights,
            )
        similarity_rows.append(scores)

        if args.dataset == "circo":
            gts = [cache.row_index(i) for i in item["gt_img_ids"] if i in cache._id_to_row]
            gt_lists.append(gts)
        else:
            if "target_name" in item:
                target_indices.append(cache.row_index(item["target_name"]))
            member_indices.append([cache.row_index(m) for m in item["member_set"] if m in cache._id_to_row])

    similarity = torch.stack(similarity_rows, dim=0)

    if args.dataset == "circo":
        results = mean_average_precision_at_k(similarity, gt_lists, [5, 10, 25, 50])
        print({f"E0 mAP@{k}": v for k, v in results.items()})
    else:
        results = recall_at_k(similarity, torch.tensor(target_indices), [1, 5, 10, 50])
        print({f"E0 R@{k}": v for k, v in results.items()})
        subset_results = recall_subset_at_k(similarity, torch.tensor(target_indices), member_indices, [1, 2, 3])
        print({f"E0 R_subset@{k}": v for k, v in subset_results.items()})


if __name__ == "__main__":
    main()
