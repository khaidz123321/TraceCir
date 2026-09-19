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

import numpy as np
import torch
from tqdm import tqdm

from ..data.datasets import CIRCODataset, CIRRDataset
from ..eval import average_precision_per_query, mean_average_precision_at_k, recall_at_k, recall_subset_at_k
from ..models.openclip_utils import encode_text, load_openclip
from ..seed import set_seed
from .compiler import load_compiled_queries
from .scoring import E0Weights, e0_score_batched
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
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="Candidate images scored per GPU step (lower it if you run out of GPU memory).")
    parser.add_argument("--keep-reference", action="store_true",
                        help="Do not remove the reference image from the ranking (default: remove it).")
    parser.add_argument("--per-query-out", type=str, default=None,
                        help="CIRCO: save per-query AP@5/10/25/50 (.npz) for paired comparisons.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _query_id(dataset: str, item: dict) -> str:
    if dataset == "circo":
        return str(item["reference_img_id"])
    return item["reference_name"] + "|" + str(item["pair_id"])


def compute_similarity(
    queries: list[dict],
    cache: FeatureCache,
    weights: E0Weights,
    device: torch.device,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Score every query against every cached candidate image.

    The loop is candidate-chunk OUTER, query INNER: each chunk of the
    (disk-backed, ~11GB) local-feature cache is read from disk once and
    scored against all queries, instead of re-reading the whole cache once
    per query.

    Args:
        queries: per query, a dict with tensors on `device`: "target" (d,),
            "add"/"preserve"/"remove" ((n, d) or None), "reference_local" (M, d).

    Returns:
        (Q, N) float tensor on CPU of retrieval scores.
    """
    n_candidates = len(cache)
    similarity = torch.empty(len(queries), n_candidates, device=device)

    for start in tqdm(range(0, n_candidates, chunk_size), desc="E0 scoring (candidate chunks)"):
        end = min(start + chunk_size, n_candidates)
        cand_global = torch.from_numpy(np.asarray(cache.global_features[start:end])).float().to(device)
        cand_local = torch.from_numpy(np.asarray(cache.local_features[start:end])).float().to(device)
        for qi, q in enumerate(queries):
            similarity[qi, start:end] = e0_score_batched(
                q["target"], q["add"], q["preserve"], q["remove"],
                q["reference_local"], cand_global, cand_local, weights,
            )
    return similarity.cpu()


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

    queries: list[dict] = []
    reference_rows: list[int] = []
    target_indices: list[int] = []
    gt_lists: list[list[int]] = []
    member_indices: list[list[int]] = []

    for item in tqdm(query_ds, desc="Encoding queries"):
        spec = compiled.get(_query_id(args.dataset, item))
        use_spec = spec is not None and spec.parse_ok

        def probes(*operation_fields: tuple[str, str]) -> torch.Tensor | None:
            if not use_spec:
                return None
            phrases = spec.phrases(*operation_fields)
            return encode_text(model, tokenizer, phrases, device) if phrases else None

        target_text = spec.target if use_spec else item["relative_caption"]
        reference_id = item["reference_img_id"] if args.dataset == "circo" else item["reference_name"]

        queries.append({
            "target": encode_text(model, tokenizer, [target_text], device)[0],
            # E0 is TAPR's factor-wise baseline, which has no REPLACE concept: a
            # replacement is scored as the independent Add (its target_state)
            # + Remove (its source_state) that H3 argues against. Dropping
            # REPLACE atoms here would silently discard part of the edit.
            "add": probes(("ADD", "target_state"), ("REPLACE", "target_state")),
            "preserve": probes(("PRESERVE", "source_state")),
            "remove": probes(("REMOVE", "source_state"), ("REPLACE", "source_state")),
            "reference_local": cache.local_vectors(reference_id).to(device),
        })
        reference_rows.append(cache.row_index(reference_id))

        if args.dataset == "circo":
            gt_lists.append([cache.row_index(i) for i in item["gt_img_ids"] if i in cache._id_to_row])
        else:
            if "target_name" in item:
                target_indices.append(cache.row_index(item["target_name"]))
            member_indices.append([cache.row_index(m) for m in item["member_set"] if m in cache._id_to_row])

    similarity = compute_similarity(queries, cache, weights, device, args.chunk_size)

    # The reference image is trivially the best "Preserve" match for itself
    # (identical local features) and is never a valid target, so it is
    # removed from the ranking, as is standard for CIR evaluation.
    if not args.keep_reference:
        similarity[torch.arange(len(reference_rows)), torch.tensor(reference_rows)] = float("-inf")

    if args.dataset == "circo":
        results = mean_average_precision_at_k(similarity, gt_lists, [5, 10, 25, 50])
        print({f"E0 mAP@{k}": v for k, v in results.items()})
        if args.per_query_out:
            import numpy as np

            ranking = similarity.argsort(dim=-1, descending=True)
            np.savez(args.per_query_out, **{
                f"ap{k}": np.array(average_precision_per_query(ranking, gt_lists, k)) for k in (5, 10, 25, 50)
            })
    else:
        results = recall_at_k(similarity, torch.tensor(target_indices), [1, 5, 10, 50])
        print({f"E0 R@{k}": v for k, v in results.items()})
        subset_results = recall_subset_at_k(similarity, torch.tensor(target_indices), member_indices, [1, 2, 3])
        print({f"E0 R_subset@{k}": v for k, v in subset_results.items()})


if __name__ == "__main__":
    main()
