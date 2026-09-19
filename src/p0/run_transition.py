#!/usr/bin/env python
"""P0 E1-E4 (protocol Sec. 9): TRACE-CIR transition scoring on CIRCO and/or
CIRR validation, on the same feature cache and compiled queries as E0.

E1 = absolute transition matching (H1 vs E0); E2 = E1 + source-grounded source state (Sec. 7, 9.2).

Example:
    python -m src.p0.run_transition --variant e1 \
        --dataset circo --split val --data-root /workspace/data/CIRCO \
        --feature-cache-dir /workspace/features/circo \
        --compiled-queries-path /workspace/outputs/circo_compiled_q36.jsonl \
        --per-query-out /workspace/outputs/e1_circo_perquery.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from ..data.datasets import CIRCODataset, CIRRDataset
from ..eval import (
    average_precision_per_query,
    mean_average_precision_at_k,
    recall_at_k,
    recall_subset_at_k,
)
from ..models.openclip_utils import encode_text, load_openclip
from ..seed import set_seed
from .compiler import TransitionSpec, load_compiled_queries
from .feature_cache import FeatureCache
from .run_e0 import _query_id
from .scoring_transition import TransitionQuery, e1_score_batched, e2_score_batched, ground_query

VARIANTS = {"e1": e1_score_batched, "e2": e2_score_batched}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--feature-cache-dir", type=str, required=True)
    parser.add_argument("--compiled-queries-path", type=str, required=True)
    parser.add_argument("--openclip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip-pretrained", type=str, default="laion2b_s32b_b82k")
    parser.add_argument("--lambda-edit", type=float, default=0.5, help="Sec. 10 default 0.5; grid 0.25/0.5/1.0.")
    parser.add_argument("--tau-m", type=float, default=0.02, help="Sec. 8 default 0.02.")
    parser.add_argument("--tau-g", type=float, default=0.02, help="Sec. 7 grounding temperature; grid 0.01/0.02/0.04.")
    parser.add_argument("--normalize-prototype", action="store_true",
                        help="L2-normalise the grounded prototypes z- (not asked for by the protocol).")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--keep-reference", action="store_true")
    parser.add_argument("--per-query-out", type=str, default=None,
                        help="CIRCO: save per-query AP@5/10/25/50 (.npz) for paired comparisons.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_transition_query(spec: TransitionSpec | None, fallback_text: str, model, tokenizer, device) -> tuple[TransitionQuery, int]:
    """Encode one compiled query. Falls back to Target-only scoring on the raw
    modification text when the spec is missing or unparseable (same rule as
    E0). Returns the query and the number of atoms dropped for a missing
    required field (Sec. 5 table)."""
    use_spec = spec is not None and spec.parse_ok
    target_text = spec.target if use_spec else fallback_text
    target = encode_text(model, tokenizer, [target_text], device)[0]
    if not use_spec:
        return TransitionQuery(target=target), 0

    dropped = 0
    add, remove, preserve, rep_plus, rep_minus = [], [], [], [], []
    for atom in spec.atoms:
        op, src, tgt = atom.operation, atom.source_state, atom.target_state
        if op == "ADD" and tgt:
            add.append(tgt)
        elif op == "REMOVE" and src:
            remove.append(src)
        elif op == "PRESERVE" and src:
            preserve.append(src)
        elif op == "REPLACE" and src and tgt:
            rep_minus.append(src)
            rep_plus.append(tgt)
        else:
            dropped += 1

    def enc(texts):
        return encode_text(model, tokenizer, texts, device) if texts else None

    return TransitionQuery(
        target=target,
        add_plus=enc(add),
        remove_minus=enc(remove),
        preserve_minus=enc(preserve),
        replace_plus=enc(rep_plus),
        replace_minus=enc(rep_minus),
    ), dropped


def compute_similarity(score_fn, queries, cache, device, chunk_size, **score_kwargs) -> torch.Tensor:
    """(Q, N) scores; candidate chunk outer / query inner so the disk-backed
    local cache is read once (same loop structure as run_e0)."""
    similarity = torch.empty(len(queries), len(cache), device=device)
    for start in tqdm(range(0, len(cache), chunk_size), desc="scoring (candidate chunks)"):
        end = min(start + chunk_size, len(cache))
        cand_global = torch.from_numpy(np.asarray(cache.global_features[start:end])).float().to(device)
        cand_local = torch.from_numpy(np.asarray(cache.local_features[start:end])).float().to(device)
        for qi, query in enumerate(queries):
            similarity[qi, start:end] = score_fn(query, cand_global, cand_local, **score_kwargs)
    return similarity.cpu()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model, preprocess, tokenizer = load_openclip(args.openclip_model_name, args.openclip_pretrained, device)
    cache = FeatureCache(args.feature_cache_dir)
    compiled = load_compiled_queries(args.compiled_queries_path)

    cls = CIRCODataset if args.dataset == "circo" else CIRRDataset
    query_ds = cls(args.data_root, args.split, "relative", preprocess)

    queries: list[TransitionQuery] = []
    reference_rows: list[int] = []
    target_indices: list[int] = []
    gt_lists: list[list[int]] = []
    member_indices: list[list[int]] = []
    dropped_total = fallback = 0

    for item in tqdm(query_ds, desc="Encoding queries"):
        spec = compiled.get(_query_id(args.dataset, item))
        if spec is None or not spec.parse_ok:
            fallback += 1
        query, dropped = build_transition_query(spec, item["relative_caption"], model, tokenizer, device)
        dropped_total += dropped
        reference_id = item["reference_img_id"] if args.dataset == "circo" else item["reference_name"]
        if args.variant != "e1":
            query.reference_local = cache.local_vectors(reference_id).float().to(device)
            ground_query(query, args.tau_g, args.normalize_prototype)
        queries.append(query)
        reference_rows.append(cache.row_index(reference_id))
        if args.dataset == "circo":
            gt_lists.append([cache.row_index(i) for i in item["gt_img_ids"] if i in cache._id_to_row])
        else:
            if "target_name" in item:
                target_indices.append(cache.row_index(item["target_name"]))
            member_indices.append([cache.row_index(m) for m in item["member_set"] if m in cache._id_to_row])

    atom_counts = [q.num_atoms for q in queries]
    print(f"{len(queries)} queries | {fallback} fell back to Target-only | {dropped_total} atoms dropped "
          f"(missing required field) | mean atoms/query {np.mean(atom_counts):.2f} | "
          f"queries without atoms {sum(1 for n in atom_counts if n == 0)}", flush=True)

    similarity = compute_similarity(
        VARIANTS[args.variant], queries, cache, device, args.chunk_size,
        lambda_edit=args.lambda_edit, tau_m=args.tau_m,
    )
    if not args.keep_reference:
        similarity[torch.arange(len(reference_rows)), torch.tensor(reference_rows)] = float("-inf")

    tag = args.variant.upper()
    if args.dataset == "circo":
        ks = [5, 10, 25, 50]
        print({f"{tag} mAP@{k}": v for k, v in mean_average_precision_at_k(similarity, gt_lists, ks).items()})
        if args.per_query_out:
            ranking = similarity.argsort(dim=-1, descending=True)
            np.savez(args.per_query_out, **{f"ap{k}": np.array(average_precision_per_query(ranking, gt_lists, k)) for k in ks})
    else:
        print({f"{tag} R@{k}": v for k, v in recall_at_k(similarity, torch.tensor(target_indices), [1, 5, 10, 50]).items()})
        print({f"{tag} R_subset@{k}": v for k, v in
               recall_subset_at_k(similarity, torch.tensor(target_indices), member_indices, [1, 2, 3]).items()})


if __name__ == "__main__":
    main()
