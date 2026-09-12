"""Retrieval metrics: Recall@K (Sec. 5) and mAP@K (Eq. 6, used for CIRCO)."""
from __future__ import annotations

import torch


def recall_at_k(similarity: torch.Tensor, target_indices: torch.Tensor, k_values: list[int]) -> dict[int, float]:
    """
    Args:
        similarity: (N_queries, N_index) similarity matrix (higher = more similar).
        target_indices: (N_queries,) index into the index set of the correct target.
        k_values: list of K to report Recall@K for.
    """
    ranking = similarity.argsort(dim=-1, descending=True)  # (N_queries, N_index)
    results: dict[int, float] = {}
    for k in k_values:
        topk = ranking[:, :k]
        hits = (topk == target_indices.unsqueeze(1)).any(dim=1)
        results[k] = hits.float().mean().item() * 100
    return results


def mean_average_precision_at_k(
    similarity: torch.Tensor, ground_truth: list[list[int]], k_values: list[int]
) -> dict[int, float]:
    """mAP@K as defined in Eq. (6), for datasets with multiple ground truths
    per query (CIRCO).

    Args:
        similarity: (N_queries, N_index) similarity matrix.
        ground_truth: for each query, the list of index-set positions that
            are valid (relevant) targets.
        k_values: list of K to report mAP@K for.
    """
    ranking = similarity.argsort(dim=-1, descending=True).cpu()
    results: dict[int, float] = {}
    for k in k_values:
        aps = []
        for query_idx, gt in enumerate(ground_truth):
            gt_set = set(gt)
            num_gt = max(len(gt_set), 1)
            topk = ranking[query_idx, :k].tolist()
            relevant_hits = 0
            precision_sum = 0.0
            for rank, idx in enumerate(topk, start=1):
                if idx in gt_set:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank
            aps.append(precision_sum / min(k, num_gt))
        results[k] = sum(aps) / len(aps) * 100
    return results
