"""Retrieval metrics: Recall@K, RecallSubset@K (Sec. 5, Table 2, CIRR), and
mAP@K (Eq. 6, used for CIRCO)."""
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


def recall_subset_at_k(
    similarity: torch.Tensor,
    target_indices: torch.Tensor,
    member_indices: list[list[int]],
    k_values: list[int],
) -> dict[int, float]:
    """CIRR's RecallSubset@K (Sec. 5, Table 2): rank only the small subset
    of ~6 visually-similar images each query was built from (which contains
    both the reference and target), instead of the whole index set. This is
    what makes CIRR's "Image-only" baseline reduce to random guessing on
    Recall_subset@1 (5 candidates after excluding the reference) even though
    it can look artificially strong on the global Recall@K.

    Args:
        similarity: (N_queries, N_index) similarity matrix.
        target_indices: (N_queries,) index-set position of the correct target.
        member_indices: for each query, the index-set positions of every
            image in its subset (paper: 6 members, including the target).
        k_values: list of K to report RecallSubset@K for.
    """
    results: dict[int, float] = {}
    for k in k_values:
        hits = []
        for row, members in enumerate(member_indices):
            target = int(target_indices[row].item())
            member_sims = similarity[row, members]
            ranking = torch.argsort(member_sims, descending=True).tolist()
            ranked_members = [members[i] for i in ranking[:k]]
            hits.append(target in ranked_members)
        results[k] = (sum(hits) / len(hits)) * 100 if hits else float("nan")
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
