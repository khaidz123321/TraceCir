"""TRACE-CIR transition scoring for P0 (protocol Sec. 8-10).

Implemented so far: E1 (Sec. 9.1, absolute transition matching). E2-E4 (source
grounding, local delta, coupled REPLACE) build on the same primitives and the
same query record and are added here as they are implemented.

Everything E1-E4 share follows the protocol: one local matcher

    M(u, I) = tau_m * log sum_k exp(u^T v_k(I) / tau_m),   tau_m = 0.02

over the M = 64 local vectors of a candidate, one holistic target score
Phi_T(I_c) = t_T^T v0(I_c), and the final score

    S(I_c) = Phi_T(I_c) + lambda_edit * (1/J) * sum_j Phi_j(I_c),   lambda_edit = 0.5

with J the number of usable atoms (S = Phi_T when a query has none). There are
no per-operation weights (Sec. 10).

Unlike TAPR's E0 there is no reference-continuity term for Preserve and no
factor renormalisation: E1 changes only the transition representation/scoring,
exactly as Sec. 9.1 defines it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransitionQuery:
    """One query's encoded probes (all unit-norm, on the scoring device).

    add_plus:    (n, d) target_state of each ADD atom
    remove_minus:(n, d) source_state of each REMOVE atom
    preserve_minus: (n, d) source_state of each PRESERVE atom
    replace_plus / replace_minus: (n, d) target_state / source_state of each
        REPLACE atom, row-aligned.
    Any of them is None when the query has no such atom.
    """

    target: torch.Tensor
    add_plus: torch.Tensor | None = None
    remove_minus: torch.Tensor | None = None
    preserve_minus: torch.Tensor | None = None
    replace_plus: torch.Tensor | None = None
    replace_minus: torch.Tensor | None = None

    @property
    def num_atoms(self) -> int:
        return sum(
            t.shape[0]
            for t in (self.add_plus, self.remove_minus, self.preserve_minus, self.replace_plus)
            if t is not None
        )


def local_match_batched(probes: torch.Tensor, candidate_local: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """M(u, I) of Sec. 8 for every probe and every candidate.

    Args:
        probes: (n, d) unit text vectors.
        candidate_local: (N, M, d) unit local vectors of N candidates.

    Returns:
        (n, N) tensor, element [i, k] = M(probes[i], candidate k).
    """
    sims = torch.einsum("nd,kmd->nkm", probes, candidate_local)
    return tau * torch.logsumexp(sims / tau, dim=-1)


def e1_score_batched(
    query: TransitionQuery,
    candidate_global: torch.Tensor,
    candidate_local: torch.Tensor,
    lambda_edit: float = 0.5,
    tau_m: float = 0.02,
) -> torch.Tensor:
    """E1 (Sec. 9.1): absolute transition matching, scored for N candidates.

        ADD:      M(u+, I_c)
        REMOVE:  -M(u-, I_c)
        PRESERVE: M(u-, I_c)
        REPLACE:  M(u+, I_c) - M(u-, I_c)

    Args:
        candidate_global: (N, d); candidate_local: (N, M, d).

    Returns:
        (N,) tensor of S(I_c).
    """
    score = candidate_global @ query.target
    num_atoms = query.num_atoms
    if num_atoms == 0:
        return score

    edit = torch.zeros_like(score)
    if query.add_plus is not None:
        edit = edit + local_match_batched(query.add_plus, candidate_local, tau_m).sum(dim=0)
    if query.remove_minus is not None:
        edit = edit - local_match_batched(query.remove_minus, candidate_local, tau_m).sum(dim=0)
    if query.preserve_minus is not None:
        edit = edit + local_match_batched(query.preserve_minus, candidate_local, tau_m).sum(dim=0)
    if query.replace_plus is not None:
        gain = local_match_batched(query.replace_plus, candidate_local, tau_m)
        lose = local_match_batched(query.replace_minus, candidate_local, tau_m)
        edit = edit + (gain - lose).sum(dim=0)
    return score + lambda_edit * edit / num_atoms
