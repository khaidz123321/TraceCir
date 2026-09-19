"""TRACE-CIR transition scoring for P0 (protocol Sec. 8-10).

Implemented: E1 (Sec. 9.1, absolute transition matching), E2 (Sec. 7 and 9.2,
source-grounded source state), E3 (Sec. 9.3, local transition delta for REPLACE)
and E4 (Sec. 9.4, coupled REPLACE via SoftMin), all on the same primitives and
the same query record.

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
    # E2+: the reference image's (M, d) local vectors and the source-grounded
    # prototypes z- (Sec. 7) of the REMOVE / PRESERVE source phrases, filled by
    # `ground_query`.
    reference_local: torch.Tensor | None = None
    remove_z: torch.Tensor | None = None
    preserve_z: torch.Tensor | None = None
    replace_z: torch.Tensor | None = None

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


def ground_source(
    source_probes: torch.Tensor,
    reference_local: torch.Tensor,
    tau_g: float = 0.02,
    normalize: bool = False,
) -> torch.Tensor:
    """Source-grounded visual prototypes z- (Sec. 7).

        alpha_jk = softmax_k( (u-_j)^T v_k^r / tau_g ),   z-_j = sum_k alpha_jk v_k^r

    Args:
        source_probes: (n, d) unit text vectors of the source phrases.
        reference_local: (M, d) unit local vectors of the reference image.
        normalize: L2-normalise z-. The protocol does not ask for it, so the
            default keeps z- as the raw convex combination (norm <= 1).

    Returns:
        (n, d) prototypes.
    """
    alpha = torch.softmax(source_probes @ reference_local.t() / tau_g, dim=-1)
    z = alpha @ reference_local
    return torch.nn.functional.normalize(z, dim=-1) if normalize else z


def ground_query(query: TransitionQuery, tau_g: float = 0.02, normalize: bool = False) -> TransitionQuery:
    """Fill `remove_z` / `preserve_z` / `replace_z` of a query that has `reference_local`."""
    if query.reference_local is None:
        raise ValueError("ground_query needs query.reference_local")
    if query.remove_minus is not None:
        query.remove_z = ground_source(query.remove_minus, query.reference_local, tau_g, normalize)
    if query.preserve_minus is not None:
        query.preserve_z = ground_source(query.preserve_minus, query.reference_local, tau_g, normalize)
    if query.replace_minus is not None:
        query.replace_z = ground_source(query.replace_minus, query.reference_local, tau_g, normalize)
    return query


def e2_score_batched(
    query: TransitionQuery,
    candidate_global: torch.Tensor,
    candidate_local: torch.Tensor,
    lambda_edit: float = 0.5,
    tau_m: float = 0.02,
) -> torch.Tensor:
    """E2 (Sec. 9.2): E1 with the REMOVE and PRESERVE source states grounded
    in the reference image.

        ADD:      M(u+, I_c)                      (as E1)
        REMOVE:   M(z-, I_r) - M(z-, I_c)
        PRESERVE: M(z-, I_c)
        REPLACE:  M(u+, I_c) - M(u-, I_c)         (as E1: Sec. 9.2 lists only
                                                   REMOVE and PRESERVE)

    `query` must have been through `ground_query`.
    """
    score = candidate_global @ query.target
    num_atoms = query.num_atoms
    if num_atoms == 0:
        return score

    edit = torch.zeros_like(score)
    if query.add_plus is not None:
        edit = edit + local_match_batched(query.add_plus, candidate_local, tau_m).sum(dim=0)
    if query.remove_z is not None:
        at_reference = local_match_batched(query.remove_z, query.reference_local.unsqueeze(0), tau_m)  # (n, 1)
        edit = edit + (at_reference - local_match_batched(query.remove_z, candidate_local, tau_m)).sum(dim=0)
    if query.preserve_z is not None:
        edit = edit + local_match_batched(query.preserve_z, candidate_local, tau_m).sum(dim=0)
    if query.replace_plus is not None:
        gain = local_match_batched(query.replace_plus, candidate_local, tau_m)
        lose = local_match_batched(query.replace_minus, candidate_local, tau_m)
        edit = edit + (gain - lose).sum(dim=0)
    return score + lambda_edit * edit / num_atoms


def softmin(x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """SoftMin_beta(x, y) = -(1/beta) log(exp(-beta x) + exp(-beta y)) (Sec. 9.4)."""
    return -torch.logsumexp(-beta * torch.stack([x, y]), dim=0) / beta


def _replace_deltas(
    query: TransitionQuery, candidate_local: torch.Tensor, tau_m: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sec. 9.3 for the REPLACE atoms of a grounded query, each (n, N):

        delta+ = M(u+, I_c) - M(u+, I_r)     desired-state gain
        delta- = M(z-, I_r) - M(z-, I_c)     source-state disappearance
    """
    reference = query.reference_local.unsqueeze(0)
    gain = local_match_batched(query.replace_plus, candidate_local, tau_m) - local_match_batched(
        query.replace_plus, reference, tau_m
    )
    loss = local_match_batched(query.replace_z, reference, tau_m) - local_match_batched(
        query.replace_z, candidate_local, tau_m
    )
    return gain, loss


def _e2_non_replace_terms(query: TransitionQuery, candidate_local: torch.Tensor, tau_m: float) -> torch.Tensor:
    """ADD / REMOVE / PRESERVE contributions, identical in E2, E3 and E4."""
    edit = torch.zeros(candidate_local.shape[0], device=candidate_local.device)
    if query.add_plus is not None:
        edit = edit + local_match_batched(query.add_plus, candidate_local, tau_m).sum(dim=0)
    if query.remove_z is not None:
        at_reference = local_match_batched(query.remove_z, query.reference_local.unsqueeze(0), tau_m)
        edit = edit + (at_reference - local_match_batched(query.remove_z, candidate_local, tau_m)).sum(dim=0)
    if query.preserve_z is not None:
        edit = edit + local_match_batched(query.preserve_z, candidate_local, tau_m).sum(dim=0)
    return edit


def e3_score_batched(
    query: TransitionQuery,
    candidate_global: torch.Tensor,
    candidate_local: torch.Tensor,
    lambda_edit: float = 0.5,
    tau_m: float = 0.02,
) -> torch.Tensor:
    """E3 (Sec. 9.3): E2 with REPLACE scored as the local transition delta

        Phi_REPLACE^sum = delta+ + delta-.

    ADD, REMOVE and PRESERVE are as in E2 (Sec. 9.3 defines only REPLACE).
    Note that delta+ and delta- differ from absolute matching only by
    per-query constants, which never change a ranking.
    """
    score = candidate_global @ query.target
    num_atoms = query.num_atoms
    if num_atoms == 0:
        return score
    edit = _e2_non_replace_terms(query, candidate_local, tau_m)
    if query.replace_plus is not None:
        gain, loss = _replace_deltas(query, candidate_local, tau_m)
        edit = edit + (gain + loss).sum(dim=0)
    return score + lambda_edit * edit / num_atoms


def e4_score_batched(
    query: TransitionQuery,
    candidate_global: torch.Tensor,
    candidate_local: torch.Tensor,
    lambda_edit: float = 0.5,
    tau_m: float = 0.02,
    beta: float = 10.0,
) -> torch.Tensor:
    """E4 (Sec. 9.4, full P0): E3 with REPLACE coupled non-compensatorily,

        Phi_REPLACE^coupled = SoftMin_beta(delta+, delta-),

    so a candidate that satisfies only one side of a replacement is not
    rewarded. Everything else is as in E3.
    """
    score = candidate_global @ query.target
    num_atoms = query.num_atoms
    if num_atoms == 0:
        return score
    edit = _e2_non_replace_terms(query, candidate_local, tau_m)
    if query.replace_plus is not None:
        gain, loss = _replace_deltas(query, candidate_local, tau_m)
        edit = edit + softmin(gain, loss, beta).sum(dim=0)
    return score + lambda_edit * edit / num_atoms
