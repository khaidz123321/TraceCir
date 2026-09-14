"""E0 scoring: TAPR-style factor-wise matching (TAPR paper Eq. 4-10), used
as the "matched control" baseline for P0 (Table 3, row E0).

This module intentionally implements *only* TAPR's original formulas.
E1-E4 (TRACE-CIR's transition-atom variants, P0 protocol Sec. 9) build on
top of a *different* local-matching primitive (P0 protocol Sec. 8,
M(u, I) = tau_m * log sum_k exp(u^T v_k(I) / tau_m)) and belong in a
separate module once P0 moves past E0.

KNOWN GAP: the TAPR paper text available to us does not publish the exact
numeric default values for the factor weights (theta_T, theta_A, theta_P,
theta_R) used in its main experiments -- only that they exist and that a
sensitivity analysis was run around some anchor point. `E0Weights` below
defaults to equal weighting (1.0 each) as a documented placeholder; treat
this as a hyperparameter to tune/report explicitly for your own E0 run,
not as a value taken from the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def local_interaction(probes: torch.Tensor, candidates: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """TAPR Eq. (4)-(5): L_tau(U, V).

    Args:
        probes: (n, d) unit probe vectors U = {u_1, ..., u_n}.
        candidates: (M, d) unit candidate local visual vectors V = {v_1, ..., v_M}.
        tau: softmax temperature.

    Returns:
        scalar L_tau(U, V) = (1/n) * sum_i sum_j pi_ij * (u_i . v_j),
        where pi_ij = softmax_j(u_i . v_j / tau).
    """
    logits = probes @ candidates.t() / tau  # (n, M)
    weights = F.softmax(logits, dim=-1)  # pi_ij
    values = probes @ candidates.t()  # (n, M), raw dot products
    return (weights * values).sum(dim=-1).mean()


@dataclass
class E0Weights:
    """Factor weights theta_T, theta_A, theta_P, theta_R in TAPR Eq. (10).

    See the module docstring: the paper does not publish exact defaults in
    the text available to us, so these are a documented placeholder.
    """

    target: float = 1.0
    add: float = 1.0
    preserve: float = 1.0
    remove: float = 1.0
    tau_local: float = 0.02


def e0_score(
    target_text_vec: torch.Tensor,
    add_probes: torch.Tensor | None,
    preserve_probes: torch.Tensor | None,
    remove_probes: torch.Tensor | None,
    reference_local: torch.Tensor,
    candidate_global: torch.Tensor,
    candidate_local: torch.Tensor,
    weights: E0Weights = E0Weights(),
) -> float:
    """TAPR's factor-wise score S_FW(q, I), Eq. (6)-(10).

    Args:
        target_text_vec: (d,) unit vector t(T), the holistic target description.
        add_probes: (n_A, d) unit probe vectors U_A, or None/empty if the
            query has no Add atoms (factor deactivated, Eq. 10).
        preserve_probes: (n_P, d) unit probe vectors U_P, or None/empty.
            Preserve is never fully deactivated: it falls back to reference
            continuity alone when U_P is empty (Eq. 8).
        remove_probes: (n_R, d) unit probe vectors U_R, or None/empty.
        reference_local: (M, d) unit local visual vectors V_l(I_r) of the
            reference image (used for Preserve's continuity term, Eq. 8).
        candidate_global: (d,) unit global visual vector v0(I_c).
        candidate_local: (M, d) unit local visual vectors V_l(I_c).
        weights: E0Weights (theta_T/A/P/R and the local-interaction temperature).

    Returns:
        Scalar retrieval score S_FW(q, I_c). Higher is more relevant.
    """
    tau = weights.tau_local

    # Phi_T, Eq. (6): global target-description to candidate-global similarity.
    phi_t = float(target_text_vec @ candidate_global)

    active_terms: list[tuple[float, float]] = [(weights.target, phi_t)]

    # Phi_A, Eq. (7): reward Add evidence, only if the Add factor is active.
    if add_probes is not None and add_probes.numel() > 0:
        phi_a = float(local_interaction(add_probes, candidate_local, tau))
        active_terms.append((weights.add, phi_a))

    # Phi_P, Eq. (8): reference-aware continuity, optionally + textual support.
    continuity = float(local_interaction(reference_local, candidate_local, tau))
    if preserve_probes is not None and preserve_probes.numel() > 0:
        textual = float(local_interaction(preserve_probes, candidate_local, tau))
        phi_p = 0.5 * textual + 0.5 * continuity
    else:
        phi_p = continuity
    active_terms.append((weights.preserve, phi_p))

    # Phi_R, Eq. (9): penalize Remove evidence, only if the Remove factor is active.
    if remove_probes is not None and remove_probes.numel() > 0:
        phi_r = float(local_interaction(remove_probes, candidate_local, tau))
        active_terms.append((-weights.remove, phi_r))

    # Eq. (10): "Inactive Add and Remove factors are omitted, and active
    # weights are renormalized." We renormalize the (signed) weights of the
    # terms that are actually active so they sum to the same total mass
    # regardless of how many factors fired for this particular query.
    total_abs_weight = sum(abs(w) for w, _ in active_terms)
    score = sum((w / total_abs_weight) * phi for w, phi in active_terms)
    return score
