"""Certified rank-r projection of max-affine losses.

Implements the objects of Sections 2-5 of the paper:

    P_j(F)      = max_t (a_jt + b_jt' F)                       max-affine loss
    F           = mu + U z + e,   z = U'(F-mu),  e = Pi_U^perp (F-mu)
    Ptil_j(z)   = max_t (atil_jt + btil_jt' z)                 surrogate loss
    gamma_j(U)  = max_t ||Pi_U^perp b_jt||                     orthogonal slope radius

    |rho(P_j) - rho(Ptil_j)| <= gamma_j(U) * rho(||e||)        certified bracket
    |R^rho(P_j) - R^rho(Ptil_j)| <= gamma_j(U)(rho(||e||) + rad*iota)
                                                               robust bracket

    band B_j = {i : Ptil_ji + delta_ji >= tau_j}               certified band
    with tau_j the m-th largest of Ptil_ji - delta_ji;
    B_j provably contains the true tail set                    band correctness
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import rho, weights

__all__ = ["Candidate", "Projection", "bracket", "robust_bracket",
           "tight_envelope", "certified_band", "screen"]


@dataclass(frozen=True)
class Candidate:
    """A max-affine exposure: intercepts `a` (K,) and slopes `b` (K, d)."""
    a: np.ndarray
    b: np.ndarray

    @property
    def K(self) -> int:
        return len(self.a)

    def evaluate(self, F: np.ndarray) -> np.ndarray:
        """P_j at every scenario.  Cost O(N K d) -- the baseline being avoided."""
        return (self.a[None, :] + F @ self.b.T).max(axis=1)

    def lipschitz(self) -> float:
        return float(np.linalg.norm(self.b, axis=1).max())


class Projection:
    """Rank-r projection data shared across the whole candidate population.

    Everything here is computed once per catalogue and subspace; per-candidate
    work is O(K d) and touches no scenario data (the cost proposition).
    """

    def __init__(self, F: np.ndarray, r: int, U: np.ndarray | None = None):
        self.F = F
        self.mu = F.mean(axis=0)
        Fc = F - self.mu
        if U is None:                      # default rule: PCA of Sigma_F (U_F)
            Sig = np.cov(Fc, rowvar=False)
            evec = np.linalg.eigh(Sig)[1][:, ::-1]
            U = evec[:, :r]
        self.U = U
        self.r = U.shape[1]
        self.z = Fc @ U                    # retained coordinates (N, r)
        self.E = Fc - self.z @ U.T         # residual (N, d)
        self.enorm = np.linalg.norm(self.E, axis=1)
        self.N = len(F)

    def constant(self, g) -> float:
        """C^e_sigma(U) = rho_sigma(||e||): candidate-independent, computed once."""
        return rho(self.enorm, weights(self.N, g))

    def gamma(self, c: Candidate) -> float:
        """Orthogonal slope radius gamma_j(U).  Cost O(K d), no catalogue access."""
        bt = c.b @ self.U
        return float(np.linalg.norm(c.b - bt @ self.U.T, axis=1).max())

    def surrogate(self, c: Candidate) -> np.ndarray:
        """Ptil_j at every scenario.  Cost O(N K r) by direct evaluation."""
        bt = c.b @ self.U
        at = c.a + c.b @ self.mu
        return (at[None, :] + self.z @ bt.T).max(axis=1)

    def surrogate_lipschitz(self, c: Candidate) -> float:
        return float(np.linalg.norm(c.b @ self.U, axis=1).max())


# --------------------------------------------------------------------------
# certificates
# --------------------------------------------------------------------------

def bracket(proj: Projection, c: Candidate, g) -> float:
    """Nominal half-width gamma_j(U) * C^e_sigma(U): the certified bracket."""
    return proj.gamma(c) * proj.constant(g)


def robust_bracket(proj: Projection, c: Candidate, g, radius: float,
                   iota_val: float) -> float:
    """Wasserstein-robust half-width gamma_j (C^e + radius * iota).

    For a W_k ball with k > 1, pass `lp_norm(g, k/(k-1))` as `iota_val`; see the
    W_k robust-inflation theorem and its corollary on finite robust capital.
    """
    return proj.gamma(c) * (proj.constant(g) + radius * iota_val)


def tight_envelope(proj: Projection, c: Candidate, g) -> tuple[float, np.ndarray]:
    """Tight-envelope certificate.

    D_j(F_i) = max_t |(Pi_U^perp b_jt)' e_i| is the first bound of the pointwise
    sandwich, before Cauchy-Schwarz.  Returns (rho_sigma(D_j), D_j).

    Cost O(N K (d-r)): comparable to exact evaluation, so this tier pays only
    when the true loss is expensive relative to its affine description, i.e.
    when K_j (d-r) < (beta_CS - beta_tight) * c_P with c_P the per-scenario cost
    of evaluating the true loss.
    """
    bt = c.b @ proj.U
    Bp = c.b - bt @ proj.U.T
    D = np.abs(proj.E @ Bp.T).max(axis=1)
    return rho(D, weights(proj.N, g)), D


def certified_band(surrogate: np.ndarray, delta: np.ndarray, m: int
                   ) -> np.ndarray:
    """Boolean mask of the certified band B_j.

    `m` is the number of order statistics the risk measure touches, i.e.
    ceil((1-alpha)N) for TVaR(alpha).  The threshold is taken from the LOWER
    envelope and membership tested against the UPPER one; both are forced
    """
    n = len(surrogate)
    lo = surrogate - delta
    tau = np.partition(lo, n - m)[n - m]
    return (surrogate + delta) >= tau


def screen(surrogate_value: float, half_width: float, threshold: float
           ) -> str:
    """Certified accept/reject label, or 'undecided': the screening rule.

    The label is exact even though the value is approximate.
    """
    if surrogate_value + half_width < threshold:
        return "below"
    if surrogate_value - half_width > threshold:
        return "above"
    return "undecided"
