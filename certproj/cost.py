"""Operation-count cost model behind the speedup columns of the main table.

The fallback-cost proposition gives, for an undecided candidate, an exact cost
of O(N c_sur + beta_j N K_j d) against Theta(N K_j d) for full revaluation, so
the per-candidate saving factor is

    K_j d / (c_sur + beta_j K_j d).

Screening resolves a fraction f of the population outright, and the certified
branch is O(K_j d) per candidate with no catalogue access, hence negligible
against the N-proportional terms.  The total-cost speedup reported in the main
table is therefore

    speedup = K d / [ (1 - f) (c_sur + beta K d) ].

Two surrogate-evaluation regimes are distinguished:

    direct           c_sur = K r          take the max over K affine pieces
    upper envelope   c_sur = log2 K       point location in the precomputed
                                          envelope of the K hyperplanes

These are operation counts, not timings.  The wall-clock section of the paper
reports that the direct column does not survive conversion for inexpensive
losses; `speedups` therefore returns both and the caller is expected to carry
that caveat.
"""
from __future__ import annotations

import math

__all__ = ["surrogate_cost", "saving_factor", "speedup", "speedups"]


def surrogate_cost(K: int, r: int, regime: str) -> float:
    """Amortised per-scenario cost c_sur of evaluating the surrogate."""
    if regime == "direct":
        return float(K * r)
    if regime == "envelope":
        return math.log2(K)
    raise ValueError(f"unknown regime {regime!r}; use 'direct' or 'envelope'")


def saving_factor(K: int, d: int, r: int, beta: float, regime: str) -> float:
    """Per-candidate saving on the exact branch, for an undecided candidate."""
    return K * d / (surrogate_cost(K, r, regime) + beta * K * d)


def speedup(K: int, d: int, r: int, f: float, beta: float, regime: str
            ) -> float:
    """Total-cost speedup over the Theta(N K d) baseline across a population.

    `f` is the certified fraction and `beta` the band fraction; both are
    measured by the rank-ladder experiment.  Certified candidates cost O(K d),
    which is independent of N and dropped here.
    """
    if not 0.0 <= f < 1.0:
        raise ValueError("certified fraction must lie in [0, 1)")
    return saving_factor(K, d, r, beta, regime) / (1.0 - f)


def speedups(K: int, d: int, r: int, f: float, beta: float
             ) -> tuple[float, float]:
    """(direct, envelope) total-cost speedups for one table row."""
    return (speedup(K, d, r, f, beta, "direct"),
            speedup(K, d, r, f, beta, "envelope"))
