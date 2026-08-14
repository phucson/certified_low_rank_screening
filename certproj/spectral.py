"""Spectral risk measures on empirical laws.

A spectral (distortion) risk measure is represented by its concave distortion
`g` of the survival function.  On a law with `N` equally weighted atoms,

    rho_sigma(X) = sum_i w_i X_(i),      X_(1) <= ... <= X_(N),
    w_i = g((N-i+1)/N) - g((N-i)/N),

which is what `weights` returns.  All results in the paper are stated for a
general spectral measure; the module therefore treats `g` as the primitive and
derives everything else from it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "weights", "rho", "spectrum", "iota", "lp_norm", "effective_tail_support",
    "tvar", "dual_power", "proportional_hazard", "wang", "FAMILIES",
]


# --------------------------------------------------------------------------
# distortion functions g : [0,1] -> [0,1], concave, g(0)=0, g(1)=1
# --------------------------------------------------------------------------

def tvar(alpha: float):
    """Tail value-at-risk at level `alpha`.  Spectrum vanishes on [0, alpha)."""
    return lambda s: np.minimum(s / (1.0 - alpha), 1.0)


def dual_power(nu: float):
    """Dual-power transform, g(s) = 1 - (1-s)^nu.  Spectrum nu u^(nu-1)."""
    return lambda s: 1.0 - (1.0 - s) ** nu


def proportional_hazard(gamma: float):
    """Proportional-hazard transform, g(s) = s^(1/gamma).  iota = infinity."""
    return lambda s: s ** (1.0 / gamma)


def wang(lam: float):
    """Wang transform, g(s) = Phi(Phi^{-1}(s) + lambda).  iota = infinity."""
    return lambda s: norm.cdf(norm.ppf(np.clip(s, 1e-15, 1 - 1e-15)) + lam)


FAMILIES = {
    "TVaR(0.95)": tvar(0.95),
    "TVaR(0.99)": tvar(0.99),
    "dual-power 2": dual_power(2.0),
    "dual-power 6": dual_power(6.0),
    "PH 1.5": proportional_hazard(1.5),
    "PH 3": proportional_hazard(3.0),
    "Wang 0.5": wang(0.5),
}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def weights(n: int, g) -> np.ndarray:
    """Order-statistic weights w_1..w_n (ascending order).  Sum to one."""
    k = np.arange(1, n + 1)
    return g((n - k + 1) / n) - g((n - k) / n)


def rho(x: np.ndarray, w: np.ndarray) -> float:
    """Spectral risk measure of a sample, given precomputed weights.

    Valid for signed x: the order-statistic form handles the negative part.
    """
    return float(np.dot(np.sort(x), w))


def spectrum(g, n: int) -> np.ndarray:
    """Risk spectrum phi_sigma on an equal-mass grid of `n` cells."""
    return weights(n, g) * n


def iota(g, n: int = 200_000) -> float:
    """Tail-weight functional iota(sigma) = ||phi||_inf = g'(0+).

    Returned as a finite grid estimate; infinite for PH (gamma>1) and Wang,
    where the value diverges as n grows.
    """
    return float(spectrum(g, n).max())


def lp_norm(g, p: float, n: int = 200_000, exact: bool = True) -> float:
    """||phi_sigma||_p, the W_k inflation constant with p = k/(k-1).

    Closed forms for the standard families:
        TVaR(alpha)  : (1-alpha)^(-1/k)
        dual-power nu: nu * ((k-1)/(k*nu-1))^((k-1)/k)
        PH gamma     : gamma^-1 * (gamma(k-1)/(k-gamma))^((k-1)/k),  k > gamma
        Wang lambda  : exp(lambda^2 / (2(k-1)))

    Substituting s = 1-u gives ||phi||_p^p = int_0^1 g'(s)^p ds.  For spectra
    unbounded at the upper tail (PH with gamma>1, Wang) the integrand is singular
    at s=0, and an equal-mass grid converges far too slowly to be useful: at
    PH(3) with k=4 a grid of 2*10^7 cells still returns 1.63 against a true
    1.7321.  With `exact=True` (the default) the integral is therefore evaluated
    by adaptive quadrature with the singularity declared, which is accurate to
    ~1e-8 on all four families.  Pass `exact=False` to fall back to the grid.

    Returns +inf when the integral diverges, i.e. when phi_sigma is not in L^p;
    for PH this happens exactly when k <= gamma.
    """
    if not exact:
        phi = spectrum(g, n)
        return float((np.mean(phi ** p)) ** (1.0 / p))

    import warnings

    from scipy.integrate import quad

    def gprime(s):
        # Step scaled to the evaluation point, so the singularity at s -> 0 is
        # resolved rather than clipped by a fixed step.
        s = min(max(s, 1e-14), 1.0 - 1e-12)
        h = min(1e-7, 0.25 * s, 0.25 * (1.0 - s))
        return (g(s + h) - g(s - h)) / (2.0 * h)

    def tail(eps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v, _ = quad(lambda s: gprime(s) ** p, eps, 1.0, limit=400)
        return v

    # Divergence test. The integrand is singular at s=0 for spectra with
    # unbounded tail weight; quadrature on [0,1] returns a finite number even
    # when the integral diverges (PH at the boundary k = gamma diverges only
    # logarithmically). Truncating at successively smaller eps settles to a
    # limit when the integral converges and grows without bound when it does
    # not, which distinguishes the two reliably.
    try:
        v1, v2, v3 = tail(1e-6), tail(1e-9), tail(1e-12)
    except Exception:
        return float("inf")
    if not all(np.isfinite(x) and x > 0 for x in (v1, v2, v3)):
        return float("inf")
    if (v3 - v2) > 0.02 * max(v2, 1e-12) and (v2 - v1) > 0.02 * max(v1, 1e-12):
        return float("inf")            # still growing: not in L^p
    return float(v3 ** (1.0 / p))


def effective_tail_support(g, n: int = 200_000, eta: float = 0.99) -> float:
    """s_eta(sigma): smallest upper-tail fraction carrying weight >= eta.

    Localisation, and hence the banded fallback, is inexpensive only when this
    is small; it degenerates for full-support distortions.
    """
    w = weights(n, g)[::-1]          # descending from the largest
    return float((np.searchsorted(np.cumsum(w) / w.sum(), eta) + 1) / n)
