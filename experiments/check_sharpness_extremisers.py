"""Sharpness constants: R*_sigma = sup{rho(e)/rho(|e|) : E[e]=0}, by extremiser family.

This maximises the ratio over the mean-zero families the proofs identify as
extremal -- symmetric two-point, asymmetric two-point, and the dilute
three-atom law -- on an equal-mass quantile grid.  It is therefore a LOWER
BOUND on R*_sigma by construction, which is what makes agreement with the
proven constants informative.  It is not a free maximisation over all mean-zero
laws.

TVaR attains 1 exactly (blind lower spectrum); PH approaches 1 (unbounded tail
weight); dual-power attains max{1/2, 1-2^(1-nu)} < 1, satisfying neither
condition.
"""
import sys
import numpy as np
sys.path.insert(0, "..")
from certproj import weights, rho, tvar, dual_power, proportional_hazard

N = 4_000_000


def ratio(qf, g):
    w = weights(N, g); e = qf(N)
    return rho(e, w) / rho(np.abs(e), w)


def two_point_symmetric(n):
    return np.concatenate([np.full(n // 2, -1.0), np.full(n - n // 2, 1.0)])


def two_point_vanishing(p):
    def f(n):
        k = max(int(p * n), 1); a = p / (1 - p)
        return np.concatenate([np.full(n - k, -a), np.full(k, 1.0)])
    return f


def dilute(pi):
    def f(n):
        k = max(int(pi * n), 1)
        return np.concatenate([np.full(k, -1.0), np.zeros(n - 2 * k),
                               np.full(k, 1.0)])
    return f


def two_point_asymmetric(alpha):
    """+a w.p. 1-alpha, -b w.p. alpha with (1-alpha)a = alpha b: the extremiser
    of the TVaR and two-step slack propositions, whose mass split sits at the
    spectral kink."""
    def f(n):
        k = max(int(round(alpha * n)), 1)
        b = (1 - alpha) / alpha
        return np.concatenate([np.full(k, -b), np.full(n - k, 1.0)])
    return f


def bounded_positive(c1, c2, alpha):
    """Two-step spectrum: phi = c1 on [0,alpha), c2 on [alpha,1].

    Normalised by c1*alpha + c2*(1-alpha) = 1, this is the family for which the
    two-step proposition gives R* in closed form.  The distortion is the
    piecewise-affine integral of phi.
    """
    def g(s):
        s = np.asarray(s, dtype=float)
        return np.where(s <= 1 - alpha, c2 * s,
                        c2 * (1 - alpha) + c1 * (s - (1 - alpha)))
    return g


def from_spectrum(phi, n_quad=200_000):
    """Distortion g built from an arbitrary nondecreasing spectrum phi.

    The spectrum weights the u-quantile, so g'(s) = phi(1-s) and

        g(s) = int_0^s phi(1-v) dv,   normalised so g(1) = 1.

    Reversing phi is what makes g concave: phi is nondecreasing, so phi(1-v) is
    nonincreasing.  Integrating phi itself would give a convex g, which is not a
    distortion and yields a meaningless ratio.
    """
    u = (np.arange(n_quad) + 0.5) / n_quad
    vals = phi(1.0 - u)
    vals = vals / vals.mean()                     # int_0^1 phi = 1
    cdf = np.concatenate([[0.0], np.cumsum(vals) / n_quad])
    grid = np.concatenate([[0.0], (np.arange(n_quad) + 1) / n_quad])

    def g(s):
        return np.interp(np.asarray(s, dtype=float), grid, cdf / cdf[-1])
    return g


def best_ratio(g, families):
    return max(ratio(qf, g) for qf in families)


if __name__ == "__main__":
    print("TVaR -- symmetric two-point, should be exactly 1")
    for al in (0.5, 0.9, 0.95, 0.99):
        print(f"  alpha={al}: {ratio(two_point_symmetric, tvar(al)):.6f}")
    print("\ndual-power -- R* = max(1/2, 1-2^(1-nu))")
    for nu in (1.25, 1.5, 2.0, 3.0, 4.0, 6.0):
        g = dual_power(nu)
        best = max([ratio(two_point_symmetric, g)]
                   + [ratio(dilute(pi), g) for pi in (1e-2, 1e-3, 1e-4)])
        print(f"  nu={nu:<5}: attained={best:.6f}  claimed={max(.5,1-2**(1-nu)):.6f}")
    print("\nPH -- vanishing mass approaches 1")
    for gam in (1.5, 3.0):
        v = [ratio(two_point_vanishing(p), proportional_hazard(gam))
             for p in (1e-2, 1e-3, 1e-4, 1e-5)]
        print(f"  gamma={gam}: " + " -> ".join(f"{x:.4f}" for x in v))

    # ------------------------------------------------------------------
    # Spectra outside both sufficiency conditions: bounded and positive on
    # the interior, where the strict-slack propositions apply.  These are the
    # second block of the supplementary sharpness table.
    # ------------------------------------------------------------------
    print("\nbounded, everywhere-positive and two-step spectra -- R* < 1")
    fam = ([two_point_symmetric]
           + [dilute(pi) for pi in (1e-2, 1e-3, 1e-4)]
           + [two_point_asymmetric(al) for al in
              (0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.5, 0.6)])

    r_tvar03 = best_ratio(tvar(0.3), fam)
    print(f"  TVaR(0.3)                     : {r_tvar03:.4f}"
          f"   exact {0.7/(2-0.9):.4f}")

    g_lin = from_spectrum(lambda u: (1 + 2 * u) / 2)
    print(f"  phi(u)=(1+2u)/2               : {best_ratio(g_lin, fam):.4f}   < 1")

    g_exp = from_spectrum(lambda u: np.exp(2 * u))
    print(f"  phi(u) propto exp(2u)         : {best_ratio(g_exp, fam):.4f}   < 1")

    # blind on [0,0.6] with a 5% floor: two-step at alpha=0.6, c1=0.05,
    # c2 fixed by c1*alpha + c2*(1-alpha) = 1.
    al, c1 = 0.6, 0.05
    c2 = (1 - c1 * al) / (1 - al)
    exact = al * (c2 - c1) / (1 + c2 * (2 * al - 1))
    print(f"  blind [0,0.6] + 5% floor      : "
          f"{best_ratio(bounded_positive(c1, c2, al), fam):.4f}"
          f"   exact {exact:.4f}")

    print("\nTVaR sweep below the phase boundary -- R* = (1-alpha)/(2-3alpha)")
    for al in (0.10, 0.20, 0.30, 0.40, 0.45, 0.49, 0.50):
        got = best_ratio(tvar(al), fam)
        print(f"  alpha={al:<5}: attained={got:.4f}"
              f"  exact={(1-al)/(2-3*al):.4f}")
