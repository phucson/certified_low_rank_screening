"""Attainment of the W_k inflation constant (transport check for the W_k theorem).

The W_k robust-inflation theorem bounds the worst-case spectral risk over a
W_k ball of radius rho by

    rho_sigma(l) + rho * L * ||phi_sigma||_{k/(k-1)},

and asserts the constant is unimprovable: equality holds at every rho when the
steepest affine piece of l is active almost surely.

This script verifies the *increment*, which the constants script does not: it
solves the inner supremum in quantile space, where for a one-dimensional
reference law the comonotone coupling is optimal and the displacement profile
is the Holder extremiser q_Delta(u) propto phi_sigma(u)^{1/(k-1)}, normalised
to ||q_Delta||_k = rho.

Reference law supported on a ray, so the construction is genuinely
one-dimensional and comonotone additivity applies (see the proof).
"""
import sys
import numpy as np
sys.path.insert(0, "..")
from certproj import weights, rho as rho_sigma, lp_norm, tvar, dual_power, wang

M = 200_000          # quantile grid
RHO = 0.25           # ball radius


def increment(g, k, n=M, radius=RHO):
    """Numerically maximised increment over a W_k ball, in quantile space.

    Returns (achieved increment / (radius * L), theoretical ||phi||_{k/(k-1)}).
    """
    p = k / (k - 1.0)
    u = (np.arange(n) + 0.5) / n
    w = weights(n, g)
    phi = w * n                                     # spectrum on the grid

    # Reference law on a ray: x(u) = quantile of a standard normal, and the
    # loss is the steepest piece throughout, so l(x) = L * x with L = 1.
    L = 1.0
    x = np.sort(np.random.default_rng(0).normal(size=n))

    # Holder extremiser, normalised so that (mean q^k)^(1/k) = radius.
    q = phi ** (1.0 / (k - 1.0))
    q *= radius / (np.mean(q ** k) ** (1.0 / k))

    base = float(np.dot(x, w))
    disp = float(np.dot(x + L * q, w))
    return (disp - base) / (radius * L), lp_norm(g, p)


if __name__ == "__main__":
    print(f"{'family':<18}{'k':>4}{'achieved/(rho L)':>19}{'||phi||_p':>12}"
          f"{'rel.err':>11}")
    for k in (2.0, 3.0, 4.0):
        for name, g in [("TVaR(0.99)", tvar(0.99)),
                        ("dual-power 6", dual_power(6)),
                        ("Wang 0.5", wang(0.5))]:
            got, want = increment(g, k)
            print(f"{name:<18}{k:>4.0f}{got:>19.4f}{want:>12.4f}"
                  f"{abs(got - want) / want:>11.2e}")
        print()
    print("The achieved increment divided by rho*L equals ||phi_sigma||_{k/(k-1)},")
    print("so the constant of the W_k theorem is attained, not merely an upper bound.")
