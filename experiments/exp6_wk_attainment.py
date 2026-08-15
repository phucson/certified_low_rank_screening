"""Attainment of the W_k inflation constant (transport check for the W_k theorem).

The W_k robust-inflation theorem bounds the worst-case spectral risk over a
W_k ball of radius rho by

    rho_sigma(l) + rho * L * ||phi_sigma||_{k/(k-1)},

and asserts the constant is unimprovable: equality holds at every rho when the
steepest affine piece of l is active almost surely.

This script verifies the *increment*, which the constants script does not.  It
solves the inner supremum in quantile space, where for a one-dimensional
reference law the comonotone coupling is optimal, and the optimal displacement
profile is the Holder extremiser q_Delta(u) propto phi_sigma(u)^{1/(k-1)},
normalised to ||q_Delta||_k = rho.

The loss is genuinely piecewise affine,

    l(x) = max(0, x) + 0.3 * max(0, x - 1),        L = lip(l) = 1.3,

with kinks at 0 and 1 and slopes 0, 1, 1.3.  The reference law is supported on
(1, infinity), so the steepest piece is active almost surely -- the hypothesis
under which the theorem asserts equality.  Because the Holder extremiser is
nondecreasing in u (phi_sigma is nondecreasing) and nonnegative, the displaced
quantile function stays inside the steepest piece and stays sorted, so the
displacement is comonotone with the reference and no re-sorting is needed.

Testing on a linear loss would make the attainment claim vacuous: the point is
that a loss with strictly flatter pieces elsewhere still attains the constant
once the steepest piece carries the mass.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import weights, lp_norm, tvar, dual_power, wang, assert_version, dump

M = 200_000          # quantile grid
RHO = 0.25           # ball radius
SLOPES = (1.0, 0.3)  # l(x) = max(0,x) + 0.3*max(0,x-1)
LIP = sum(SLOPES)    # steepest slope, attained for x >= 1


def loss(x):
    return np.maximum(0.0, x) + SLOPES[1] * np.maximum(0.0, x - 1.0)


def increment(g, k, n=M, radius=RHO, seed=0):
    """Achieved increment over a W_k ball, in quantile space.

    Returns (achieved / (radius * L), ||phi||_{k/(k-1)}, budget check).
    """
    p = k / (k - 1.0)
    w = weights(n, g)
    phi = w * n                                     # spectrum on the grid

    # phi_sigma is nondecreasing by definition, but the grid weights are
    # differences of a distortion evaluated at neighbouring points, so cells
    # where g has saturated lose all significance to cancellation and come back
    # as O(n * eps) noise about zero.  Raising that to the power 1/(k-1) < 1
    # amplifies it: a 2e-11 wobble becomes 3e-4 at k = 4, which is not small.
    # Restoring the theoretical monotonicity is a repair of the discretisation,
    # not a change to the object, and the assertion below keeps it honest.
    phi_raw = phi
    phi = np.maximum.accumulate(np.maximum(phi, 0.0))
    repair = float(np.abs(phi - phi_raw).max() / max(phi.max(), 1e-300))
    assert repair < 1e-8, f"monotonicity repair too large ({repair:.2e}); " \
                          "the spectrum grid is not merely noisy"

    # Reference supported on (1, inf): the steepest piece of l is active a.s.
    rng = np.random.default_rng(seed)
    x = np.sort(1.0 + np.abs(rng.normal(size=n)) + 1e-3)
    assert x.min() > 1.0, "reference must sit strictly inside the steepest piece"

    # Holder extremiser, normalised so that (mean q^k)^(1/k) = radius.
    q = phi ** (1.0 / (k - 1.0))
    q *= radius / (np.mean(q ** k) ** (1.0 / k))
    assert np.all(np.diff(q) >= 0.0), "extremiser must be nondecreasing in u"
    assert np.all(q >= 0), "displacement must be nonnegative"

    y = x + q
    assert np.all(np.diff(y) >= -1e-12), "displaced quantile must stay sorted"
    assert y.min() > 1.0, "displaced law must stay in the steepest piece"

    base = float(np.dot(loss(x), w))
    disp = float(np.dot(loss(y), w))
    budget = float(np.mean(q ** k) ** (1.0 / k))     # must equal radius
    return (disp - base) / (radius * LIP), lp_norm(g, p), budget, repair


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=M)
    ap.add_argument("--radius", type=float, default=RHO)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    print(f"loss l(x) = max(0,x) + {SLOPES[1]}*max(0,x-1), L = {LIP}; "
          f"reference supported on (1, inf)")
    print(f"grid M={a.M}, radius rho={a.radius}\n")
    print(f"{'family':<18}{'k':>4}{'achieved/(rho L)':>19}{'||phi||_p':>12}"
          f"{'rel.err':>11}")
    rows, worst = [], 0.0
    for k in (2.0, 3.0, 4.0):
        for name, g in [("TVaR(0.99)", tvar(0.99)),
                        ("dual-power 6", dual_power(6)),
                        ("Wang 0.5", wang(0.5))]:
            got, want, budget, repair = increment(g, k, n=a.M,
                                                  radius=a.radius, seed=a.seed)
            rel = abs(got - want) / want
            worst = max(worst, rel)
            assert abs(budget - a.radius) < 1e-9, "transport budget not tight"
            rows.append(dict(family=name, k=k, achieved_over_rhoL=got,
                             lp_norm=want, rel_err=rel,
                             monotonicity_repair=repair))
            print(f"{name:<18}{k:>4.0f}{got:>19.4f}{want:>12.4f}{rel:>11.2e}")
        print()
    print(f"worst relative error {worst:.2e}")
    print("The achieved increment divided by rho*L equals ||phi_sigma||_{k/(k-1)},")
    print("so the constant of the W_k theorem is attained, not merely an upper bound.")
    dump("../results/wk_attainment.json", rows,
         dict(M=a.M, radius=a.radius, seed=a.seed, loss="max(0,x)+0.3*max(0,x-1)",
              lipschitz=LIP, reference="1 + |N(0,1)|, steepest piece active a.s.",
              worst_rel_err=worst))
