"""Verification of the W_1 closed form, its invariance, and the lift lemma.

Three checks, all in quantile space where the comonotone coupling is optimal
for a one-dimensional reference law:

  (a) the robust-value programme solved as a linear programme on an equal-mass
      grid, permitting mass splitting, and compared against rho * iota(sigma);
  (b) invariance of the increment across reference laws, which the closed form
      requires since the reference enters only through the nominal term;
  (c) the pushforward lemma, by lifting a one-dimensional plan along U and
      comparing transport cost measured in R^d against cost along the retained
      direction.

The dual-power residual in (a) is discretisation of a continuous spectrum on an
equal-mass grid and decays as O(1/M); TVaR, whose spectrum is a step function
represented exactly on the grid, is exact to machine precision.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import weights, iota, tvar, dual_power, assert_version, dump


def lp_increment(g, M, radius):
    """Worst-case increment over a W_1 ball, as an explicit LP.

    Variables are the per-cell displacements d_i >= 0 of the reference quantile
    function.  Maximising sum_i w_i d_i subject to the transport budget
    (1/M) sum_i d_i <= radius is a linear programme; mass splitting is what
    makes the continuous relaxation exact.
    """
    from scipy.optimize import linprog
    w = weights(M, g)
    res = linprog(c=-w, A_ub=np.ones((1, M)) / M, b_ub=[radius],
                  bounds=[(0, None)] * M, method="highs")
    assert res.status == 0, res.message
    return float(-res.fun)


def reference_sample(kind, n, rng):
    if kind == "Gaussian":
        return rng.normal(size=n)
    if kind == "Student-t3":
        return rng.standard_t(3, size=n)
    if kind == "Pareto(2.5)":
        return rng.pareto(2.5, size=n)
    if kind == "V-shaped":
        return np.abs(rng.normal(size=n)) * rng.choice([-1.0, 1.0], size=n)
    raise ValueError(kind)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=2000)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()
    rng = np.random.default_rng(a.seed)

    # (a) LP against the closed form.
    print(f"(a) W_1 increment, LP in quantile space, M={a.M}, "
          f"radius={a.radius}")
    rows = []
    for name, g in [("TVaR(0.99)", tvar(0.99)), ("dual-power 6", dual_power(6))]:
        got = lp_increment(g, a.M, a.radius)
        want = a.radius * iota(g)
        rel = abs(got - want) / want
        rows.append(dict(check="lp", family=name, lp=got, closed_form=want,
                         rel_err=rel))
        print(f"    {name:<14} LP {got:12.6f}   rho*iota {want:12.6f}"
              f"   rel.err {rel:.2e}")

    # Decay of the dual-power residual, which the text attributes to the grid.
    print("\n    dual-power residual against grid size")
    prev = None
    for M in (500, 1000, 2000, 4000):
        g = dual_power(6)
        rel = abs(lp_increment(g, M, a.radius) - a.radius * iota(g)) \
            / (a.radius * iota(g))
        ratio = "" if prev is None else f"   halving ratio {prev / rel:.2f}"
        print(f"      M={M:<6} rel.err {rel:.3e}{ratio}")
        rows.append(dict(check="grid_decay", M=M, rel_err=rel))
        prev = rel

    # (b) Invariance across reference laws.
    print("\n(b) invariance of the increment across reference laws")
    g = tvar(0.99)
    base = None
    for kind in ("Gaussian", "Student-t3", "Pareto(2.5)", "V-shaped"):
        x = np.sort(reference_sample(kind, a.M, rng))
        w = weights(a.M, g)
        nominal = float(np.dot(x, w))
        # comonotone displacement along the Holder/extreme direction
        d = np.zeros(a.M); d[int(np.argmax(w))] = a.radius * a.M
        robust = float(np.dot(np.sort(x + d), w))
        inc = robust - nominal
        base = inc if base is None else base
        rows.append(dict(check="invariance", reference=kind, increment=inc))
        print(f"    {kind:<14} increment {inc:.10f}"
              f"   deviation {abs(inc - base):.2e}")

    # (c) Lift of a one-dimensional plan along U.
    print("\n(c) pushforward lemma: cost in R^d vs cost along the retained axis")
    d_dim, r_dim, n = 8, 2, 5000
    U = np.linalg.qr(rng.normal(size=(d_dim, r_dim)))[0]
    X = rng.normal(size=(n, d_dim))
    step = rng.normal(size=(n, r_dim))                # plan on retained coords
    Y = X + step @ U.T                                # lifted to R^d
    cost_d = float(np.mean(np.linalg.norm(Y - X, axis=1)))
    cost_r = float(np.mean(np.linalg.norm(step, axis=1)))
    rel = abs(cost_d - cost_r) / cost_r
    sig = -int(np.floor(np.log10(rel))) if rel > 0 else 16
    print(f"    R^d cost {cost_d:.12f}   retained cost {cost_r:.12f}")
    print(f"    agree to {sig} significant figures")
    rows.append(dict(check="lift", cost_Rd=cost_d, cost_retained=cost_r,
                     rel_err=rel, significant_figures=sig))

    dump("../results/w1_verification.json", rows,
         dict(M=a.M, radius=a.radius, seed=a.seed, d=d_dim, r=r_dim,
              n_lift=n,
              references=["Gaussian", "Student-t3", "Pareto(2.5)", "V-shaped"]))
