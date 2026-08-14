"""The gamma-diagnostic as a pre-commitment rule (Table S.5).

No aggregate measure of catalogue concentration predicts the attainable
certified fraction.  What governs it is the population distribution of
gamma_j(U), which costs O(K d) per candidate and needs no catalogue access.
Combined with a small pilot of surrogate values, the predicted fraction is

    fhat(r) = mean over pilot of  P_j( gamma_j(U_r) < |vtil - Q| / C^e(U_r) ),

which is compared here against the realised f(r) at each rank.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, screen,
                      assert_version, dump)
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books

ALPHA = 0.99
PILOT = 30


def run(F, cands, r, g, rng):
    proj = Projection(F, r)
    w = weights(len(F), g)
    Ce = proj.constant(g)
    gam = np.array([proj.gamma(c) for c in cands])
    surr = np.array([rho(proj.surrogate(c), w) for c in cands])
    true = np.array([rho(c.evaluate(F), w) for c in cands])
    half = gam * Ce
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"

    Q = float(np.median(true))
    realised = float(np.mean([screen(s, h, Q) != "undecided"
                              for s, h in zip(surr, half)]))
    # Pilot of surrogate values only; the predictor uses no true values and no
    # catalogue access beyond the shared constant.
    pilot = rng.choice(len(cands), size=PILOT, replace=False)
    predicted = float(np.mean([(gam < abs(surr[i] - Q) / Ce).mean()
                               for i in pilot]))
    return dict(r=r, realised=realised, predicted=predicted,
                error=predicted - realised,
                gamma_max_over_median=float(gam.max() / np.median(gam)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    F = synthetic_catalogue(a.N, seed=a.seed)
    cands = bond_books(a.B, m=4, K=5, seed=a.seed)
    g = FAMILIES["TVaR(0.99)"]
    rng = np.random.default_rng(a.seed)

    rows = [run(F, cands, r, g, rng) for r in (1, 2, 3, 4)]
    print(f"synthetic population, B={a.B}, N={a.N}, pilot={PILOT}, TVaR(0.99)")
    print(f"{'r':>3}{'realised f':>13}{'predicted f':>14}{'error':>9}")
    for o in rows:
        print(f"{o['r']:>3}{o['realised']:>13.3f}{o['predicted']:>14.3f}"
              f"{o['error']:>+9.3f}")
    print(f"\nmax error {max(abs(o['error']) for o in rows):.3f}")
    dump("../results/tableS5_gamma_diagnostic.json", rows,
         dict(population="synthetic (S.4)", N=a.N, B=a.B, seed=a.seed,
              pilot=PILOT, sigma="TVaR(0.99)", threshold="population median",
              ranks=[1, 2, 3, 4]))
