"""Threshold sweep on the synthetic calibration population (Table S.3).

The certified fraction depends on where the screening threshold sits relative
to the population of true values, not only on the rank.  The population median
is the adverse case, since candidates then cluster on the threshold; capital
screening typically applies upper-tail thresholds, where certification is
materially easier.  This sweeps Q across quantiles of the realised true-value
distribution at fixed rank.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, screen,
                      assert_version, dump)
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books

ALPHA = 0.99
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def sweep(F, cands, r, g):
    proj = Projection(F, r)
    w = weights(len(F), g)
    Ce = proj.constant(g)
    true = np.array([rho(c.evaluate(F), w) for c in cands])
    surr = np.array([rho(proj.surrogate(c), w) for c in cands])
    half = np.array([proj.gamma(c) * Ce for c in cands])
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"
    out = []
    for q in QUANTILES:
        Q = float(np.quantile(true, q))
        f = float(np.mean([screen(s, h, Q) != "undecided"
                           for s, h in zip(surr, half)]))
        out.append(dict(quantile=q, threshold=Q, certified=f))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("-r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    F = synthetic_catalogue(a.N, seed=a.seed)
    cands = bond_books(a.B, m=4, K=5, seed=a.seed)
    g = FAMILIES["TVaR(0.99)"]
    rows = sweep(F, cands, a.r, g)

    print(f"synthetic population, r={a.r}, B={a.B}, N={a.N}, TVaR(0.99)")
    print(f"{'Q quantile':>12}{'threshold':>14}{'certified f':>14}")
    for o in rows:
        print(f"{o['quantile']:>12.2f}{o['threshold']:>14.4f}"
              f"{o['certified']:>14.3f}")
    dump("../results/tableS3_threshold_sweep.json", rows,
         dict(population="synthetic (S.4)", N=a.N, B=a.B, r=a.r,
              seed=a.seed, sigma="TVaR(0.99)", m_holdings=4, K=5,
              quantiles=list(QUANTILES)))
