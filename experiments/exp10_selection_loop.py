"""Certification inside a selection loop (the optimisation table).

Screening resolves a candidate only when its bracket clears a threshold fixed
in advance.  A search instead maintains an incumbent, and the threshold against
which candidates are judged improves as the search proceeds.  The certificate
applies unchanged with Q the incumbent value: candidate j is skipped whenever

    rho_sigma(Ptil_j) - h_j > incumbent,

that candidate being certifiably unable to improve on it.  Skipped candidates
are never evaluated on the catalogue.

Selection turns out to be easier than fixed-threshold screening, not harder,
because a search need only identify the minimiser whereas screening must
resolve candidates on both sides of the threshold.  The processing order
matters, so random (averaged over permutations), best-first and reverse orders
are all reported: the adversarial case exists and the gain is not guaranteed.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, screen,
                      assert_version, dump)
from certproj.data import yield_changes, block_bootstrap
from certproj.populations import bond_books

ALPHA = 0.99


def prepare(F, cands, r, g):
    proj = Projection(F, r)
    w = weights(len(F), g)
    Ce = proj.constant(g)
    surr = np.array([rho(proj.surrogate(c), w) for c in cands])
    half = np.array([proj.gamma(c) * Ce for c in cands])
    true = np.array([rho(c.evaluate(F), w) for c in cands])
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"
    return true, surr, half


def selection_loop(order, true, surr, half):
    """Returns (pruned fraction, index of the returned minimiser)."""
    incumbent, best, pruned = np.inf, -1, 0
    for j in order:
        if surr[j] - half[j] > incumbent:      # certifiably cannot improve
            pruned += 1
            continue
        v = true[j]                            # exact evaluation
        if v < incumbent:
            incumbent, best = v, j
    return pruned / len(order), best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--casdatasets", required=True)
    ap.add_argument("-N", type=int, default=20_000)
    ap.add_argument("-B", type=int, default=300)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert_version()

    F = block_bootstrap(yield_changes(a.casdatasets), a.N, a.seed, block=6)
    cands = bond_books(a.B, m=4, K=5, seed=104)
    g = FAMILIES["TVaR(0.99)"]
    rng = np.random.default_rng(a.seed)

    rows = []
    for r in (1, 2, 3, 4):
        true, surr, half = prepare(F, cands, r, g)
        exact_min = int(np.argmin(true))

        rand, correct = [], True
        for _ in range(a.perms):
            p, best = selection_loop(rng.permutation(a.B), true, surr, half)
            rand.append(p); correct &= (true[best] == true[exact_min])

        # Best-first: process in increasing surrogate value, so the incumbent
        # descends immediately. Reverse is the adversarial order.
        bf, best_bf = selection_loop(np.argsort(surr), true, surr, half)
        correct &= (true[best_bf] == true[exact_min])
        rev, best_rev = selection_loop(np.argsort(surr)[::-1], true, surr, half)
        correct &= (true[best_rev] == true[exact_min])

        # Fixed-threshold screening at the population median, for comparison.
        Q = float(np.median(true))
        fixed = float(np.mean([screen(s, h, Q) != "undecided"
                               for s, h in zip(surr, half)]))
        rows.append(dict(r=r, random=float(np.mean(rand)),
                         random_sd=float(np.std(rand)), best_first=bf,
                         reverse=rev, fixed_threshold=fixed,
                         minimiser_exact=bool(correct)))

    print(f"B={a.B} Treasury books, N={a.N}, TVaR(0.99), "
          f"{a.perms} permutations")
    print(f"{'r':>3}{'random':>10}{'best-first':>12}{'reverse':>10}"
          f"{'fixed thr.':>12}{'min exact':>11}")
    for o in rows:
        print(f"{o['r']:>3}{o['random']:>10.3f}{o['best_first']:>12.3f}"
              f"{o['reverse']:>10.3f}{o['fixed_threshold']:>12.3f}"
              f"{str(o['minimiser_exact']):>11}")
    dump("../results/table3_selection_loop.json", rows,
         dict(dataset="US Treasury key rates", sigma="TVaR(0.99)",
              N=a.N, B=a.B, seed=a.seed, permutations=a.perms,
              m_holdings=4, K=5, ranks=[1, 2, 3, 4],
              fixed_threshold="population median"))
