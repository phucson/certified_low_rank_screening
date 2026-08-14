"""Limitations and rejected refinements, and the pilot-tightened certificate.

Three attempts to close the gap between worst-case sharpness and typical
conservatism, all reported because their failure localises the residual slack:

  (1) remove the coherence step by evaluating rho(Ptil +- gamma||e||) directly;
  (2) a second-moment envelope calibrated on Gaussian factors (rejected: it is
      probabilistic, and its realised coverage on real factors is erratic);
  (3) the tight envelope rho(D_j), deployed only on candidates the nominal
      bracket leaves unresolved.

Also evaluates the pilot-tightened Clopper-Pearson certificate: a small pilot
of true-loss evaluations on unresolved candidates only, certifying a quantile
of the pointwise deviation at simultaneous confidence.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, screen,
                      tight_envelope, assert_version, dump)
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books

ALPHA = 0.99


def clopper_pearson_upper(k, n, delta):
    """Upper Clopper-Pearson limit for a binomial proportion."""
    from scipy.stats import beta as beta_dist
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("-r", type=int, default=2)
    ap.add_argument("--pilot", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    F = synthetic_catalogue(a.N, seed=a.seed)
    cands = bond_books(a.B, m=4, K=5, seed=a.seed)
    g = FAMILIES["TVaR(0.99)"]
    proj = Projection(F, a.r)
    w = weights(a.N, g)
    Ce = proj.constant(g)
    rng = np.random.default_rng(a.seed)

    true = np.array([rho(c.evaluate(F), w) for c in cands])
    surr = np.array([rho(proj.surrogate(c), w) for c in cands])
    gam = np.array([proj.gamma(c) for c in cands])
    half = gam * Ce
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"

    Q = float(np.median(true))
    labels = [screen(s, h, Q) for s, h in zip(surr, half)]
    f_nom = float(np.mean([l != "undecided" for l in labels]))
    unresolved = [j for j, l in enumerate(labels) if l == "undecided"]
    print(f"synthetic population, r={a.r}, B={a.B}, N={a.N}")
    print(f"nominal certified fraction f = {f_nom:.3f}, "
          f"{len(unresolved)} unresolved\n")

    # (1) Coherence-free variant: bracket the surrogate shifted by the envelope,
    # which avoids the subadditivity step but keeps Cauchy-Schwarz.
    ratios = []
    for j in unresolved[:100]:
        d = gam[j] * proj.enorm
        hi = rho(proj.surrogate(cands[j]) + d, w)
        lo = rho(proj.surrogate(cands[j]) - d, w)
        ratios.append(half[j] / max((hi - lo) / 2, 1e-12))
    print(f"(1) coherence-free: recovers a factor "
          f"{np.min(ratios):.2f}-{np.max(ratios):.2f} "
          f"(median {np.median(ratios):.2f})")

    # (3) Tight envelope on unresolved candidates only.
    tight_h, recert = {}, 0
    for j in unresolved:
        h, _ = tight_envelope(proj, cands[j], g)
        tight_h[j] = h
        if screen(surr[j], h, Q) != "undecided":
            recert += 1
    gain = float(np.mean([half[j] / max(tight_h[j], 1e-12)
                          for j in unresolved]))
    f_tight = f_nom + recert / a.B
    print(f"(3) tight envelope: re-certifies {recert} of {len(unresolved)} "
          f"({recert / max(len(unresolved), 1):.1%}), lifting f from "
          f"{f_nom:.3f} to {f_tight:.3f}; mean tightening {gain:.2f}x")

    # Pilot-tightened Clopper-Pearson certificate on unresolved candidates.
    delta = (1 - a.conf) / max(len(unresolved), 1)      # Bonferroni
    valid, ratio_to_tight = 0, []
    for j in unresolved:
        idx = rng.integers(0, a.N, size=a.pilot)        # with replacement
        Pj = cands[j].evaluate(F[idx])
        Ptj = proj.surrogate(cands[j])[idx]
        dev = np.abs(Pj - Ptj)
        q = float(np.quantile(dev, 0.99))
        k = int((dev > q).sum())
        cp = clopper_pearson_upper(k, a.pilot, delta)
        h_pilot = q + cp * gam[j] * float(proj.enorm.max())
        full_dev = abs(true[j] - surr[j])
        valid += bool(full_dev <= h_pilot + 1e-9)
        ratio_to_tight.append(h_pilot / max(tight_h[j], 1e-12))
    print(f"    pilot certificate: valid on {valid}/{len(unresolved)} at "
          f"{a.conf:.0%} simultaneous confidence; median ratio to the "
          f"deterministic tight envelope {np.median(ratio_to_tight):.2f}")

    rows = dict(f_nominal=f_nom, unresolved=len(unresolved),
                coherence_free_factor=[float(np.min(ratios)),
                                       float(np.max(ratios))],
                tight_recertified=recert, f_after_tight=f_tight,
                tight_mean_gain=gain, pilot_valid=valid,
                pilot_median_ratio=float(np.median(ratio_to_tight)))
    dump("../results/rejected_refinements.json", rows,
         dict(population="synthetic (S.4)", N=a.N, B=a.B, r=a.r,
              seed=a.seed, pilot=a.pilot, confidence=a.conf,
              sigma="TVaR(0.99)", threshold="population median"))
