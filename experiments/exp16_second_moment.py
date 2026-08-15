"""Rejected refinement (2): the Gaussian-calibrated second-moment envelope.

RECONSTRUCTED FROM THE MANUSCRIPT PROSE.  Section 8 states that this refinement
"replaces the worst-case envelope by a second-moment bound calibrated on
Gaussian factors" and "fails on real data".  No script in the archive computes
it, so the estimator below is a reconstruction of that description and must be
checked against whatever was originally run before its output is quoted.

The construction.  The deterministic certificate bounds the pointwise deviation
by the Cauchy-Schwarz envelope

    |P_j(F_i) - Ptil_j(z_i)|  <=  gamma_j(U) * ||e_i||,          (worst case)

which is attained only when the residual e_i aligns with the steepest orthogonal
slope.  Under a Gaussian factor law that alignment is rare: b' e is centred
normal with variance b' Sigma_perp b, where Sigma_perp = Pi^perp Sigma_F
Pi^perp, so a two-sided level-eta bound is

    delta_j^{2m} = z_{1-eta/2} * max_t sqrt( b_jt' Sigma_perp b_jt ),

a constant per candidate rather than a per-scenario envelope.  Because a
spectral risk measure maps a constant to itself, the resulting half-width is
h_j^{2m} = delta_j^{2m}.  This is far narrower than gamma_j ||e|| -- and it is
probabilistic, so it can fail.

Coverage is realised, not nominal: the fraction of candidates whose true
deviation |rho(P_j) - rho(Ptil_j)| falls inside h_j^{2m}.  Scenario-level
coverage is reported alongside it, since the two can diverge sharply when the
factor law is heavy-tailed.  Calibration is Gaussian throughout; the catalogues
are not, which is the point of the experiment.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, assert_version, dump)
from certproj.data import (yield_changes, block_bootstrap,
                           catastrophe_catalogue)
from certproj.populations import bond_books, treaties

ALPHA = 0.99
ETA = 0.99                      # nominal coverage the bound is calibrated to


def gaussian_z(eta):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + eta / 2.0))


def second_moment(F, cands, r, g, eta=ETA):
    """Realised coverage of the Gaussian-calibrated second-moment envelope."""
    proj = Projection(F, r)
    w = weights(len(F), g)
    z = gaussian_z(eta)

    # Residual covariance.  proj.E is the residual Pi_U^perp (F - mu) the
    # certificate itself uses, so its covariance is Pi^perp Sigma_F Pi^perp
    # without forming the projector separately.
    Sig_perp = np.cov(proj.E, rowvar=False)

    cand_hits, scen_hits, scen_tot, ratios = 0, 0, 0, []
    for c in cands:
        P, Pt = c.evaluate(F), proj.surrogate(c)
        sd = np.sqrt(np.maximum(np.einsum("td,de,te->t", c.b, Sig_perp, c.b), 0.0))
        h2m = z * float(sd.max())                       # constant per candidate

        # candidate level: does the bound contain the risk-measure deviation?
        dev = abs(rho(P, w) - rho(Pt, w))
        cand_hits += int(dev <= h2m)

        # scenario level: does it contain the pointwise deviation?
        pw = np.abs(P - Pt)
        scen_hits += int((pw <= h2m).sum())
        scen_tot += len(pw)

        ratios.append(h2m / max(proj.gamma(c) * float(proj.enorm.max()), 1e-12))

    return dict(r=r,
                coverage_candidate=cand_hits / len(cands),
                coverage_scenario=scen_hits / scen_tot,
                median_narrowing=float(np.median(ratios)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--casdatasets", required=True)
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert_version()
    g = FAMILIES["TVaR(0.99)"]

    datasets = []
    F_tr = block_bootstrap(yield_changes(a.casdatasets), a.N, a.seed, block=6)
    datasets.append(("Treasury m=4", F_tr, bond_books(400, 4, K=5, seed=104)))
    datasets.append(("Treasury m=8", F_tr, bond_books(400, 8, K=5, seed=108)))
    F_cat, _ = catastrophe_catalogue(a.casdatasets, a.N, a.seed)
    datasets.append(("Catastrophe", F_cat, treaties(300, K=4, seed=0)))

    print(f"Gaussian-calibrated second-moment envelope, nominal coverage "
          f"{ETA:.0%}, N={a.N}")
    print(f"{'dataset':<14}{'r':>3}{'cand cover':>12}{'scen cover':>12}"
          f"{'vs worst case':>15}")
    rows, cover = [], []
    for label, F, cands in datasets:
        for r in (1, 2, 3, 4):
            o = second_moment(F, cands, r, g)
            o["dataset"] = label
            rows.append(o)
            cover.append(o["coverage_candidate"])
            print(f"{label:<14}{r:>3}{o['coverage_candidate']:>12.3f}"
                  f"{o['coverage_scenario']:>12.3f}"
                  f"{o['median_narrowing']:>14.3f}x")
    print(f"\nrealised candidate coverage ranges over "
          f"[{min(cover):.3f}, {max(cover):.3f}] against a nominal {ETA:.2f}")
    dump("../results/exp16_second_moment.json", rows,
         dict(nominal_coverage=ETA, N=a.N, seed=a.seed, sigma="TVaR(0.99)",
              calibration="Gaussian",
              note="reconstruction of rejected refinement (2); confirm the "
                   "estimator against the original before quoting"))
