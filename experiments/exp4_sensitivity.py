"""Bootstrap sensitivity of the catastrophe dependence fit (supplementary table).

The five-peril dependence structure is estimated from only 48 annual
aggregates.  We bootstrap the *fitting step*: resample the 48 year-rows with
replacement, refit the zero-inflated lognormal marginals and the log-scale
Gaussian copula, regenerate the catalogue, and recompute the certified
fraction `f` and band fraction `beta`.  The candidate population is held fixed
across replicates so the spread isolates catalogue uncertainty.

The headline is not the spread but the absence of violations: the certified
bracket is valid for whatever catalogue is supplied, so misspecifying the fit
degrades efficiency, never correctness.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, certified_band,
                      screen)
from certproj.data import annual_peril_aggregates, fit_and_simulate, PERILS
from certproj.populations import treaties

ALPHA, K = 0.99, 4


def metrics(F, cands, r, g, alpha=ALPHA):
    proj = Projection(F, r)
    w = weights(len(F), g)
    m = int(np.ceil((1 - alpha) * len(F)))
    Ce = proj.constant(g)
    true, surr, half, band = [], [], [], []
    for c in cands:
        P, Pt = c.evaluate(F), proj.surrogate(c)
        gam = proj.gamma(c)
        true.append(rho(P, w)); surr.append(rho(Pt, w)); half.append(gam * Ce)
        band.append(certified_band(Pt, gam * proj.enorm, m).mean())
    true, surr, half = map(np.array, (true, surr, half))
    # Count per candidate, so the reported total means what the text says.
    violated = int(np.sum(np.abs(true - surr) > half + 1e-9))
    Q = np.median(true)
    f = float(np.mean([screen(s, h, Q) != "undecided"
                       for s, h in zip(surr, half)]))
    return f, float(np.median(band)), violated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--casdatasets", required=True)
    ap.add_argument("-N", type=int, default=20_000)
    ap.add_argument("-B", type=int, default=200)
    ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()

    agg = annual_peril_aggregates(a.casdatasets)
    print(f"real annual aggregates: {agg.shape[0]} years x {agg.shape[1]} perils")
    g = FAMILIES["TVaR(0.99)"]
    cands = treaties(a.B, K=K, seed=0)
    iu = np.triu_indices(len(PERILS), 1)

    Fref, Cref = fit_and_simulate(agg, np.random.default_rng(1), a.N)
    print(f"reference log-correlation range: "
          f"[{Cref[iu].min():+.3f}, {Cref[iu].max():+.3f}]")
    ref = {r: metrics(Fref, cands, r, g)[:2] for r in (1, 2, 3, 4)}

    res = {r: {"f": [], "b": []} for r in (1, 2, 3, 4)}
    violations = 0
    for rep in range(a.reps):
        rng = np.random.default_rng(1000 + rep)
        idx = rng.integers(0, len(agg), size=len(agg))
        Fb, _ = fit_and_simulate(agg[idx], rng, a.N)
        for r in (1, 2, 3, 4):
            f, b, v = metrics(Fb, cands, r, g)
            res[r]["f"].append(f); res[r]["b"].append(b); violations += v

    print(f"\n{'r':>3}{'ref f':>8}{'mean':>8}{'s.d.':>7}{'5%':>8}{'95%':>8}"
          f"{'min':>7}{'beta 5-95%':>20}")
    for r in (1, 2, 3, 4):
        fa, ba = np.array(res[r]["f"]), np.array(res[r]["b"])
        print(f"{r:>3}{ref[r][0]:>8.3f}{fa.mean():>8.3f}{fa.std():>7.3f}"
              f"{np.percentile(fa,5):>8.3f}{np.percentile(fa,95):>8.3f}"
              f"{fa.min():>7.3f}"
              f"   [{np.percentile(ba,5):.4f}, {np.percentile(ba,95):.4f}]")
    print(f"\nbracket violations across {a.reps} x 4 x {a.B} = "
          f"{a.reps * 4 * a.B} candidate evaluations: {violations}")
